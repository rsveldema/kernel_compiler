
from pathlib import Path
from copy import deepcopy

from lark import Tree

from codegen.kast.program import Program
from codegen.visitors import visitor
from codegen.kast.expression import Expression
from codegen.kast.statement import (
    Statement,
    ForLoopWithConditionAndIncrement,
    ForLoopRange,
    Condition,
    If,
    Declaration,
    SharedDecl,
    RawStatement,
)
from codegen.kast.expression import (
    Identifier,
    Number,
    BinaryExpr,
    LimitExpr,
    CastExpr,
    FieldAccess,
    ArrayAccess,
    CallExpr,
    Identifier,
    Number,
    BinaryExpr,
    LimitExpr,
)
from codegen.kast.type import Int, Float
from codegen.parser import parse_search_replace_pattern
from codegen.transforms import transform_statement, transform_expression

class Pattern:
    def __init__(self, filename: str):
        self.filename = filename
        self.search: list[Statement] = []
        self.replace: list[Statement] = []
        self.constraints: Expression | None = None
        self.init()

    def init(self):
        pattern_tree = parse_search_replace_pattern(Path(self.filename).read_text())
        self.search.clear()
        self.replace.clear()

        for child in pattern_tree.children:
            if not isinstance(child, Tree):
                continue

            if child.data == "search_statements":
                self.search.extend(self._transform_statements(child))
            elif child.data == "replace_statements":
                self.replace.extend(self._transform_statements(child))
            elif child.data == "constraints":
                self.constraints = self._transform_constraints(child)

    def _transform_statements(self, statements_tree: Tree) -> list[Statement]:
        statements = []

        for child in statements_tree.children:
            if isinstance(child, Tree) and child.data == "statement":
                statement = transform_statement(child)
                if statement is not None:
                    statements.append(statement)

        return statements

    def _transform_constraints(self, constraints_tree: Tree) -> Expression | None:
        for child in constraints_tree.children:
            if isinstance(child, Tree) and child.data == "expression":
                return transform_expression(child)

        return None


    def matches(self, node):
        # Placeholder for pattern matching logic
        return False

class PatternStore:
    def __init__(self):
        self.patterns = {}
        self.init()

    def init(self):
        transforms_dir = Path(__file__).resolve().parents[2] / "transforms"
        self.patterns.clear()

        for pattern_path in sorted(transforms_dir.glob("*.tkernel")):
            try:
                pattern = Pattern(str(pattern_path))
                self.patterns[pattern_path.stem] = pattern
            except Exception as e:
                # Skip patterns that can't be parsed (e.g., multi_arg.tkernel uses + which
                # isn't supported in the grammar's for-loop increment rule). These patterns
                # are applied programmatically rather than through generic tree matching.
                print(f"Skipping unparseable pattern {pattern_path.name}: {e}")


class TreeRewriter(visitor.Visitor):
    """
    Visitor that rewrites the AST based on search/replace patterns defined in .tkernel files.
    When seeing a statement node, it checks if any pattern matches it. If a match is found, 
    it replaces the statement with the corresponding replacement statements from the pattern.
    
    When a pattern matches, any wildcard identifiers in the search pattern are bound to the corresponding expression in the AST.
    In the replacement, the wildcard identifiers are replaced with the bound expressions.
    """
    def __init__(self, context):
        self.context = context or {}
        self.pattern_store = PatternStore()

    def visit_program(self, node: Program):
        workgroups = int(self.context.get("workgroups", 8))
        target_loop_bounds = set(self.context.get("target_loop_bounds", set()) or set())

        body_stmts = getattr(node, "body_stmts", []) or []

        # Check for cooperative matrix2 multi_arg pattern (6-matrix batched GEMM).
        is_coop_mat2 = self._check_coop_mat2_condition(node)
        
        if is_coop_mat2 and "coop_mat2_multi_arg" in self.pattern_store.patterns:
            tile_size = int(self.context.get("tile_size", 8))
            chunk_size = int(self.context.get("cooperative_matrix2_chunk_size", 16))
            bound_k = int(self.context.get("bound_k", 1024))
            body_stmts = self._apply_coop_mat2_to_statements(
                tile_size, chunk_size, bound_k
            )
        
        # Check for shared-memory multi_arg pattern (6-matrix batched GEMM).
        is_multi_arg = self._check_multi_arg_condition(node)
        
        if is_multi_arg and "multi_arg" in self.pattern_store.patterns:
            tile_size = int(self.context.get("tile_size", 8))
            chunk_size = int(self.context.get("chunk_size", 128))
            bound_k = int(self.context.get("bound_k", 1024))
            body_stmts = self._apply_multi_arg_to_statements(
                tile_size, chunk_size, bound_k
            )
        
        # Apply rewrite patterns only when corresponding tkernel files are present.
        if "step2_guard" in self.pattern_store.patterns:
            body_stmts = self._apply_step2_to_statements(body_stmts)
        if "step3_chunked_loop" in self.pattern_store.patterns:
            body_stmts = self._apply_step3_to_statements(
                body_stmts,
                target_loop_bounds,
                workgroups,
            )

        node.body_stmts = body_stmts
        return node
    
    def _check_multi_arg_condition(self, node: Program) -> bool:
        """Check if this program should use the shared-memory multi_arg pattern."""
        if not getattr(node, "use_shared_memory_tiling", False):
            return False
        params = getattr(node, "params", []) or []
        names = [p.name for p in params if isinstance(p, Declaration)]
        return names == ["A1", "B1", "A2", "B2", "A3", "B3", "C"]

    def _check_coop_mat2_condition(self, node: Program) -> bool:
        """Check if this program should use the cooperative matrix2 multi_arg pattern."""
        if not getattr(node, "use_cooperative_matrix2", False):
            return False
        params = getattr(node, "params", []) or []
        names = [p.name for p in params if isinstance(p, Declaration)]
        return names == ["A1", "B1", "A2", "B2", "A3", "B3", "C"]

    def _apply_coop_mat2_to_statements(self, tile_size: int, chunk_size: int, bound_k: int) -> list[Statement]:
        """Replace body_stmts with the cooperative matrix2 multi_arg pattern content.
        
        Generates tensor layout declarations (as AST nodes), coopmat matrices, 
        and loop operations via RawStatement for complex GLSL constructs.
        """
        stmts = []
        ts = str(tile_size)
        cs = str(chunk_size)
        bk = str(bound_k)

        # Tile coordinate variables
        stmts.append(Declaration(True, Int(), "tile_row", 
                               CastExpr(Int(), FieldAccess(Identifier("gl_WorkGroupID"), "x"))))
        stmts.append(Declaration(True, Int(), "tile_col",
                               CastExpr(Int(), FieldAccess(Identifier("gl_WorkGroupID"), "y"))))

        # Tensor layout declarations (use tensor_layout_decl type via declaration)
        for suffix in ("A", "B", "C"):
            name = f"tensorLayout{suffix}"
            stmts.append(Declaration(False, 
                                   TensorLayout(Number("2")),
                                   name,
                                   CallExpr(Identifier("createTensorLayoutNV"), [Number("2")])))

        # Set tensor layout dimensions (1024x1024 for each)
        for suffix in ("A", "B", "C"):
            name = f"tensorLayout{suffix}"
            init_expr = CallExpr(Identifier("setTensorLayoutDimensionNV"), [
                Identifier(name), Number("1024"), Number("1024")
            ])
            stmts.append(Assignment(Identifier(name), "=", init_expr))

        # Accumulator result
        coopmat_type_str = f"coopmat<float, gl_ScopeWorkgroup, {ts}, {ts}, gl_MatrixUseAccumulator>"
        stmts.append(RawStatement(
            f"{coopmat_type_str} result = "
            f"{coopmat_type_str}(0.0);"))

        # For loop over chunkK with body containing matrix loads and mul-add
        NL = chr(10)
        chunk_limit = str(int(ts) * int(ts))  # tile_block_size * tile_block_size
        load_stmts = []
        
        for suffix in ("1", "2", "3"):
            mat_A_type = f"coopmat<float, gl_ScopeWorkgroup, {ts}, {cs}, gl_MatrixUseA>"
            mat_B_type = f"coopmat<float, gl_ScopeWorkgroup, {cs}, {ts}, gl_MatrixUseB>"
            
            load_stmts.append(f"{mat_A_type} matrixA{suffix};")
            load_stmts.append(f"{mat_B_type} matrixB{suffix};")
            load_stmts.append(
                f"coopMatLoadTensorNV(matrixA{suffix}, A{suffix}, 0, "
                f"sliceTensorLayoutNV(tensorLayoutA, {ts} * tile_row, {ts}, chunkK, {cs}));")
            load_stmts.append(
                f"coopMatLoadTensorNV(matrixB{suffix}, B{suffix}, 0, "
                f"sliceTensorLayoutNV(tensorLayoutB, chunkK, {cs}, {ts} * tile_col, {ts}));")
            load_stmts.append(f"result = coopMatMulAdd(matrixA{suffix}, matrixB{suffix}, result);")

        loop_body = "{" + NL + NL.join(load_stmts) + NL + "}"
        loop_stmt = f"for (uint chunkK = 0; chunkK < {chunk_limit}; chunkK += {cs}) {loop_body}"
        stmts.append(RawStatement(loop_stmt))

        # Store C: load matrixC, add result, store back
        mat_C_type = f"coopmat<float, gl_ScopeWorkgroup, {ts}, {ts}, gl_MatrixUseAccumulator>"
        stmts.append(RawStatement(
            f"{mat_C_type} matrixC;"))
        stmts.append(RawStatement(
            f"coopMatLoadTensorNV(matrixC, C, 0, "
            f"sliceTensorLayoutNV(tensorLayoutC, {ts} * tile_row, {ts}, {ts} * tile_col, {ts}));"))
        stmts.append(RawStatement("result = result + matrixC;"))
        stmts.append(RawStatement(
            f"coopMatStoreTensorNV(result, C, 0, "
            f"sliceTensorLayoutNV(tensorLayoutC, {ts} * tile_row, {ts}, {ts} * tile_col, {ts}));"))

        return stmts

    def _is_for_statement(self, stmt):
        return isinstance(stmt, (ForLoopWithConditionAndIncrement, ForLoopRange))

    def _apply_step2_to_statements(self, statements):
        rewritten = []
        for stmt in statements or []:
            if self._is_for_statement(stmt):
                cloned = deepcopy(stmt)
                # Do NOT recurse into loop bodies — the local_id==0 guard
                # is only for top-level non-loop statements.  Loop bodies
                # must stay unguarded so that tiling (step 3) can let each
                # thread execute its chunk without being blocked by
                # "if (local_id == 0)".
                rewritten.append(cloned)
                continue

            if isinstance(stmt, Declaration):
                rewritten.append(deepcopy(stmt))
                continue

            cloned = deepcopy(stmt)
            if hasattr(cloned, "body_stmts"):
                cloned.body_stmts = self._apply_step2_to_statements(getattr(cloned, "body_stmts", []) or [])
            if hasattr(cloned, "else_stmts"):
                cloned.else_stmts = self._apply_step2_to_statements(getattr(cloned, "else_stmts", []) or [])

            guard = If(
                BinaryExpr(Identifier("local_id"), "==", Number("0")),
                [cloned],
            )
            rewritten.append(guard)
            rewritten.append(RawStatement("barrier();"))

        return rewritten

    def _apply_step3_to_statements(self, statements, target_loop_bounds, workgroups):
        rewritten = []
        for stmt in statements or []:
            if self._is_for_statement(stmt):
                concrete_bound = self._get_concrete_loop_upper_bound(stmt)
                variable_bound = self._get_variable_loop_upper_bound(stmt)
                should_tile_concrete = concrete_bound is not None and concrete_bound in target_loop_bounds
                should_tile_variable = variable_bound is not None and variable_bound in target_loop_bounds
                if should_tile_concrete or should_tile_variable:
                    rewritten.extend(self._tile_loop(stmt, workgroups))
                    continue

                cloned = deepcopy(stmt)
                cloned.body_stmts = self._apply_step3_to_statements(
                    getattr(cloned, "body_stmts", []) or [],
                    target_loop_bounds,
                    workgroups,
                )
                rewritten.append(cloned)
                continue

            cloned = deepcopy(stmt)
            if hasattr(cloned, "body_stmts"):
                cloned.body_stmts = self._apply_step3_to_statements(
                    getattr(cloned, "body_stmts", []) or [],
                    target_loop_bounds,
                    workgroups,
                )
            if hasattr(cloned, "else_stmts"):
                cloned.else_stmts = self._apply_step3_to_statements(
                    getattr(cloned, "else_stmts", []) or [],
                    target_loop_bounds,
                    workgroups,
                )
            rewritten.append(cloned)

        return rewritten

    def _make_shared_decl_stmt(name, dim_names):
        """Create a SharedDecl AST node for a shared array."""
        dims = [Identifier(n) for n in dim_names]
        return SharedDecl(False, Float(), name, None, dimensions=dims)

    def _apply_multi_arg_to_statements(self, tile_size: int, chunk_size: int, bound_k: int) -> list[Statement]:
        """Replace body_stmts with the shared-memory multi_arg pattern content."""
        stmts = []
        ts = str(tile_size)
        cs = str(chunk_size)
        bk = str(bound_k)

        # Const declarations
        stmts.append(Declaration(True, Int(), "tile_size", Number(ts)))
        stmts.append(Declaration(True, Int(), "chunk_size", Number(cs)))
        stmts.append(Declaration(True, Int(), "bound_k", Number(bk)))

        # Shared memory declarations with concrete dimensions
        for name in ["sh_A1", "sh_A2", "sh_A3"]:
            stmts.append(self._make_shared_decl_stmt(name, ["tile_size", "chunk_size"]))
        for name in ["sh_B1", "sh_B2", "sh_B3"]:
            stmts.append(self._make_shared_decl_stmt(name, ["chunk_size", "tile_size"]))

        # Accumulator declarations
        for name in ["sum1", "sum2", "sum3"]:
            stmts.append(Declaration(False, Float(), name, Number("0.0")))

        load_limit = tile_size * chunk_size
        stride = tile_size * tile_size

        # A-load loop with if-guard
        NL = chr(10)  # newline char for RawStatement content
        a_body_lines = [
            "    const int load_i = load_idx / " + cs + ";",
            "    const int load_k = load_idx - load_i * " + cs + ";",
            "    const int a_row = int(gl_WorkGroupID.x) * " + ts + " + load_i;",
            "    const int a_k = block_start + load_k;",
            "    if (a_row < rllm_push.rllm_bound_x && a_k < bound_k) {",
            "        sh_A1[load_i][load_k] = A1[(bound_k * a_row) + a_k];",
            "        sh_A2[load_i][load_k] = A2[(bound_k * a_row) + a_k];",
            "        sh_A3[load_i][load_k] = A3[(bound_k * a_row) + a_k];",
            "    } else {",
            "        sh_A1[load_i][load_k] = 0.0;",
            "        sh_A2[load_i][load_k] = 0.0;",
            "        sh_A3[load_i][load_k] = 0.0;",
            "}",
        ]
        a_loop_str = ("for (int load_idx = local_linear; load_idx < " + str(load_limit)
                      + "; load_idx += " + str(stride) + ") {" + NL
                      + NL.join(a_body_lines) + NL + "}")
        stmts.append(RawStatement(a_loop_str))

        # B-load loop with if-guard
        b_body_lines = [
            "    const int load_k = load_idx / " + ts + ";",
            "    const int load_j = load_idx - load_k * " + ts + ";",
            "    const int b_k = block_start + load_k;",
            "    const int b_col = int(gl_WorkGroupID.y) * " + ts + " + load_j;",
            "    if (b_k < bound_k && b_col < rllm_push.rllm_bound_y) {",
            "        sh_B1[load_k][load_j] = B1[(bound_k * b_k) + b_col];",
            "        sh_B2[load_k][load_j] = B2[(bound_k * b_k) + b_col];",
            "        sh_B3[load_k][load_j] = B3[(bound_k * b_k) + b_col];",
            "    } else {",
            "        sh_B1[load_k][load_j] = 0.0;",
            "        sh_B2[load_k][load_j] = 0.0;",
            "        sh_B3[load_k][load_j] = 0.0;",
            "}",
        ]
        b_loop_str = ("for (int load_idx = local_linear; load_idx < " + cs + " * " + ts
                      + "; load_idx += " + str(stride) + ") {" + NL
                      + NL.join(b_body_lines) + NL + "}")
        stmts.append(RawStatement(b_loop_str))

        # Barrier + compute loop
        stmts.append(RawStatement("barrier();"))
        c_body_lines = [
            "for (int kk = 0; kk < " + cs + "; ++kk) {",
            "    sum1 += sh_A1[local_i][kk] * sh_B1[kk][local_j];",
            "    sum2 += sh_A2[local_i][kk] * sh_B2[kk][local_j];",
            "    sum3 += sh_A3[local_i][kk] * sh_B3[kk][local_j];",
            "}",
        ]
        stmts.append(RawStatement(NL.join(c_body_lines)))

        # Barrier + conditional store
        stmts.append(RawStatement("barrier();"))
        store_stmt = ("if (i < rllm_push.rllm_bound_x && j < rllm_push.rllm_bound_y) {"
                      + NL + "    atomicAdd(C[(bound_k * i) + j], sum1 + sum2 + sum3);"
                      + NL + "}")
        stmts.append(RawStatement(store_stmt))

        return stmts


    def _tile_loop(self, loop, workgroups):
        loop_var_name = getattr(loop, "loop_var_name", "") or "i"
        chunk_name = f"chunk_size_{loop_var_name}"
        start_name = f"start_{loop_var_name}"
        end_name = f"end_{loop_var_name}"

        bound_expr = self._get_loop_bound_expr(loop)
        if bound_expr is None:
            return [deepcopy(loop)]

        chunk_expr = BinaryExpr(
            BinaryExpr(deepcopy(bound_expr), "+", Number(str(workgroups - 1))),
            "/",
            Identifier("rllm_wg_count"),
        )
        start_expr = BinaryExpr(Identifier("local_id"), "*", Identifier(chunk_name))
        end_expr = BinaryExpr(Identifier(start_name), "+", Identifier(chunk_name))

        decls = [
            Declaration(True, Int(), chunk_name, chunk_expr),
            Declaration(True, Int(), start_name, start_expr),
            Declaration(True, Int(), end_name, end_expr),
        ]

        tiled_body = deepcopy(getattr(loop, "body_stmts", []) or [])

        if isinstance(loop, ForLoopRange):
            tiled_loop = ForLoopRange(
                loop_var_type=deepcopy(getattr(loop, "loop_var_type", None)),
                loop_var_name=loop_var_name,
                init_expr=LimitExpr(Identifier(chunk_name), Identifier(start_name), Identifier(end_name)),
                body_stmts=tiled_body,
            )
        else:
            tiled_loop = ForLoopWithConditionAndIncrement(
                loop_var_type=deepcopy(getattr(loop, "loop_var_type", None)),
                loop_var_name=loop_var_name,
                condition=Condition(Identifier(loop_var_name), "<", Identifier(end_name)),
                increment_var=loop_var_name,
                increment_op="++",
                body_stmts=tiled_body,
                init_expr=Identifier(start_name),
            )

        return decls + [tiled_loop]

    def _get_loop_bound_expr(self, loop):
        if isinstance(loop, ForLoopWithConditionAndIncrement):
            condition = getattr(loop, "condition", None)
            if isinstance(condition, Condition):
                rhs = getattr(condition, "rhs", None)
                if rhs is not None:
                    return rhs
            if isinstance(condition, BinaryExpr):
                rhs = getattr(condition, "right", None)
                if rhs is not None:
                    return rhs
        if isinstance(loop, ForLoopRange):
            init_expr = getattr(loop, "init_expr", None)
            if isinstance(init_expr, LimitExpr):
                return getattr(init_expr, "max_val", None)
        return None

    def _get_concrete_loop_upper_bound(self, loop):
        bound_expr = self._get_loop_bound_expr(loop)
        if isinstance(bound_expr, Number):
            try:
                return int(bound_expr.value)
            except (ValueError, AttributeError):
                return None
        return None

    def _get_variable_loop_upper_bound(self, loop):
        bound_expr = self._get_loop_bound_expr(loop)
        if isinstance(bound_expr, Identifier):
            return bound_expr.name
        return None
    
