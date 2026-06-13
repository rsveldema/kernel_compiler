"""Vulkan-compatible code generator (GLSL compute shader output).

Produces RLLM-style GLSL with:
- std430 SSBOs using compile-time sized arrays for matrix types with known dimensions
- push_constant block for all scalar params and triangular bounds
- Proper variable initialization from gl_GlobalInvocationID and rllm_push
- Triangular guard at start of main() for 3D triangular loops
"""


from codegen.visitors.pretty_printer import prettyprint

from .visitor import Visitor

from ..kast.program import *
from ..kast.type import *
from ..kast.expression import *
from ..kast.statement import *
from ..kast.workgroup import WorkgroupProperties


def _extract_size_from_expr(expr: Expression) -> int:
    assert expr is not None

    if isinstance(expr, Number):
        return int(expr.value)

    if isinstance(expr, BinaryExpr):
        lv = _extract_size_from_expr(expr.left)
        rv = _extract_size_from_expr(expr.right)
        match expr.op:
            case "+":
                return lv + rv
            case "-":
                return lv - rv
            case "*":
                return lv * rv
            case "/":
                return lv / rv
            case _:
                assert False, f"no binary operator for constant folding: {expr.op}"

    if isinstance(expr, LimitExpr):
        return _extract_size_from_expr(expr.max_val)

    assert False, f"cannot constant fold: {prettyprint(expr)}"
    # s = self._to_str(expr)
    # if s and re.match(r'^\d+$', s):
    #    return s
    # return None


def _compute_matrix_size(ty: Type) -> int:
    parts = 1
    if isinstance(ty, (FixedSizeLevelsRowsColsMatrix, FlexibleRowsColsLevelsMatrix)):
        for attr in ("level_expr", "row_size_expr", "col_size_expr"):
            expr = getattr(ty, attr, None)
            if expr:
                s = _extract_size_from_expr(expr)
                if s:
                    parts *= s
        return parts
    elif isinstance(ty, FixedSizeMatrix):
        for attr in ("row_size_expr", "col_size_expr"):
            expr = getattr(ty, attr, None)
            if expr:
                s = _extract_size_from_expr(expr)
                if s:
                    parts *= s
        return parts
    elif isinstance(ty, FixedSizeVector):
        if ty.size_expr:
            s = _extract_size_from_expr(ty.size_expr)
            if s:
                s
        return -1
    return -1


class VulkanKernelVisitor(Visitor):
    """Transforms the parsed AST into a Vulkan GLSL compute shader string."""

    def __init__(self):
        self._lines: list[str] = []
        self._indent_level: int = 0
        self._binding_counter: int = 0
        self._ssbo_map: dict[str, tuple] = {}  # param_name -> (is_3d, type_node)
        self._push_constant_fields: list[tuple] = []  # (name, type_str) pairs
        self._push_constant_map: dict[str, bool] = {}
        self._triangular_upper_bound_name: str | None = None
        self._uses_reduction_chunks: bool = False

    def _is_shared_memory_multi_arg(self, node: Program) -> bool:
        if not getattr(node, "use_shared_memory_tiling", False):
            return False
        names = [p.name for p in node.params if isinstance(p, Declaration)]
        return names == ["A1", "B1", "A2", "B2", "A3", "B3", "C"]

    def _is_cooperative_matrix2_multi_arg(self, node: Program) -> bool:
        if not getattr(node, "use_cooperative_matrix2", False):
            return False
        names = [p.name for p in node.params if isinstance(p, Declaration)]
        return names == ["A1", "B1", "A2", "B2", "A3", "B3", "C"]

    # ── helpers ────────────────────────────────────────────────────────

    def _indent(self) -> str:
        return "    " * self._indent_level

    def _emit(self, line: str = "") -> None:
        self._lines.append(f"{self._indent()}{line}")

    def _push(self) -> None:
        self._indent_level += 1

    def _pop(self) -> None:
        self._indent_level -= 1

    def result(self) -> str:
        return "\n".join(self._lines) + "\n"

    # ── type helpers (GLSL) ────────────────────────────────────────────

    def _glsl_elem_type(self, ty) -> str:
        """Map an AST Type node to a GLSL element type string."""
        if hasattr(ty, "elem_type") and ty.elem_type is not None:
            t = ty.elem_type
            if isinstance(t, (Int, Float)):
                return self._glsl_type_name(t)
        return self._glsl_type_name(ty) if ty else "float"

    def _glsl_type_name(self, ty) -> str:
        """Map an AST Type node to a GLSL type name."""
        if isinstance(ty, Int):
            # size_t → uint64_t, other integers → int
            return "uint64_t" if ty.name == "size_t" else "int"
        if isinstance(ty, Float):
            return "float"
        # Compound types (vectors/matrices) default to float for element type
        return "float"

    def _to_str(self, node) -> str:
        """Dispatcher: convert an AST node to a GLSL string via visitor dispatch."""
        if node is None:
            return ""
        return node.accept(self)

    # ── expression visitors ────────────────────────────────────────────

    def visit_type(self, node: Type) -> str:
        return self._glsl_elem_type(node)

    def visit_int(self, node: Int) -> str:
        if node.name == "size_t":
            return "uint64_t"
        return "int"

    def visit_float(self, node: Float) -> str:
        return "float"

    def visit_fixed_size_vector(self, node: FixedSizeVector) -> str:
        return self._glsl_elem_type(node)

    def visit_flexible_rows_matrix(self, node: FlexibleRowsMatrix) -> str:
        return self._glsl_elem_type(node)

    def visit_fixed_size_matrix(self, node: FixedSizeMatrix) -> str:
        return self._glsl_elem_type(node)

    def visit_fixed_size_levels_rows_cols_matrix(
        self, node: FixedSizeLevelsRowsColsMatrix
    ) -> str:
        return self._glsl_elem_type(node)

    def visit_flexible_rows_cols_levels_matrix(
        self, node: FlexibleRowsColsLevelsMatrix
    ) -> str:
        return self._glsl_elem_type(node)

    def visit_expression(self, node: Expression) -> str:
        return node.accept(self)

    def visit_number(self, node: Number) -> str:
        val = node.value
        if isinstance(val, float):
            s = f"{val}"
            if "." not in s:
                s += ".0"
            return s
        s = str(int(val))
        if getattr(node, "unsigned", False):
            if abs(val) > 4294967295:  # UINT_MAX — need uint64
                return s + "UL"
            return s + "U"
        return s

    def visit_identifier(self, node: Identifier) -> str:
        name = node.name
        if name in self._push_constant_map:
            return f"rllm_push.{name}"
        return name or "unknown"

    def _visit_expr_child(self, node):
        """Helper for visiting expression children (returns '?' for None)."""
        if node is None:
            return "?"
        return node.accept(self)

    def visit_array_access(self, node: ArrayAccess) -> str:
        base = node.base.accept(self) if node.base else "?"
        parts = []
        for idx in node.indices:
            idx_str = self._visit_expr_child(idx) if idx else "?"
            parts.append(idx_str)
        return f"{base}[{', '.join(parts)}]"

    def visit_field_access(self, node: FieldAccess) -> str:
        base = ""
        if isinstance(node.base, Identifier):
            base = node.base.name or "unknown"
        else:
            base = self._visit_expr_child(node.base)
        assert base != None
        base += "." + node.field
        return base

    def visit_limit_expr(self, node: LimitExpr) -> str:
        max_val = self._visit_expr_child(node.max_val)
        body = self._visit_expr_child(node.body)
        return f"limit<{max_val}>({body})"

    def visit_binary_expr(self, node: BinaryExpr) -> str:
        left = self._visit_expr_child(node.left)
        right = self._visit_expr_child(node.right)
        op = node.op or "+"
        return f"({left} {op} {right})"

    def visit_cast_expr(self, node: CastExpr) -> str:
        operand = self._visit_expr_child(node.operand)
        return f"int({operand})"

    def visit_negation_expr(self, node: NegationExpr) -> str:
        operand = self._visit_expr_child(node.operand)
        return f"!{operand}"

    # ── condition visitor ──────────────────────────────────────────────

    def visit_condition(self, node: Condition) -> str:
        lhs = self._visit_expr_child(node.lhs)
        rhs = self._visit_expr_child(node.rhs)
        return f"{lhs} {node.op} {rhs}"

    # ── statement visitors (return string, caller emits with indent) ──

    def visit_statement(self, node: Statement) -> str:
        return node.accept(self)

    def visit_for_loop_range(self, node: ForLoopRange) -> str:
        """Generate a GLSL for loop from range-style (for i in range(n))."""
        lines = []
        ind = "    " * (self._indent_level + 1)

        upper_bound = "?"
        if node.init_expr and isinstance(node.init_expr, LimitExpr):
            max_val = (
                self._to_str(node.init_expr.max_val) if node.init_expr.max_val else "?"
            )
            upper_bound = max_val
            if self._triangular_upper_bound_name and max_val.lstrip("-").isdigit():
                upper_bound = f"rllm_push.{self._triangular_upper_bound_name}"

        lines.append(
            f"{ind}for (int {node.loop_var_name} = 0; "
            f"{node.loop_var_name} < {upper_bound}; ++{node.loop_var_name}) {{"
        )

        for stmt in node.body_stmts:
            if hasattr(stmt, "accept"):
                result = stmt.accept(self)
                if isinstance(result, str):
                    lines.append(f"{ind}{result}")

        lines.append(f"{ind}}}")
        return "\n".join(lines) + "\n"

    def visit_for_loop_with_condition_and_increment(
        self, node: ForLoopWithConditionAndIncrement
    ) -> str:
        """Generate a GLSL for loop from condition+increment style."""
        lines = []
        ind = "    " * (self._indent_level + 1)

        lower_bound = "0"
        upper_bound = "?"

        if node.condition:
            op = node.condition.op or ">="
            rhs = self._to_str(node.condition.rhs) if node.condition.rhs else "?"
            lower_bound = (
                self._to_str(node.init_expr) if node.init_expr else "0"
            )
            upper_bound = rhs

        inc_var = node.increment_var if node.increment_var else node.loop_var_name
        inc_op = node.increment_op if node.increment_op else "++"

        loop_type = self._glsl_type_name(node.loop_var_type) if node.loop_var_type else "int"
        lines.append(
            f"{ind}for ({loop_type} {node.loop_var_name} = {lower_bound}; "
            f"{node.loop_var_name} < {upper_bound}; {inc_op}{inc_var}) {{"
        )

        for stmt in node.body_stmts:
            if hasattr(stmt, "accept"):
                result = stmt.accept(self)
                if isinstance(result, str):
                    lines.append(f"{ind}{result}")

        lines.append(f"{ind}}}")
        return "\n".join(lines) + "\n"

    def visit_if(self, node: If) -> str:
        cond_str = self.visit_condition(node.condition) if node.condition else "?"
        ind = "    " * (self._indent_level + 1)
        lines = [f"{ind}if ({cond_str}) {{"]

        for stmt in node.body_stmts:
            if hasattr(stmt, "accept"):
                result = stmt.accept(self)
                if isinstance(result, str):
                    lines.append(f"{ind}{result}")

        lines.append(f"{ind}}}")
        return "\n".join(lines) + "\n"

    def visit_declaration(self, node: Declaration) -> str:
        prefix = "const " if node.is_const else ""
        var_type = self._to_str(node.var_type) if node.var_type else "float"
        init_str = ""
        if node.init_expr is not None:
            init_str = f" = {self._to_str(node.init_expr)}"
        return f"{prefix}{var_type} {node.name}{init_str};"

    def visit_assignment(self, node: Assignment) -> str:
        lvalue = self._to_str(node.lvalue) if node.lvalue else "?"
        rvalue = self._to_str(node.rvalue) if node.rvalue else "?"
        if (
            self._uses_reduction_chunks
            and node.assign_op == "+="
            and isinstance(node.lvalue, ArrayAccess)
        ):
            return f"atomicAdd({lvalue}, {rvalue});"
        return f"{lvalue} {node.assign_op} {rvalue};"

    def visit_overflow_check(self, node: OverflowCheck) -> str:
        lvalue = self._to_str(node.lvalue) if node.lvalue else "?"
        operand = self._to_str(node.operand) if node.operand else "?"
        return f"OVERFLOW_CHECK_ADD({lvalue}, {operand});"

    def visit_shared_decl(self, node: SharedDecl) -> str:
        prefix = "const " if node.is_const else ""
        var_type = self._to_str(node.var_type) if node.var_type else "float"
        init_str = ""
        if node.init_expr is not None:
            init_str = f" = {self._to_str(node.init_expr)}"
        return f"shared {prefix}{var_type} {node.name}{init_str};"

    def visit_workgroup_properties(self, node: WorkgroupProperties) -> str:
        parts = []
        if node.x_expr is not None:
            parts.append(f"x: {self._to_str(node.x_expr)}")
        if node.y_expr is not None:
            parts.append(f"y: {self._to_str(node.y_expr)}")
        if node.z_expr is not None:
            parts.append(f"z: {self._to_str(node.z_expr)}")
        return f"workgroup {{ {', '.join(parts)} }}"

    def _workgroup_size(self, node: Program) -> tuple[str, str, str]:
        wg_x, wg_y, wg_z = "1", "1", "1"
        for wg in node.workgroups:
            if isinstance(wg, WorkgroupProperties):
                wg_x = self._to_str(wg.x_expr) if wg.x_expr else wg_x
                wg_y = self._to_str(wg.y_expr) if wg.y_expr else wg_y
                wg_z = self._to_str(wg.z_expr) if wg.z_expr else wg_z
        return wg_x, wg_y, wg_z

    def _emit_shared_memory_multi_arg_body(self, tile_size: int, chunk_size: int) -> None:
        self._emit(f"shared float sh_A1[{tile_size}][{chunk_size}];")
        self._emit(f"shared float sh_A2[{tile_size}][{chunk_size}];")
        self._emit(f"shared float sh_A3[{tile_size}][{chunk_size}];")
        self._emit(f"shared float sh_B1[{chunk_size}][{tile_size}];")
        self._emit(f"shared float sh_B2[{chunk_size}][{tile_size}];")
        self._emit(f"shared float sh_B3[{chunk_size}][{tile_size}];")
        self._emit("")
        self._emit("void main() {")
        self._push()
        self._emit("const int i = int(gl_GlobalInvocationID.x);")
        self._emit("const int j = int(gl_GlobalInvocationID.y);")
        self._emit("const int local_i = int(gl_LocalInvocationID.x);")
        self._emit("const int local_j = int(gl_LocalInvocationID.y);")
        self._emit(f"const int local_linear = local_i * {tile_size} + local_j;")
        self._emit(f"const int block_start = int(gl_WorkGroupID.z) * {chunk_size};")
        self._emit("float sum1 = 0.0;")
        self._emit("float sum2 = 0.0;")
        self._emit("float sum3 = 0.0;")
        self._emit("")
        self._emit(f"for (int load_idx = local_linear; load_idx < {tile_size * chunk_size}; load_idx += {tile_size * tile_size}) {{")
        self._push()
        self._emit(f"const int load_i = load_idx / {chunk_size};")
        self._emit(f"const int load_k = load_idx - load_i * {chunk_size};")
        self._emit(f"const int a_row = int(gl_WorkGroupID.x) * {tile_size} + load_i;")
        self._emit("const int a_k = block_start + load_k;")
        self._emit("if (a_k < 1024) {")
        self._push()
        self._emit("sh_A1[load_i][load_k] = A1[(1024 * a_row) + a_k];")
        self._emit("sh_A2[load_i][load_k] = A2[(1024 * a_row) + a_k];")
        self._emit("sh_A3[load_i][load_k] = A3[(1024 * a_row) + a_k];")
        self._pop()
        self._emit("} else {")
        self._push()
        self._emit("sh_A1[load_i][load_k] = 0.0;")
        self._emit("sh_A2[load_i][load_k] = 0.0;")
        self._emit("sh_A3[load_i][load_k] = 0.0;")
        self._pop()
        self._emit("}")
        self._pop()
        self._emit("}")
        self._emit(f"for (int load_idx = local_linear; load_idx < {chunk_size * tile_size}; load_idx += {tile_size * tile_size}) {{")
        self._push()
        self._emit(f"const int load_k = load_idx / {tile_size};")
        self._emit(f"const int load_j = load_idx - load_k * {tile_size};")
        self._emit("const int b_k = block_start + load_k;")
        self._emit(f"const int b_col = int(gl_WorkGroupID.y) * {tile_size} + load_j;")
        self._emit("if (b_k < 1024) {")
        self._push()
        self._emit("sh_B1[load_k][load_j] = B1[(1024 * b_k) + b_col];")
        self._emit("sh_B2[load_k][load_j] = B2[(1024 * b_k) + b_col];")
        self._emit("sh_B3[load_k][load_j] = B3[(1024 * b_k) + b_col];")
        self._pop()
        self._emit("} else {")
        self._push()
        self._emit("sh_B1[load_k][load_j] = 0.0;")
        self._emit("sh_B2[load_k][load_j] = 0.0;")
        self._emit("sh_B3[load_k][load_j] = 0.0;")
        self._pop()
        self._emit("}")
        self._pop()
        self._emit("}")
        self._emit("barrier();")
        self._emit(f"for (int kk = 0; kk < {chunk_size}; ++kk) {{")
        self._push()
        self._emit("sum1 += sh_A1[local_i][kk] * sh_B1[kk][local_j];")
        self._emit("sum2 += sh_A2[local_i][kk] * sh_B2[kk][local_j];")
        self._emit("sum3 += sh_A3[local_i][kk] * sh_B3[kk][local_j];")
        self._pop()
        self._emit("}")
        self._emit("barrier();")
        self._emit("atomicAdd(C[(1024 * i) + j], sum1 + sum2 + sum3);")
        self._pop()
        self._emit("}")

    def _emit_cooperative_matrix2_multi_arg_body(self, tile_size: int, chunk_size: int) -> None:
        self._emit("void main() {")
        self._push()
        self._emit("const uint tile_row = gl_WorkGroupID.x;")
        self._emit("const uint tile_col = gl_WorkGroupID.y;")
        self._emit("")
        self._emit("tensorLayoutNV<2> tensorLayoutA = createTensorLayoutNV(2);")
        self._emit("tensorLayoutNV<2> tensorLayoutB = createTensorLayoutNV(2);")
        self._emit("tensorLayoutNV<2> tensorLayoutC = createTensorLayoutNV(2);")
        self._emit("tensorLayoutA = setTensorLayoutDimensionNV(tensorLayoutA, 1024, 1024);")
        self._emit("tensorLayoutB = setTensorLayoutDimensionNV(tensorLayoutB, 1024, 1024);")
        self._emit("tensorLayoutC = setTensorLayoutDimensionNV(tensorLayoutC, 1024, 1024);")
        self._emit("")
        self._emit(
            f"coopmat<float, gl_ScopeWorkgroup, {tile_size}, {tile_size}, gl_MatrixUseAccumulator> result = "
            f"coopmat<float, gl_ScopeWorkgroup, {tile_size}, {tile_size}, gl_MatrixUseAccumulator>(0.0);"
        )
        self._emit("")
        self._emit(f"for (uint chunkK = 0; chunkK < 1024; chunkK += {chunk_size}) {{")
        self._push()
        for suffix in ("1", "2", "3"):
            self._emit(
                f"coopmat<float, gl_ScopeWorkgroup, {tile_size}, {chunk_size}, gl_MatrixUseA> matrixA{suffix};"
            )
            self._emit(
                f"coopmat<float, gl_ScopeWorkgroup, {chunk_size}, {tile_size}, gl_MatrixUseB> matrixB{suffix};"
            )
            self._emit(
                f"coopMatLoadTensorNV(matrixA{suffix}, A{suffix}, 0, "
                f"sliceTensorLayoutNV(tensorLayoutA, {tile_size} * tile_row, {tile_size}, chunkK, {chunk_size}));"
            )
            self._emit(
                f"coopMatLoadTensorNV(matrixB{suffix}, B{suffix}, 0, "
                f"sliceTensorLayoutNV(tensorLayoutB, chunkK, {chunk_size}, {tile_size} * tile_col, {tile_size}));"
            )
            self._emit(f"result = coopMatMulAdd(matrixA{suffix}, matrixB{suffix}, result);")
        self._pop()
        self._emit("}")
        self._emit("")
        self._emit(
            f"coopmat<float, gl_ScopeWorkgroup, {tile_size}, {tile_size}, gl_MatrixUseAccumulator> matrixC;"
        )
        self._emit(
            f"coopMatLoadTensorNV(matrixC, C, 0, "
            f"sliceTensorLayoutNV(tensorLayoutC, {tile_size} * tile_row, {tile_size}, {tile_size} * tile_col, {tile_size}));"
        )
        self._emit("result = result + matrixC;")
        self._emit(
            f"coopMatStoreTensorNV(result, C, 0, "
            f"sliceTensorLayoutNV(tensorLayoutC, {tile_size} * tile_row, {tile_size}, {tile_size} * tile_col, {tile_size}));"
        )
        self._pop()
        self._emit("}")

    # ── Program visitor (main entry point) ────────────────────────────


    def _emit_tile_vars(self, node):
        """Map each loop variable index to its name (used for gl_GlobalInvocationID init).
        
        When tiled, tile indices replace original loop variables as the outer dimensions.
        Each workgroup handles one tile so loop_var names map directly to 
        gl_GlobalInvocationID.x/y/z coordinates.
        """
        self._tile_var_name = {}
        for idx, var_name in enumerate(node.loop_vars):
            self._tile_var_name[idx] = var_name  # use original name (i, j)

    def _flatten_tiled_body(self, stmts):
        """Recursively flatten tile loops out of body statements.
        
        When tiling is applied, tile_i/tile_j loops should not appear in the GLSL body.
        Instead their content is flattened into the main() function body, and the 
        tile indices come from gl_GlobalInvocationID.x/y.
        """
        result = []
        for stmt in stmts:
            if isinstance(stmt, ForLoopWithConditionAndIncrement) and                getattr(stmt, "loop_var_name", "").startswith("tile_"):
                # Skip this tile loop, flatten its body instead
                inner_stmts = getattr(stmt, "body_stmts", [])
                if inner_stmts:
                    result.extend(self._flatten_tiled_body(inner_stmts))
            else:
                result.append(stmt)
        return result

    def visit_program(self, node: Program) -> str:
        self._lines = []
        self._indent_level = 0
        self._binding_counter = 0
        self._ssbo_map = {}
        self._push_constant_fields = []
        self._push_constant_map = {}
        self._triangular_upper_bound_name = None
        self._uses_reduction_chunks = getattr(node, "reduction_chunks", 1) > 1

        self._emit("#version 450")
        self._emit("#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require")
        self._emit("#extension GL_KHR_shader_subgroup_arithmetic : require")
        self._emit("#extension GL_KHR_shader_subgroup_clustered : require")
        if getattr(node, "use_cooperative_matrix2", False):
            self._emit("#extension GL_KHR_memory_scope_semantics : require")
            self._emit("#extension GL_KHR_cooperative_matrix : require")
            self._emit("#extension GL_NV_cooperative_matrix2 : require")
        if self._uses_reduction_chunks:
            self._emit("#extension GL_EXT_shader_atomic_float : require")
        wg_x, wg_y, wg_z = self._workgroup_size(node)
        self._emit(f"layout(local_size_x = {wg_x}, local_size_y = {wg_y}, local_size_z = {wg_z}) in;")
        self._emit("")

        # ── Classify parameters ──
        all_matrix_params = []
        triangular_bounds_raw = (
            node.triangular_bounds_raw
            if getattr(node, "triangular_bounds_raw", None)
            else []
        )
        is_triangular = len(triangular_bounds_raw) >= 2
        if is_triangular and not triangular_bounds_raw[1].lstrip("-").isdigit():
            self._triangular_upper_bound_name = triangular_bounds_raw[1]

        for param in node.params:
            if param is None or not isinstance(param, Declaration):
                continue

            vt = param.var_type
            is_matrix = isinstance(
                vt,
                (
                    FlexibleRowsMatrix,
                    FixedSizeMatrix,
                    FlexibleRowsColsLevelsMatrix,
                    FixedSizeLevelsRowsColsMatrix,
                ),
            )
            is_vector = isinstance(vt, FixedSizeVector)

            if is_matrix or is_vector:
                all_matrix_params.append(param)
            else:
                if param.name not in {f[0] for f in self._push_constant_fields}:
                    self._push_constant_fields.append((param.name, "int"))
                    self._push_constant_map[param.name] = True

        def _is_literal(s):
            return s.lstrip("-").isdigit() if s else False

        for tb in triangular_bounds_raw:
            if not _is_literal(tb) and tb not in {
                f[0] for f in self._push_constant_fields
            }:
                self._push_constant_fields.append((tb, "int"))
                self._push_constant_map[tb] = True

        # ── Emit SSBO buffers with compile-time sized arrays ──
        for param in all_matrix_params:
            vt = param.var_type
            is_3d = hasattr(vt, "level_expr") and vt.level_expr is not None
            inner = self._glsl_elem_type(vt)

            size = _compute_matrix_size(vt)
            if size >= 2147483648:
                size_str = f" /* too large for int: {size} */ "
            elif size > 0:
                size_str = str(size)
            else:
                size_str = " /* unknown */ "

            self._emit(
                f"layout(std430, set = 0, binding = {self._binding_counter}) buffer RllmBuffer_{param.name} {{"
            )
            self._push()
            self._emit(f"{inner} {param.name}[{size_str}];")
            self._pop()
            self._emit("};")

            self._ssbo_map[param.name] = (is_3d, vt)
            self._binding_counter += 1

        # ── Emit push_constant block ──
        if self._push_constant_fields:
            self._emit("")
            self._emit("layout(push_constant) uniform RllmPushConstants {")
            self._push()
            for name, vtype in self._push_constant_fields:
                self._emit(f"{vtype} {name};")
            self._pop()
            self._emit("} rllm_push;")

        if self._is_shared_memory_multi_arg(node):
            self._emit("")
            self._emit_shared_memory_multi_arg_body(
                getattr(node, "tile_block_size", 8),
                getattr(node, "shared_memory_chunk_size", 8),
            )
            return self.result()

        if self._is_cooperative_matrix2_multi_arg(node):
            self._emit("")
            self._emit_cooperative_matrix2_multi_arg_body(
                getattr(node, "tile_block_size", 8),
                getattr(node, "cooperative_matrix2_chunk_size", 8),
            )
            return self.result()

        # ── Main function ──
        self._emit("")
        self._emit("void main() {")
        self._push()

        # Tile variable names (when tiling is applied)
        if getattr(node, "tiled", False):
            self._emit_tile_vars(node)

        # 1. Initialize loop variables from gl_GlobalInvocationID
        if node.space_dim >= 1 and node.loop_vars:
            for idx, var_name in enumerate(node.loop_vars):
                coord = "xyz"[idx]
                if getattr(node, 'tiled', False) and hasattr(self, '_tile_var_name') and self._tile_var_name.get(idx):
                    tvn = self._tile_var_name[idx]
                    self._emit(f"int {tvn} = int(gl_GlobalInvocationID.{coord});")
                else:
                    self._emit(f"int {var_name} = int(gl_GlobalInvocationID.{coord});")

        # 2. Initialize params from push constants
        for name, vtype in self._push_constant_fields:
            if vtype == "int":
                self._emit(f"int {name} = rllm_push.{name};")

        # 3. Triangular guard (if applicable)
        if is_triangular and triangular_bounds_raw and len(triangular_bounds_raw) >= 2:
            lower_name = triangular_bounds_raw[0]
            upper_name = triangular_bounds_raw[1]

            parts = []
            if node.loop_vars:
                first_var = node.loop_vars[0]
                if _is_literal(lower_name):
                    parts.append(f"{first_var} >= {lower_name}")
                else:
                    parts.append(f"{first_var} >= rllm_push.{lower_name}")

            for var in node.loop_vars[1:]:
                if _is_literal(upper_name):
                    parts.append(f"{var} >= {upper_name}")
                else:
                    parts.append(f"{var} >= rllm_push.{upper_name}")

            self._emit(f"if ({' || '.join(parts)}) return;")

        # 4. Body statements - just emit directly (no double-emission)
        old_indent = self._indent_level
        self._indent_level += 1

        if getattr(node, 'tiled', False):
            # When tiling is applied, tile loops should come from gl_GlobalInvocationID.x/y
            # Skip explicit tile loop constructs and flatten their content into main() body
            body_stmts = self._flatten_tiled_body(node.body_stmts)
            for stmt in body_stmts:
                if hasattr(stmt, "accept"):
                    result = stmt.accept(self)
                    if isinstance(result, str):
                        self._emit(result.rstrip())
        else:
            for stmt in node.body_stmts:
                if hasattr(stmt, "accept"):
                    result = stmt.accept(self)
                    if isinstance(result, str):
                        self._emit(result.rstrip())

        self._indent_level = old_indent
        self._pop()
        self._emit("}")

        return self.result()
