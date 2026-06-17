
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
    RawStatement,
)
from codegen.kast.expression import (
    Identifier,
    Number,
    BinaryExpr,
    LimitExpr,
)
from codegen.kast.type import Int
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
            pattern = Pattern(str(pattern_path))
            self.patterns[pattern_path.stem] = pattern


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

        # Apply rewrite patterns only when corresponding tkernel files are present.
        if "step2_guard" in self.pattern_store.patterns:
            body_stmts = self._apply_step2_to_statements(body_stmts)
        if "step3_chunked_loop" in self.pattern_store.patterns:
            body_stmts = self._apply_step3_to_statements(body_stmts, target_loop_bounds, workgroups)

        node.body_stmts = body_stmts
        return node

    def _is_for_statement(self, stmt):
        return isinstance(stmt, (ForLoopWithConditionAndIncrement, ForLoopRange))

    def _apply_step2_to_statements(self, statements):
        rewritten = []
        for stmt in statements or []:
            if self._is_for_statement(stmt):
                cloned = deepcopy(stmt)
                cloned.body_stmts = self._apply_step2_to_statements(getattr(cloned, "body_stmts", []) or [])
                rewritten.append(cloned)
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
                if concrete_bound is not None and concrete_bound in target_loop_bounds:
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
    