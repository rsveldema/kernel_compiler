"""Pretty-printing visitor for AST nodes."""

from codegen.ast.ast_node import AstNode

from .. import ast
from .visitor import Visitor


class PrettyPrinter(Visitor):
    """Pretty-prints the AST to a string using indentation."""

    def __init__(self, indent: int = 0):
        self.indent = indent

    def _child_indent(self):
        return PrettyPrinter(self.indent + 2)

    def _indent_str(self):
        return " " * self.indent

    # ── Type visitors ────────────────────────────────────────────────

    def visit_type(self, node: ast.Type):
        return node.accept(self._child_indent())

    def visit_int(self, node: ast.Int):
        return f"{self._indent_str()}int"

    def visit_float(self, node: ast.Float):
        return f"{self._indent_str()}float"

    def visit_fixed_size_vector(self, node: ast.FixedSizeVector):
        indent = self._indent_str()
        inner_lines = []
        if node.elem_type is not None:
            inner_lines.append(f"{indent}  elem_type: {node.elem_type.accept(self)}")
        if node.size_expr is not None:
            inner_lines.append(f"{indent}  size_expr: {node.size_expr.accept(self)}")
        inner = "\n".join(inner_lines)
        return f"{indent}fixed_size_vector<\n{inner}{indent}>"

    def visit_flexible_rows_matrix(self, node: ast.FlexibleRowsMatrix):
        indent = self._indent_str()
        inner_lines = []
        if node.elem_type is not None:
            inner_lines.append(f"{indent}  elem_type: {node.elem_type.accept(self)}")
        if node.row_size_expr is not None:
            inner_lines.append(
                f"{indent}  row_size_expr: {node.row_size_expr.accept(self)}"
            )
        if node.col_size_expr is not None:
            inner_lines.append(
                f"{indent}  col_size_expr: {node.col_size_expr.accept(self)}"
            )
        inner = "\n".join(inner_lines)
        return f"{indent}flexible_rows_matrix<\n{inner}{indent}>"

    def visit_fixed_size_matrix(self, node: ast.FixedSizeMatrix):
        indent = self._indent_str()
        inner_lines = []
        if node.elem_type is not None:
            inner_lines.append(f"{indent}  elem_type: {node.elem_type.accept(self)}")
        if node.row_size_expr is not None:
            inner_lines.append(
                f"{indent}  row_size_expr: {node.row_size_expr.accept(self)}"
            )
        if node.col_size_expr is not None:
            inner_lines.append(
                f"{indent}  col_size_expr: {node.col_size_expr.accept(self)}"
            )
        inner = "\n".join(inner_lines)
        return f"{indent}fixed_size_matrix<\n{inner}{indent}>"

    def visit_fixed_size_levels_rows_cols_matrix(
        self, node: ast.FixedSizeLevelsRowsColsMatrix
    ):
        indent = self._indent_str()
        inner_lines = []
        if node.elem_type is not None:
            inner_lines.append(f"{indent}  elem_type: {node.elem_type.accept(self)}")
        if node.level_expr is not None:
            inner_lines.append(f"{indent}  level_expr: {node.level_expr.accept(self)}")
        if node.row_size_expr is not None:
            inner_lines.append(
                f"{indent}  row_size_expr: {node.row_size_expr.accept(self)}"
            )
        if node.col_size_expr is not None:
            inner_lines.append(
                f"{indent}  col_size_expr: {node.col_size_expr.accept(self)}"
            )
        inner = "\n".join(inner_lines)
        return f"{indent}fixed_size_levels_rows_cols_matrix<\n{inner}{indent}>"

    def visit_flexible_rows_cols_levels_matrix(
        self, node: ast.FlexibleRowsColsLevelsMatrix
    ):
        indent = self._indent_str()
        inner_lines = []
        if node.elem_type is not None:
            inner_lines.append(f"{indent}  elem_type: {node.elem_type.accept(self)}")
        if node.level_expr is not None:
            inner_lines.append(f"{indent}  level_expr: {node.level_expr.accept(self)}")
        if node.row_size_expr is not None:
            inner_lines.append(
                f"{indent}  row_size_expr: {node.row_size_expr.accept(self)}"
            )
        if node.col_size_expr is not None:
            inner_lines.append(
                f"{indent}  col_size_expr: {node.col_size_expr.accept(self)}"
            )
        inner = "\n".join(inner_lines)
        return f"{indent}flexible_rows_cols_levels_matrix<\n{inner}{indent}>"

    # ── Expression visitors ─────────────────────────────────────────

    def visit_expression(self, node: ast.Expression):
        return node.accept(self._child_indent())

    def visit_number(self, node: ast.Number):
        return f"{self._indent_str()}{node.value}"

    def visit_identifier(self, node: ast.Identifier):
        return f"{self._indent_str()}{node.name}"

    def visit_array_access(self, node: ast.ArrayAccess):
        indent = self._indent_str()
        # Build the base string (could be Identifier or FieldAccess)
        if isinstance(node.base, ast.Identifier):
            base_str = node.base.name or "?"
        else:
            base_str = node.base.accept(self)
        # Join all indices with commas in a single bracket
        idx_parts = [self._to_string_node(idx) for idx in node.indices]
        return f"{indent}{base_str}[{', '.join(idx_parts)}]"

    def _to_string_node(self, node):
        """Convert an AST node to a compact string representation."""
        if isinstance(node, ast.Identifier):
            return node.name or "?"
        if isinstance(node, ast.Number):
            return str(node.value)
        if hasattr(node, "accept"):
            old_indent = self.indent
            self.indent = 0
            result = str(node.accept(self)).strip()
            self.indent = old_indent
            return result
        return str(node)

    def visit_field_access(self, node: ast.FieldAccess):
        indent = self._indent_str()
        if isinstance(node.base, ast.Identifier):
            base_str = node.base.name or "?"
        else:
            base_str = node.base.accept(self)
        base_str += f".{node.field}"
        return f"{indent}{base_str}"

    def _to_string_node(self, node):
        """Convert an AST node to a string without indentation."""
        if isinstance(node, ast.Identifier):
            return node.name or "?"
        if isinstance(node, ast.Number):
            return str(node.value)
        if hasattr(node, "accept"):
            # Use visitor but collect the result (which includes its own indent)
            return str(node.accept(self)).strip()
        return str(node)

    def visit_limit_expr(self, node: ast.LimitExpr):
        indent = self._indent_str()
        max_part = node.max_val.accept(self) if node.max_val else "None"
        body_part = node.body.accept(self) if node.body else "None"
        inner_indent = " " * (self.indent + 2)
        return (
            f"{indent}limit<\n"
            f"{inner_indent}{max_part}\n"
            f"{indent}>\n{inner_indent}(\n"
            f"{inner_indent}  {body_part}\n"
            f"{indent})"
        )

    def visit_binary_expr(self, node: ast.BinaryExpr):
        indent = self._indent_str()
        return (
            f"{indent}binary<\n"
            f"{indent}  op: {node.op}\n"
            f"{indent}  lhs: {node.left.accept(self)}\n"
            f"{indent}  rhs: {node.right.accept(self)}\n"
            f"{indent}>"
        )

    def visit_cast_expr(self, node: ast.CastExpr):
        return (
            f"cast<{node.cast_type.accept(self._child_indent())}>("
            f"{node.operand.accept(self)})"
        )

    def visit_negation_expr(self, node: ast.NegationExpr):
        return f"{self._indent_str()}!{node.operand.accept(self)}"

    # ── Condition visitors ───────────────────────────────────────────

    def visit_condition(self, node: ast.Condition):
        return f"{node.lhs.accept(self)} {node.op} {node.rhs.accept(self)}"

    # ── Statement visitors ───────────────────────────────────────────

    def visit_statement(self, node: ast.Statement):
        return node.accept(self._child_indent())

    def visit_for_loop_range(self, node: ast.ForLoopRange):
        indent = self._indent_str()
        type_str = node.loop_var_type.accept(self) if node.loop_var_type else "u32"
        init_str = f": {node.init_expr.accept(self)}" if node.init_expr else ""
        body_lines = "\n".join(s.accept(self._child_indent()) for s in node.body_stmts)
        return (
            f"{indent}for ({type_str} {node.loop_var_name}{init_str}) {{\n"
            f"{body_lines}\n"
            f"{indent}}}"
        )

    def visit_for_loop_with_condition_and_increment(
        self, node: ast.ForLoopWithConditionAndIncrement
    ):
        indent = self._indent_str()
        cond_str = node.condition.accept(self) if node.condition else "None"
        body_lines = "\n".join(s.accept(self._child_indent()) for s in node.body_stmts)
        return (
            f"{indent}for ({node.loop_var_type.accept(self) if node.loop_var_type else 'u32'} {node.loop_var_name}; "
            f"{cond_str}; {node.increment_var}{node.increment_op}) {{\n"
            f"{body_lines}\n"
            f"{indent}}}"
        )

    def visit_if(self, node: ast.If):
        indent = self._indent_str()
        cond_str = node.condition.accept(self) if node.condition else "None"
        body_lines = "\n".join(s.accept(self._child_indent()) for s in node.body_stmts)
        return f"{indent}if ({cond_str}) {{\n{body_lines}\n{indent}}}"

    def visit_declaration(self, node: ast.Declaration):
        indent = self._indent_str()
        prefix = "const " if node.is_const else ""
        init_str = f" = {node.init_expr.accept(self)}" if node.init_expr else ""
        return f"{indent}{prefix}{node.var_type.accept(self)} {node.name}{init_str};"

    def visit_assignment(self, node: ast.Assignment):
        indent = self._indent_str()
        return f"{indent}{node.lvalue.accept(self)} {node.assign_op} {node.rvalue.accept(self)};"

    def visit_overflow_check(self, node: ast.OverflowCheck):
        indent = self._indent_str()
        return (
            f"{indent}OVERFLOW_CHECK_ADD("
            f"{node.lvalue.accept(self)}, {node.operand.accept(self)})"
        )

    def visit_shared_decl(self, node: ast.SharedDecl):
        indent = self._indent_str()
        prefix = "const " if node.is_const else ""
        init_str = f" = {node.init_expr.accept(self)}" if node.init_expr else ""
        return f"{indent}shared {prefix}{node.var_type.accept(self)} {node.name}{init_str};"

    def visit_workgroup_properties(self, node: ast.WorkgroupProperties):
        indent = self._indent_str()
        parts = []
        if node.x_expr is not None:
            parts.append(f"x: {node.x_expr.accept(self)}")
        if node.y_expr is not None:
            parts.append(f"y: {node.y_expr.accept(self)}")
        if node.z_expr is not None:
            parts.append(f"z: {node.z_expr.accept(self)}")
        return f"{indent}workgroup {{ {', '.join(parts)} }}"

    def visit_program(self, node: ast.Program):
        indent = self._indent_str()
        lines = [f"{indent}PROGRAM({node.header})"]
        if node.loop_vars:
            lines.append(f"{indent}  space: {node.loop_vars}")
        if node.limit_expr is not None:
            lines.append(f"{indent}  limit: {node.limit_expr.accept(self)}")
        param_lines = "\n".join(p.accept(self._child_indent()) for p in node.params)
        body_lines = "\n".join(s.accept(self._child_indent()) for s in node.body_stmts)
        wg_lines = "\n".join(w.accept(self._child_indent()) for w in node.workgroups)

        return (
            f"{indent}Program {{\n"
            f"  header: {node.header}\n"
            f"  space: {node.loop_vars}\n"
            f"  params:\n{param_lines}\n"
            f"  body:\n{body_lines}\n"
            f"  workgroups:\n{wg_lines}\n"
            f"{indent}}}"
        )



def prettyprint(x: AstNode) -> str:
    printer = PrettyPrinter()
    s = x.accept(printer)
    return s
