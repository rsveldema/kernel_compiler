"""Visitor pattern infrastructure: base Visitor class and AST type references."""

from .. import ast


class Visitor:
    """Base visitor class. Subclasses override visit_* methods as needed."""

    def visit_type(self, node: ast.Type):
        raise NotImplementedError

    def visit_int(self, node: ast.Int):
        raise NotImplementedError

    def visit_float(self, node: ast.Float):
        raise NotImplementedError

    def visit_fixed_size_vector(self, node: ast.FixedSizeVector):
        raise NotImplementedError

    def visit_flexible_rows_matrix(self, node: ast.FlexibleRowsMatrix):
        raise NotImplementedError

    def visit_fixed_size_matrix(self, node: ast.FixedSizeMatrix):
        raise NotImplementedError

    def visit_fixed_size_levels_rows_cols_matrix(self, node: ast.FixedSizeLevelsRowsColsMatrix):
        raise NotImplementedError

    def visit_flexible_rows_cols_levels_matrix(self, node: ast.FlexibleRowsColsLevelsMatrix):
        raise NotImplementedError

    def visit_expression(self, node: ast.Expression):
        raise NotImplementedError

    def visit_number(self, node: ast.Number):
        raise NotImplementedError

    def visit_identifier(self, node: ast.Identifier):
        raise NotImplementedError

    def visit_array_access(self, node: ast.ArrayAccess):
        raise NotImplementedError

    def visit_field_access(self, node: ast.FieldAccess):
        raise NotImplementedError

    def visit_limit_expr(self, node: ast.LimitExpr):
        raise NotImplementedError

    def visit_binary_expr(self, node: ast.BinaryExpr):
        raise NotImplementedError

    def visit_cast_expr(self, node: ast.CastExpr):
        raise NotImplementedError

    def visit_negation_expr(self, node: ast.NegationExpr):
        raise NotImplementedError

    def visit_condition(self, node: ast.Condition):
        raise NotImplementedError

    def visit_statement(self, node: ast.Statement):
        raise NotImplementedError

    def visit_for(self, node: ast.For):
        raise NotImplementedError

    def visit_if(self, node: ast.If):
        raise NotImplementedError

    def visit_declaration(self, node: ast.Declaration):
        raise NotImplementedError

    def visit_assignment(self, node: ast.Assignment):
        raise NotImplementedError

    def visit_overflow_check(self, node: ast.OverflowCheck):
        raise NotImplementedError

    def visit_shared_decl(self, node: ast.SharedDecl):
        raise NotImplementedError

    def visit_workgroup_properties(self, node: ast.WorkgroupProperties):
        raise NotImplementedError

    def visit_program(self, node: ast.Program):
        raise NotImplementedError
