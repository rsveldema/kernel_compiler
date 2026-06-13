"""Visitor pattern infrastructure: base Visitor class and AST type references."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import kast as _ast


class Visitor:
    """Base visitor class. Subclasses override visit_* methods as needed."""

    def visit_type(self, node: _ast.Type):
        raise NotImplementedError

    def visit_int(self, node: _ast.Int):
        raise NotImplementedError

    def visit_float(self, node: _ast.Float):
        raise NotImplementedError

    def visit_fixed_size_vector(self, node: _ast.FixedSizeVector):
        raise NotImplementedError

    def visit_flexible_rows_matrix(self, node: _ast.FlexibleRowsMatrix):
        raise NotImplementedError

    def visit_fixed_size_matrix(self, node: _ast.FixedSizeMatrix):
        raise NotImplementedError

    def visit_fixed_size_levels_rows_cols_matrix(
        self, node: _ast.FixedSizeLevelsRowsColsMatrix
    ):
        raise NotImplementedError

    def visit_flexible_rows_cols_levels_matrix(
        self, node: _ast.FlexibleRowsColsLevelsMatrix
    ):
        raise NotImplementedError

    def visit_expression(self, node: _ast.Expression):
        raise NotImplementedError

    def visit_number(self, node: _ast.Number):
        raise NotImplementedError

    def visit_identifier(self, node: _ast.Identifier):
        raise NotImplementedError

    def visit_array_access(self, node: _ast.ArrayAccess):
        raise NotImplementedError

    def visit_field_access(self, node: _ast.FieldAccess):
        raise NotImplementedError

    def visit_limit_expr(self, node: _ast.LimitExpr):
        raise NotImplementedError

    def visit_binary_expr(self, node: _ast.BinaryExpr):
        raise NotImplementedError

    def visit_cast_expr(self, node: _ast.CastExpr):
        raise NotImplementedError

    def visit_negation_expr(self, node: _ast.NegationExpr):
        raise NotImplementedError

    def visit_condition(self, node: _ast.Condition):
        raise NotImplementedError

    def visit_statement(self, node: _ast.Statement):
        raise NotImplementedError

    def visit_for_loop_range(self, node: _ast.ForLoopRange):
        raise NotImplementedError

    def visit_for_loop_with_condition_and_increment(
        self, node: _ast.ForLoopWithConditionAndIncrement
    ):
        raise NotImplementedError

    def visit_if(self, node: _ast.If):
        raise NotImplementedError

    def visit_declaration(self, node: _ast.Declaration):
        raise NotImplementedError

    def visit_assignment(self, node: _ast.Assignment):
        raise NotImplementedError

    def visit_overflow_check(self, node: _ast.OverflowCheck):
        raise NotImplementedError

    def visit_shared_decl(self, node: _ast.SharedDecl):
        raise NotImplementedError

    def visit_workgroup_properties(self, node: _ast.WorkgroupProperties):
        raise NotImplementedError

    def visit_program(self, node: _ast.Program):
        raise NotImplementedError
