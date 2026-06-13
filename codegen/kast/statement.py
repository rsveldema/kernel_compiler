"""Statement AST nodes for code generation."""

from codegen.kast.ast_node import AstNode
from codegen.kast.expression import Expression
from codegen.kast.type import Type


class Condition(AstNode):
    def __init__(self, lhs: Expression, op: str, rhs: Expression):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    def accept(self, visitor):
        return visitor.visit_condition(self)


class Statement(AstNode):
    """Base class for statement AST nodes."""

    def accept(self, visitor):
        return visitor.visit_statement(self)


class ForStatement(Statement):
    """Base class for for-loop AST nodes."""

    pass


class ForLoopRange(ForStatement):
    """Range-style loop: `for (const int i: expr)` — iterates from 0..expr-1."""

    def __init__(
        self,
        loop_var_type: Type | None = None,
        loop_var_name: str = "",
        init_expr: Expression | None = None,
        body_stmts: list[Statement] | None = None,
    ):
        self.loop_var_type = loop_var_type
        self.loop_var_name = loop_var_name
        self.init_expr = init_expr
        self.body_stmts = body_stmts or []

    def accept(self, visitor):
        return visitor.visit_for_loop_range(self)


class ForLoopWithConditionAndIncrement(ForStatement):
    """C-style for loop: `for (init; condition; increment)`."""

    def __init__(
        self,
        loop_var_type: Type | None = None,
        loop_var_name: str = "",
        condition: Expression | None = None,
        increment_var: str = "",
        increment_op: str = "",
        body_stmts: list[Statement] | None = None,
        init_expr: Expression | None = None,
    ):
        self.loop_var_type = loop_var_type
        self.loop_var_name = loop_var_name
        self.condition = condition
        self.increment_var = increment_var
        self.increment_op = increment_op
        self.body_stmts = body_stmts or []
        self.init_expr = init_expr

    def accept(self, visitor):
        return visitor.visit_for_loop_with_condition_and_increment(self)


# Backwards compatibility alias — prefer explicit subclass usage
For = ForLoopWithConditionAndIncrement


class If(Statement):
    def __init__(self, condition: Expression, body_stmts: list[Statement]):
        self.condition = condition
        self.body_stmts = body_stmts or []

    def accept(self, visitor):
        return visitor.visit_if(self)


class Declaration(Statement):
    def __init__(
        self,
        is_const: bool,
        var_type: Type,
        name: str,
        init_expr: Expression | None = None,
    ):
        self.is_const = is_const
        self.var_type = var_type
        self.name = name
        self.init_expr = init_expr

    def accept(self, visitor):
        return visitor.visit_declaration(self)


class Assignment(Statement):
    def __init__(self, lvalue: Expression, assign_op: str, rvalue: Expression = None):
        self.lvalue = lvalue
        self.assign_op = assign_op
        self.rvalue = rvalue

    def accept(self, visitor):
        return visitor.visit_assignment(self)


class OverflowCheck(Statement):
    """OVERFLOW_CHECK_ADD(lvalue, operand) statement."""

    def __init__(self, lvalue: Expression, operand: str):
        self.lvalue = lvalue
        self.operand = operand

    def accept(self, visitor):
        return visitor.visit_overflow_check(self)


class SharedDecl(Statement):
    """shared declaration in workgroup context (from 'shared' alternative)."""

    def __init__(
        self,
        is_const: bool,
        var_type: Type,
        name: str,
        init_expr: Expression | None = None,
    ):
        self.is_const = is_const
        self.var_type = var_type
        self.name = name
        self.init_expr = init_expr

    def accept(self, visitor):
        return visitor.visit_shared_decl(self)


__all__ = [
    "Condition",
    "Statement",
    "ForStatement",
    "ForLoopRange",
    "ForLoopWithConditionAndIncrement",
    "For",
    "If",
    "Declaration",
    "Assignment",
    "OverflowCheck",
    "SharedDecl",
]
