"""Statement AST nodes for code generation."""


class Condition:
    def __init__(self, lhs=None, op=None, rhs=None):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    def accept(self, visitor):
        return visitor.visit_condition(self)


class Statement:
    """Base class for statement AST nodes."""
    def accept(self, visitor):
        return visitor.visit_statement(self)


class For(Statement):
    def __init__(self, loop_var_type=None, loop_var_name="",
                 condition=None, increment_var="",
                 increment_op="", body_stmts=None, init_expr=None):
        self.loop_var_type = loop_var_type
        self.loop_var_name = loop_var_name
        self.condition = condition
        self.increment_var = increment_var
        self.increment_op = increment_op
        self.body_stmts = body_stmts or []
        self.init_expr = init_expr

    def accept(self, visitor):
        return visitor.visit_for(self)


class If(Statement):
    def __init__(self, condition=None, body_stmts=None):
        self.condition = condition
        self.body_stmts = body_stmts or []

    def accept(self, visitor):
        return visitor.visit_if(self)


class Declaration(Statement):
    def __init__(self, is_const=None, var_type=None, name=None, init_expr=None):
        self.is_const = is_const
        self.var_type = var_type
        self.name = name
        self.init_expr = init_expr

    def accept(self, visitor):
        return visitor.visit_declaration(self)


class Assignment(Statement):
    def __init__(self, lvalue=None, assign_op=None, rvalue=None):
        self.lvalue = lvalue
        self.assign_op = assign_op
        self.rvalue = rvalue

    def accept(self, visitor):
        return visitor.visit_assignment(self)


class OverflowCheck(Statement):
    """OVERFLOW_CHECK_ADD(lvalue, operand) statement."""
    def __init__(self, lvalue=None, operand=None):
        self.lvalue = lvalue
        self.operand = operand

    def accept(self, visitor):
        return visitor.visit_overflow_check(self)


class SharedDecl(Statement):
    """shared declaration in workgroup context (from 'shared' alternative)."""
    def __init__(self, is_const=None, var_type=None, name=None, init_expr=None):
        self.is_const = is_const
        self.var_type = var_type
        self.name = name
        self.init_expr = init_expr

    def accept(self, visitor):
        return visitor.visit_shared_decl(self)


__all__ = [
    'Condition', 'Statement', 'For', 'If',
    'Declaration', 'Assignment', 'OverflowCheck', 'SharedDecl',
]
