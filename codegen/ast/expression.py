"""Expression AST nodes for code generation."""

from .ast_node import AstNode


class Expression(AstNode):
    """Base class for expression AST nodes."""

    def accept(self, visitor):
        return visitor.visit_expression(self)


class Number(Expression):
    def __init__(self, value: str):
        self.value = value

    def accept(self, visitor):
        return visitor.visit_number(self)


class Identifier(Expression):
    def __init__(self, name: str):
        self.name = name

    def accept(self, visitor):
        return visitor.visit_identifier(self)


class ArrayAccess(Expression):
    """Represents a[expr1, expr2, ...] array indexing."""

    def __init__(self, base: Expression, indices: list[Expression]):
        self.base = base
        self.indices = indices or []

    def accept(self, visitor):
        return visitor.visit_array_access(self)


class FieldAccess(Expression):
    """Represents a.b chain of member accesses.

    The ``base`` is the leftmost expression (an Identifier by convention),
    and ``fields`` holds every field name in order after it.  For example
    ``obj.x.y`` becomes ``FieldAccess(Identifier("obj"), ["x", "y"])``.

    When there are no fields (length == 0) the node is effectively an alias
    for the bare base Identifier and pretty-printing will emit just ``base``.
    """

    def __init__(self, base: Expression, field: str):
        assert base is not None
        assert field is not None
        self.base = base
        self.field = field

    def accept(self, visitor):
        return visitor.visit_field_access(self)


class LimitExpr(Expression):
    def __init__(self, max_val: Expression, body: Expression):
        self.max_val = max_val
        self.body = body

    def accept(self, visitor):
        return visitor.visit_limit_expr(self)


class BinaryExpr(Expression):
    """Represents a binary operation (e.g., a + b)."""

    def __init__(self, left: Expression, op: str, right: Expression):
        self.left = left
        self.op = op
        self.right = right

    def accept(self, visitor):
        return visitor.visit_binary_expr(self)


class CastExpr(Expression):
    """Represents a cast expression (e.g., int(x))."""

    def __init__(self, cast_type: Type, operand: Expression):
        self.cast_type = cast_type
        self.operand = operand

    def accept(self, visitor):
        return visitor.visit_cast_expr(self)


class NegationExpr(Expression):
    """Represents a negation expression (e.g., !x)."""

    def __init__(self, operand: Expression):
        self.operand = operand

    def accept(self, visitor):
        return visitor.visit_negation_expr(self)


__all__ = [
    "Expression",
    "Number",
    "Identifier",
    "ArrayAccess",
    "FieldAccess",
    "LimitExpr",
    "BinaryExpr",
    "CastExpr",
    "NegationExpr",
]
