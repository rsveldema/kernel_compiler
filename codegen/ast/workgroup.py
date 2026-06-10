"""Workgroup properties AST node for code generation."""


class WorkgroupProperties:
    """workgroup { x: expr, y: expr, z: expr } properties."""
    def __init__(self, x_expr=None, y_expr=None, z_expr=None):
        self.x_expr = x_expr
        self.y_expr = y_expr
        self.z_expr = z_expr

    def accept(self, visitor):
        return visitor.visit_workgroup_properties(self)


__all__ = ['WorkgroupProperties']
