"""Program AST node for code generation."""


class Program:
    def __init__(self, header="", loop_vars=None, space_dim=0, grid_name="",
                 limit_expr=None, dispatch_size_expr=None,
                 lower_bound_expr=None, upper_bound_expr=None,
                 triangular_bounds_raw=None,  # [raw_lower_str, raw_upper_str] for triangular
                 params=None, body_stmts=None, workgroups=None):
        self.header = header
        # Loop variables from OFFLOAD_PARFOR_x_PARAM (e.g. ['i'] or ['i', 'j'])
        self.loop_vars = loop_vars or []
        # Dimensionality: 1, 2, or 3
        self.space_dim = space_dim
        # Grid name for multi-dim parfor (e.g. "grid" from OFFLOAD_PARFOR_2D_PARAM)
        self.grid_name = grid_name
        self.limit_expr = limit_expr
        self.dispatch_size_expr = dispatch_size_expr
        # Triangular parfor bounds (lower and upper)
        self.lower_bound_expr = lower_bound_expr
        self.upper_bound_expr = upper_bound_expr
        self.triangular_bounds_raw = triangular_bounds_raw or []
        self.params = params or []
        self.body_stmts = body_stmts or []
        self.workgroups = workgroups or []

    @property
    def loop_var(self):
        """Legacy alias for backward compatibility."""
        return self.loop_vars[0] if self.loop_vars else None

    def accept(self, visitor):
        return visitor.visit_program(self)

    def visit_children(self, visitor) -> None:
        """Visit this program and all its children nodes via the given visitor."""
        self.accept(visitor)
        for param in self.params:
            if hasattr(param, 'accept'):
                param.accept(visitor)
        for stmt in self.body_stmts:
            if hasattr(stmt, 'accept'):
                stmt.accept(visitor)
        for wg in self.workgroups:
            if hasattr(wg, 'accept'):
                wg.accept(visitor)


__all__ = ['Program']
