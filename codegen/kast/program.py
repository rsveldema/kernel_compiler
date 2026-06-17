"""Program AST node for code generation."""

from codegen.kast.ast_node import AstNode


class Program(AstNode):
    def __init__(
        self,
        header="",
        loop_vars=None,
        space_dim=0,
        grid_name="",
        limit_expr=None,
        dispatch_size_expr=None,
        lower_bound_expr=None,
        upper_bound_expr=None,
        triangular_bounds_raw=None,  # [raw_lower_str, raw_upper_str] for triangular
        triangular_kind="",
        params=None,
        body_stmts=None,
        workgroups=None,
        tiled=False,
        tile_block_size=1,
        reduction_chunk_size=0,
        reduction_chunks=1,
        reduction_chunk_var="",
        use_shared_memory_tiling=False,
        shared_memory_chunk_size=1,
        use_cooperative_matrix2=False,
        cooperative_matrix2_chunk_size=8,
        # Set by perform_tiling() for workgroup partitioning
        parallelized=False,
        workgroup_count=1,
        workgroup_size=1,
        _source_filename="",
        _constexpr_defines=None,
        _param_constexpr_defines=None,
    ):
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
        self.triangular_kind = triangular_kind
        self.params = params or []
        self.body_stmts = body_stmts or []
        self.workgroups = workgroups or []
        # Set by perform_blocking() when tiling is applied
        self.tiled = tiled
        self.tile_block_size = tile_block_size
        self.reduction_chunk_size = reduction_chunk_size
        self.reduction_chunks = reduction_chunks
        self.reduction_chunk_var = reduction_chunk_var
        self.use_shared_memory_tiling = use_shared_memory_tiling
        self.shared_memory_chunk_size = shared_memory_chunk_size
        self.use_cooperative_matrix2 = use_cooperative_matrix2
        self.cooperative_matrix2_chunk_size = cooperative_matrix2_chunk_size
        self._source_filename = _source_filename
        # constexpr defines extracted from kernel body [(name, expr), ...]
        self._constexpr_defines = _constexpr_defines or []
        # constexpr defines from parameter declarations (for type resolution)
        self._param_constexpr_defines = _param_constexpr_defines or []
        # Set by perform_tiling() for workgroup partitioning
        self.parallelized = parallelized
        self.workgroup_count = workgroup_count
        self.workgroup_size = workgroup_size

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
            if hasattr(param, "accept"):
                param.accept(visitor)
        for stmt in self.body_stmts:
            if hasattr(stmt, "accept"):
                stmt.accept(visitor)
        for wg in self.workgroups:
            if hasattr(wg, "accept"):
                wg.accept(visitor)


__all__ = ["Program"]
