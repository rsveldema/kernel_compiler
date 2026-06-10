"""Visitor that resolves multi-dimensional array indices to linear addresses.

Transforms indexed identifiers like ``a[i,j,k]`` into single-indexed forms
using pre-computed strides from the program's parameter type information.

Examples::

    d_scores[hi, i, k]        -> d_scores[(hi)*(16384*16384) + (i)*(16384) + k]
    A1[i, k]                  -> A1[(i)*(1024) + k]
    matrix[x, y, z]           -> matrix[(x)*s2 + (y)*s1 + z]       (3-D)

If a dimension's stride is unknown at compile time the node is left untouched.
"""

from __future__ import annotations

from typing import Dict

from codegen.ast.program import *
from codegen.ast.expression import *
from codegen.ast.statement import *
from codegen.ast.type import *


class ResolveArrayIndicesVisitor(Expression):
    """Walks an AST tree and replaces multi-dimensional ArrayAccesss with
    their linear-address equivalent based on the parameter type map.
    """

    def __init__(self, params: list[Declaration]):
        self._param_map: Dict[str, Declaration] = {}
        for p in params:
            if isinstance(p, Declaration) and p.name:
                self._param_map[p.name] = p

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _extract_size(node) -> int | None:
        """Return an integer size from a Number; None otherwise."""
        if node is None:
            return None
        if isinstance(node, Number):
            val = node.value
            if isinstance(val, (int, float)):
                return int(val)
        return None

    def _compute_strides(self, var_type) -> list[int] | None:
        """Compute row-major strides for a type's dimensions.

        Returns a list of stride multipliers (one per dimension), outermost-first.
        For row-major order with shape [d0, d1, ..., dn]:
            stride[i] = prod(d(i+1) ... dn), with stride[last] = 1
        """
        if isinstance(var_type, FlexibleRowsMatrix):
            # 2-D: [rows, cols] -> strides [cols, 1]
            col = self._extract_size(var_type.col_size_expr)
            row = self._extract_size(var_type.row_size_expr)
            if col is not None and row is not None:
                return [col, 1]

        elif isinstance(var_type, FixedSizeMatrix):
            # 2-D: [rows, cols] -> strides [cols, 1]
            col = self._extract_size(var_type.col_size_expr)
            row = self._extract_size(var_type.row_size_expr)
            if col is not None and row is not None:
                return [col, 1]

        elif isinstance(var_type, FixedSizeLevelsRowsColsMatrix):
            # 3-D: [levels, rows, cols] -> strides [rows*cols, cols, 1]
            col = self._extract_size(var_type.col_size_expr)
            row = self._extract_size(var_type.row_size_expr)
            lvl = self._extract_size(var_type.level_expr)
            if col is not None and row is not None and lvl is not None:
                s0 = row * col   # stride for levels dim
                s1 = col          # stride for rows dim
                return [s0, s1, 1]  # 3 strides for all dimensions

        elif isinstance(var_type, FlexibleRowsColsLevelsMatrix):
            # 3-D: [rows, cols, levels] -> strides [cols*levels, levels, 1]
            col = self._extract_size(var_type.col_size_expr)
            lvl = self._extract_size(var_type.level_expr)
            row = self._extract_size(var_type.row_size_expr)
            if col is not None and lvl is not None and row is not None:
                s0 = col * lvl   # stride for rows dim
                s1 = lvl          # stride for cols dim
                return [s0, s1, 1]  # 3 strides for all dimensions

        elif isinstance(var_type, FixedSizeVector):
            size = self._extract_size(var_type.size_expr)
            if size is not None:
                return [1]
            return []

        return None

    def _make_linear_index(self, ident: ArrayAccess):
        """Mutate *ident* so that its indices are replaced with linear-address expressions.

        The base identifier is preserved so the node remains valid as both an lvalue and rvalue.
        Returns the same *ident* (mutated in place), or the original *ident* unchanged when
        strides are unknown or there are too few indices.
        """
        if len(ident.indices) <= 1:
            return ident

        # Extract the base identifier name (could be bare Identifier or FieldAccess)
        base_name = None
        if isinstance(ident.base, Identifier):
            base_name = ident.base.name
        elif isinstance(ident.base, FieldAccess):
            b = ident.base.base
            while isinstance(b, FieldAccess):
                b = b.base
            if isinstance(b, Identifier):
                base_name = b.name
        
        var_type = self._param_map.get(base_name).var_type \
                   if base_name and base_name in self._param_map else None

        strides = self._compute_strides(var_type) if var_type is not None else None
        n_dims = len(ident.indices)

        if strides is None or len(strides) < n_dims - 1:
            return ident

        # Build new indices: each original index is multiplied by its stride.
        resolved = []
        for idx, stride in zip(ident.indices, strides):
            term = BinaryExpr(
                left=Number(stride),
                op="*",
                right=self._wrap_expr(idx),
            )
            resolved.append(term)

        # Combine into a single linear index expression: idx0*s0 + idx1*s1 + ... + idxN*sN
        if not resolved:
            ident.indices = []
            return ident
        
        linear_idx = resolved[0]
        for part in resolved[1:]:
            linear_idx = BinaryExpr(left=linear_idx, op="+", right=part)
        
        # Store the combined linear index as the only index.
        ident.indices = [linear_idx]
        return ident

    def _wrap_expr(self, expr: Expression) -> Expression:
        """Wrap *expr* for use inside a parenthesized context."""
        return expr

    # ------------------------------------------------------------------ visitors

    # -- Type visitors (return unchanged) --
    def visit_type(self, node):
        return node

    def visit_int(self, node):
        return Int()

    def visit_float(self, node):
        return Float()

    def visit_fixed_size_vector(self, node):
        elem = node.elem_type.accept(self) if node.elem_type else None
        size = node.size_expr.accept(self) if node.size_expr else None
        return FixedSizeVector(elem_type=elem, size_expr=size)

    def visit_flexible_rows_matrix(self, node):
        elem = node.elem_type.accept(self) if node.elem_type else None
        row_s = node.row_size_expr.accept(self) if node.row_size_expr else None
        col_s = node.col_size_expr.accept(self) if node.col_size_expr else None
        return FlexibleRowsMatrix(elem_type=elem, row_size_expr=row_s, col_size_expr=col_s)

    def visit_fixed_size_matrix(self, node):
        elem = node.elem_type.accept(self) if node.elem_type else None
        row_s = node.row_size_expr.accept(self) if node.row_size_expr else None
        col_s = node.col_size_expr.accept(self) if node.col_size_expr else None
        return FixedSizeMatrix(elem_type=elem, row_size_expr=row_s, col_size_expr=col_s)

    def visit_fixed_size_levels_rows_cols_matrix(self, node):
        elem = node.elem_type.accept(self) if node.elem_type else None
        lvl = node.level_expr.accept(self) if node.level_expr else None
        row_s = node.row_size_expr.accept(self) if node.row_size_expr else None
        col_s = node.col_size_expr.accept(self) if node.col_size_expr else None
        return FixedSizeLevelsRowsColsMatrix(elem_type=elem, level_expr=lvl, row_size_expr=row_s, col_size_expr=col_s)

    def visit_flexible_rows_cols_levels_matrix(self, node):
        elem = node.elem_type.accept(self) if node.elem_type else None
        lvl = node.level_expr.accept(self) if node.level_expr else None
        row_s = node.row_size_expr.accept(self) if node.row_size_expr else None
        col_s = node.col_size_expr.accept(self) if node.col_size_expr else None
        return FlexibleRowsColsLevelsMatrix(elem_type=elem, level_expr=lvl, row_size_expr=row_s, col_size_expr=col_s)

    # -- Expression visitors --
    def visit_expression(self, node):
        return node.accept(type(node).__bases__[0] if type(node).__bases__ else type(node))

    def visit_number(self, node):
        return Number(value=node.value)

    def visit_identifier(self, node):
        return Identifier(name=node.name)

    def visit_array_access(self, node: ArrayAccess):
        """Resolve multi-dimensional indices to linear addresses in place."""
        resolved = self._make_linear_index(node)
        if resolved is not node:
            # _make_linear_index did not mutate – visit children normally
            base = node.base.accept(self) if node.base else None
            indices = []
            changed = False
            for idx in node.indices:
                r = idx.accept(self)
                indices.append(r)
                if r is not idx:
                    changed = True
            if changed or base is not node.base:
                return ArrayAccess(base=base, indices=indices)
            return node
        # Mutated in place – already resolved
        return node

    def visit_limit_expr(self, node: LimitExpr):
        max_val = node.max_val.accept(self) if node.max_val else None
        body = node.body.accept(self) if node.body else None
        return LimitExpr(max_val=max_val, body=body)

    def visit_binary_expr(self, node: BinaryExpr):
        left = node.left.accept(self) if node.left else None
        right = node.right.accept(self) if node.right else None
        return BinaryExpr(left=left, op=node.op, right=right)

    def visit_cast_expr(self, node: CastExpr):
        operand = node.operand.accept(self) if node.operand else None
        return CastExpr(cast_type=node.cast_type, operand=operand)

    def visit_negation_expr(self, node: NegationExpr):
        operand = node.operand.accept(self) if node.operand else None
        return NegationExpr(operand=operand)

    # -- Condition visitors --
    def visit_condition(self, node: Condition):
        lhs = node.lhs.accept(self) if node.lhs else None
        rhs = node.rhs.accept(self) if node.rhs else None
        return Condition(lhs=lhs, op=node.op, rhs=rhs)

    # -- Statement visitors --
    def visit_statement(self, node):
        return node.accept(type(node).__bases__[0] if type(node).__bases__ else type(node))

    def visit_for(self, node: For):
        condition = node.condition.accept(self) if node.condition else None
        init_expr = node.init_expr.accept(self) if node.init_expr else None
        body_stmts = []
        for s in node.body_stmts:
            if hasattr(s, 'accept'):
                result = s.accept(self)
                body_stmts.append(result)
            else:
                body_stmts.append(s)
        return For(
            loop_var_type=node.loop_var_type,
            loop_var_name=node.loop_var_name,
            condition=condition,
            increment_var=node.increment_var,
            increment_op=node.increment_op,
            body_stmts=body_stmts,
            init_expr=init_expr,
        )

    def visit_if(self, node: If):
        condition = node.condition.accept(self) if node.condition else None
        body_stmts = []
        for s in node.body_stmts:
            if hasattr(s, 'accept'):
                result = s.accept(self)
                body_stmts.append(result)
            else:
                body_stmts.append(s)
        return If(condition=condition, body_stmts=body_stmts)

    def visit_declaration(self, node: Declaration):
        init_expr = node.init_expr.accept(self) if node.init_expr else None
        return Declaration(
            is_const=node.is_const,
            var_type=node.var_type,
            name=node.name,
            init_expr=init_expr,
        )

    def visit_assignment(self, node: Assignment):
        lvalue = node.lvalue.accept(self) if node.lvalue else None
        rvalue = node.rvalue.accept(self) if node.rvalue else None
        return Assignment(lvalue=lvalue, assign_op=node.assign_op, rvalue=rvalue)

    def visit_overflow_check(self, node: OverflowCheck):
        lvalue = node.lvalue.accept(self) if node.lvalue else None
        operand = node.operand.accept(self) if node.operand else None
        return OverflowCheck(lvalue=lvalue, operand=operand)

    def visit_shared_decl(self, node: SharedDecl):
        init_expr = node.init_expr.accept(self) if node.init_expr else None
        return SharedDecl(
            is_const=node.is_const,
            var_type=node.var_type,
            name=node.name,
            init_expr=init_expr,
        )

    def visit_workgroup_properties(self, node: WorkgroupProperties):
        x = node.x_expr.accept(self) if node.x_expr else None
        y = node.y_expr.accept(self) if node.y_expr else None
        z = node.z_expr.accept(self) if node.z_expr else None
        return WorkgroupProperties(x_expr=x, y_expr=y, z_expr=z)

    def visit_program(self, node: Program):
        """Transform an entire Program tree."""
        params = []
        for p in node.params:
            if hasattr(p, 'accept'):
                result = p.accept(self)
                params.append(result)
            else:
                params.append(p)

        body_stmts = []
        for s in node.body_stmts:
            if hasattr(s, 'accept'):
                result = s.accept(self)
                body_stmts.append(result)
            else:
                body_stmts.append(s)

        workgroups = []
        for w in node.workgroups:
            if hasattr(w, 'accept'):
                result = w.accept(self)
                workgroups.append(result)
            else:
                workgroups.append(w)

        return Program(
            header=node.header,
            loop_vars=node.loop_vars,
            space_dim=node.space_dim,
            grid_name=node.grid_name,
            limit_expr=node.limit_expr.accept(self) if node.limit_expr else None,
            dispatch_size_expr=node.dispatch_size_expr.accept(self) if node.dispatch_size_expr else None,
            lower_bound_expr=node.lower_bound_expr.accept(self) if getattr(node, 'lower_bound_expr', None) else None,
            upper_bound_expr=node.upper_bound_expr.accept(self) if getattr(node, 'upper_bound_expr', None) else None,
            triangular_bounds_raw=getattr(node, 'triangular_bounds_raw', []),
            params=params or node.params,
            body_stmts=body_stmts or node.body_stmts,
            workgroups=workgroups or node.workgroups,
        )


def resolve_array_indices(program: Program) -> Program:
    """Transform a Program AST so that all multi-dimensional ArrayAccesss
    are replaced with their linear-address equivalents.

    Uses parameter type information from *program.params* to compute strides for
    each supported matrix/vector type.  Indices that cannot be resolved (unknown
    dimensions) are left unchanged.
    """
    visitor = ResolveArrayIndicesVisitor(program.params)
    return program.accept(visitor)
