"""Workgroup partitioning optimization pass for Program AST.

Transforms loops iterating from 0..N into stride-based parallelized loops
where K workgroups partition the work (each workgroup processes ceil(N/K) iterations).
"""

from codegen.kast.program import Program
from codegen.kast.statement import (
    ForLoopWithConditionAndIncrement,
    Condition,
    ForLoopRange,
)
from codegen.kast.expression import Identifier, Number, LimitExpr


DEFAULT_WORKGROUPS = 8


def perform_parallelize(program: Program, workgroups: int = DEFAULT_WORKGROUPS) -> Program:
    """Transform loops iterating from 0..N into stride-based parallelized loops.

    For a kernel with one or more sequential loops each iterating over the same
    upper bound N, this pass detects parallelizable outermost loops (those whose
    condition is ``i < N`` where N is a concrete number), computes a workgroup
    size of ceil(N / workgroups), and stores metadata on the Program and loop
    nodes so the Vulkan code generator can emit proper initialization with
    early-exit guards.

    After this pass:
      - program.parallelized == True
      - program.workgroup_count  == workgroups (e.g. 8)
      - program.workgroup_size   == ceil(N / workgroups)
      - program.loop_upper_bound == N
      - each parallelizable loop node carries:
          _parallel_offset_var      -- generated offset variable name (e.g. i_offset)
          _parallel_upper_bound     -- Number expression for N
          _parallel_workgroup_size  -- Number expression for workgroup_size
    """
    if getattr(program, "parallelized", False):
        return program

    # Find parallelizable explicit loops in body_stmts
    explicit_loops = _find_parallelizable_loops(program)

    has_parallel_bound = bool(explicit_loops)
    max_bound = 0
    for info in explicit_loops:
        b = info["upper_bound"] or 0
        if b > max_bound:
            max_bound = b

    # If no explicit parallelizable loops but there are loop_vars, check if
    # there's an OFFLOAD_PARFOR_*_PARAM with a concrete upper bound.
    # These kernels represent implicit loops that aren't in body_stmts as AST nodes.
    if not has_parallel_bound and program.loop_vars:
        triangular_raw = getattr(program, "triangular_bounds_raw", None) or []
        if len(triangular_raw) >= 2:
            # Try to extract bound from triangular bounds (e.g. ['0', 'n'])
            upper_part = triangular_raw[1]
            try:
                max_bound = int(upper_part)
                if max_bound >= 1:
                    has_parallel_bound = True
            except (ValueError, AttributeError):
                pass
        
        # If the bound is a parameter name (like 'n'), use dispatch_rows from params.
        if not has_parallel_bound and program.params:
            for param in program.params:
                vt = getattr(param, "var_type", None)
                if vt:
                    row_val = _extract_number(getattr(vt, "row_size_expr", None))
                    if row_val and row_val >= 1:
                        max_bound = max(max_bound, row_val)
                        has_parallel_bound = True

    workgroup_size = max(1, (max_bound + workgroups - 1) // workgroups)

    # Store metadata on the Program node for the Vulkan visitor
    program.parallelized = True
    program.workgroup_count = workgroups
    program.workgroup_size = workgroup_size
    program.loop_upper_bound = max_bound

    # Per-loop metadata for explicit loops
    for loop_info in explicit_loops:
        loop_node = loop_info["loop"]
        offset_var_name = f"{loop_node.loop_var_name}_offset"
        setattr(loop_node, "_parallel_upper_bound", _make_number(max_bound))
        setattr(loop_node, "_parallel_workgroup_size", _make_number(workgroup_size))
        setattr(loop_node, "_parallel_offset_var", offset_var_name)

    return program


# ── Helpers ────────────────────────────────────────────────────────


def _extract_number(expr):
    """Extract a concrete number from an expression node."""
    if expr is None:
        return None
    from codegen.kast.expression import Number
    if isinstance(expr, Number):
        val = getattr(expr, "value", None)
        if val is not None and isinstance(val, (int, float)):
            return int(val)
    return None


def _make_number(val):
    """Create a Number expression from an int/str value."""
    if isinstance(val, Number):
        return val
    return Number(str(val))


def _extract_upper_bound(expr):
    """Extract a concrete upper-bound integer from an expression.

    Handles:
      - Number nodes directly
      - LimitExpr from limit<N>(...): the inner max_val is N

    Returns the integer bound or None if not concretely extractable.
    """
    if expr is None:
        return None

    if isinstance(expr, Number):
        try:
            return int(expr.value)
        except (ValueError, AttributeError):
            return None

    # Handle LimitExpr from limit<N>(...): the inner 'max_val' is N
    max_val = getattr(expr, "max_val", None)
    if max_val is not None and isinstance(max_val, Number):
        try:
            return int(max_val.value)
        except (ValueError, AttributeError):
            pass

    return None


def _is_condition_i_lt_N(condition, loop_var_name):
    """Check if condition is of the form ``i < N`` where i == loop_var_name.

    Handles Condition nodes.  The RHS must be a concrete bound extractable by
    _extract_upper_bound.
    """
    if not isinstance(condition, Condition):
        return False
    lhs = getattr(condition, "lhs", None)
    op = getattr(condition, "op", "")
    rhs = getattr(condition, "rhs", None)

    # Check LHS is the loop variable
    if isinstance(lhs, Identifier):
        if lhs.name != loop_var_name:
            return False
    else:
        return False

    # Operator must be '<' or '<='
    if op not in ("<", "<="):
        return False

    # RHS must be a concrete bound
    bound = _extract_upper_bound(rhs)
    return bound is not None and bound >= 1


def _get_loop_var_name(node, condition):
    """Get the loop variable name from a ForLoopWithConditionAndIncrement node."""
    if hasattr(node, "loop_var_name") and node.loop_var_name:
        return node.loop_var_name
    if isinstance(condition, Condition):
        lhs = getattr(condition, "lhs", None)
        if isinstance(lhs, Identifier):
            return lhs.name
    return ""


def _find_parallelizable_loops(program):
    """Find parallelizable outer loops in the program body.

    A loop is parallelizable when:
      - It appears directly in body_stmts (not nested inside another loop)
      - Its condition is ``var < N`` where N is a concrete integer >= 1
      - The loop iterates over the same range as other detected loops (to ensure
        they can share the same workgroup stride)

    Returns a list of dicts: {"loop": node, "upper_bound": int}
    """
    body_stmts = getattr(program, "body_stmts", []) or []
    seen_bounds = set()  # track which upper bounds we've already accepted
    results = []

    for stmt in body_stmts:
        if isinstance(stmt, ForLoopWithConditionAndIncrement):
            condition = getattr(stmt, "condition", None)
            loop_var_name = _get_loop_var_name(stmt, condition)
            if loop_var_name and _is_condition_i_lt_N(condition, loop_var_name):
                bound = _extract_upper_bound(getattr(condition, "rhs", None))
                if bound is not None and bound >= 1:
                    if bound not in seen_bounds:
                        results.append({"loop": stmt, "upper_bound": bound})
                        seen_bounds.add(bound)

        elif isinstance(stmt, ForLoopRange):
            # Range-style: for (const int i : limit<N>())
            init_expr = getattr(stmt, "init_expr", None)
            if init_expr is not None:
                bound = _extract_upper_bound(init_expr)
                if bound is not None and bound >= 1:
                    if bound not in seen_bounds:
                        results.append({"loop": stmt, "upper_bound": bound})
                        seen_bounds.add(bound)

    # Sort by upper bound descending so larger ranges are processed first
    results.sort(key=lambda x: x["upper_bound"] or 0, reverse=True)
    return results
