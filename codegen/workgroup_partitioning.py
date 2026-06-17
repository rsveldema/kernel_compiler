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


def perform_tiling(program: Program, workgroups: int = DEFAULT_WORKGROUPS) -> Program:
    """Transform loops iterating from 0..N into stride-based parallelized loops.
    We'll introduce this in small steps that are semantics preserving and verifiable.

    step 0: check if the kernel contains loops of the form for (int i = 0; i < N; i++) where N is a concrete integer bound.
    step 1: mark the program as tiled and set a tile_block_size (e.g., 8) if such loops are found, but do not yet transform the loops.
    step 2: wrap each non for loop statement S in an:
                if (is_first_in_local_workgroup()) S
                workgroup_barrier()
            guard
    step 3: each loop is transformed to operate on a portion of the loop. 
            For example, given a loop inside a kernel like: 
                for (int i = 0; i < N; i++)
            we transform it to 
                start_i = local_id * chunk_size
                end_i = start_i + chunk_size
                for (int i = start_i; i < end_i; i++) 
                    ...
            where chunk_size is "ceil(N / workgroups)".
    step 4: each statement X += Y we assume is a reduction and transform it to a local reduction across the workgroup
            for example:
                float sum = 0.0f;
                for (int i=0; i<N; i++)
                    sum += ...Y[i]...
            becomes
                float sum = 0.0f;
                float local_X[number_of_workgroup_threads];
                local_X[local_id] = 0.0f;
                for (partition of loop)
                    local_X[local_id] += ...Y[i]...
                workgroup_barrier()

                stride = number_of_workgroup_threads / 2
                for (int k = 0; k < log2(number_of_workgroup_threads); k++)
                    if (is_even_thread_id())
                        // perform local reduction in parallel. Thread 0 adds thread 1, thread 2 adds thread 3, etc. until we have the final sum in local_X[0]
                        if ((local_id + stride) < number_of_workgroup_threads)
                            local_X[local_id] += local_X[local_id + stride]
                        stride /= 2
                    workgroup_barrier()
    step 5: the Vulkan backend will detect the tiled loops and emit appropriate workgroup sizes
    """

    if not program.contains_fixed_size_loops():
        return program



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
