"""Workgroup partitioning optimization pass for Program AST.

Transforms loops iterating from 0..N into stride-based parallelized loops
where K workgroups partition the work (each workgroup processes ceil(N/K) iterations).
"""

from codegen.kast.program import Program
from codegen.kast.workgroup import WorkgroupProperties
from codegen.kast.statement import (
    ForLoopWithConditionAndIncrement,
    Condition,
    ForLoopRange,
)
from codegen.kast.expression import Identifier, Number, LimitExpr, CastExpr, FieldAccess
from codegen.kast.statement import Declaration
from codegen.kast.type import Int
from codegen.visitors.tree_rewriter import TreeRewriter


DEFAULT_WORKGROUPS = 8


def perform_tiling(program: Program, workgroups: int = DEFAULT_WORKGROUPS) -> Program:
    """Transform loops iterating from 0..N into stride-based parallelized loops.
    We'll introduce this in small steps that are semantics preserving and verifiable.

    step 0: check if the kernel contains loops of the form for (int i = 0; i < N; i++) where N is a concrete integer bound.
    step 1: mark the program as tiled and set a tile_block_size (e.g., 8) if such loops are found, but do not yet transform the loops.
    step 2: use the tree_rewriter:
            
            search 
                wildcard_statement(S)
            replace 
                if (is_first_in_local_workgroup()) 
                    wildcard_statement(S)
                workgroup_barrier()
            constraints S != ForStatement

    step 3:  use the tree_rewriter: each loop is transformed to operate on a portion of the loop. 

            search
                for (int i = 0; i < N; i++)
                    wildcard_statement(S)
            replace
                int chunk_size = ceil(N / workgroups)
                int start_i = local_id * chunk_size
                int end_i = start_i + chunk_size
                for (int i = start_i; i < end_i; i++) 
                    wildcard_statement(S)
    
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

    loops = _find_parallelizable_loops(program)
    if len(loops) == 0:
        return program

    parallelizable_loop_bounds = {entry["upper_bound"] for entry in loops}

    program.tiled = True
    program.tile_block_size = workgroups
    program.workgroup_count = workgroups
    program.workgroup_size = workgroups

    # Export explicit workgroup size to both GLSL and C++ stub so they are always in sync.
    # Only set when no WorkgroupProperties already exist
    if not any(isinstance(wg, WorkgroupProperties) for wg in program.workgroups):
        space_dim = getattr(program, "space_dim", 0) or len(getattr(program, "loop_vars", []))
        if space_dim >= 2:
            wg_x, wg_y, wg_z = 16, 16, 1
        else:
            wg_x, wg_y, wg_z = 16, 1, 1
        program.workgroups.append(
            WorkgroupProperties(
                x_expr=Number(str(wg_x)),
                y_expr=Number(str(wg_y)),
                z_expr=Number(str(wg_z)),
            )
        )

    rewrite_context = {
        "workgroups": workgroups,
        "target_loop_bounds": parallelizable_loop_bounds,
        "tile_size": program.tile_block_size if hasattr(program, "tile_block_size") and program.tile_block_size else 8,
        "chunk_size": program.shared_memory_chunk_size if hasattr(program, "shared_memory_chunk_size") and program.shared_memory_chunk_size else 128,
        "bound_k": getattr(program, "bound_k", 1024),
        "cooperative_matrix2_chunk_size": program.cooperative_matrix2_chunk_size if hasattr(program, "cooperative_matrix2_chunk_size") and program.cooperative_matrix2_chunk_size else 16,
    }
    rewritten_program = TreeRewriter(rewrite_context).visit_program(program)
    body_stmts = getattr(rewritten_program, "body_stmts", []) or []

    prefix_stmts = [
        Declaration(
            is_const=True,
            var_type=Int(),
            name="rllm_wg_count",
            init_expr=Number(str(workgroups)),
        ),
        Declaration(
            is_const=True,
            var_type=Int(),
            name="local_id",
            init_expr=CastExpr(Int(), FieldAccess(Identifier("gl_LocalInvocationID"), "x")),
        ),
    ]

    program.body_stmts = prefix_stmts + body_stmts

    return program


# ── Helpers ────────────────────────────────────────────────────────


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


def _extract_upper_bound_key(expr):
    concrete = _extract_upper_bound(expr)
    if concrete is not None:
        return concrete

    if isinstance(expr, Identifier):
        return expr.name

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

    return _extract_upper_bound_key(rhs) is not None


def _get_loop_var_name(node, condition):
    """Get the loop variable name from a ForLoopWithConditionAndIncrement node."""
    if hasattr(node, "loop_var_name") and node.loop_var_name:
        return node.loop_var_name
    if isinstance(condition, Condition):
        lhs = getattr(condition, "lhs", None)
        if isinstance(lhs, Identifier):
            return lhs.name
    return ""


def _find_parallelizable_loops(program) -> list[dict]:
    """Find parallelizable outer loops in the program body.

    A loop is parallelizable when:
      - It appears directly in body_stmts (not nested inside another loop)
      - Its condition is ``var < N`` where N is a concrete integer >= 1 or
        an identifier bound such as k_count
      - The loop iterates over the same range as other detected loops (to ensure
        they can share the same workgroup stride)

    Returns a list of dicts: {"loop": node, "upper_bound": int | str}
    """
    body_stmts = getattr(program, "body_stmts", []) or []
    seen_bounds = set()  # track which upper bounds we've already accepted
    results = []

    for stmt in body_stmts:
        if isinstance(stmt, ForLoopWithConditionAndIncrement):
            condition = getattr(stmt, "condition", None)
            loop_var_name = _get_loop_var_name(stmt, condition)
            if loop_var_name and _is_condition_i_lt_N(condition, loop_var_name):
                bound = _extract_upper_bound_key(getattr(condition, "rhs", None))
                if bound is not None:
                    if bound not in seen_bounds:
                        results.append({"loop": stmt, "upper_bound": bound})
                        seen_bounds.add(bound)

        elif isinstance(stmt, ForLoopRange):
            # Range-style: for (const int i : limit<N>())
            init_expr = getattr(stmt, "init_expr", None)
            if init_expr is not None:
                bound = _extract_upper_bound_key(init_expr)
                if bound is not None:
                    if bound not in seen_bounds:
                        results.append({"loop": stmt, "upper_bound": bound})
                        seen_bounds.add(bound)

    # Sort by upper bound descending so larger ranges are processed first
    results.sort(key=lambda x: str(x["upper_bound"]), reverse=True)
    return results
