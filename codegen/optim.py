"""Optimization passes for Program AST."""

from codegen.kast.program import Program
from codegen.kast.statement import (
    ForLoopWithConditionAndIncrement,
    Assignment,
    Condition,
)
from codegen.kast.expression import BinaryExpr, Identifier, Number
from codegen.kast.workgroup import WorkgroupProperties


BLOCK_SIZE = 8


def perform_blocking(program: Program, chunk_size: int = BLOCK_SIZE) -> Program:
    blocked = False
    reduction_bound = 0
    for stmt in program.body_stmts:
        if not isinstance(stmt, ForLoopWithConditionAndIncrement):
            continue
        if _can_block_loop(stmt, program):
            blocked = True
            condition = getattr(stmt, "condition", None)
            reduction_bound = _extract_number(getattr(condition, "rhs", None)) or reduction_bound

    # Mark as tiled only when a blockable inner-product loop was found.  The
    # Vulkan backend lowers this to a real local workgroup size and emits a
    # cooperative shared-memory tile for recognized matrix reductions.
    program.tiled = blocked
    program.tile_block_size = BLOCK_SIZE if blocked else 1
    program.use_shared_memory_tiling = blocked and program.space_dim == 2
    program.shared_memory_chunk_size = chunk_size if program.use_shared_memory_tiling else 1
    program.reduction_chunk_size = chunk_size if program.use_shared_memory_tiling else 0
    if program.use_shared_memory_tiling and reduction_bound:
        program.reduction_chunks = (reduction_bound + chunk_size - 1) // chunk_size
    else:
        program.reduction_chunks = 1
    if blocked and not program.workgroups:
        y_size = BLOCK_SIZE if program.space_dim >= 2 else 1
        program.workgroups.append(
            WorkgroupProperties(
                x_expr=Number(BLOCK_SIZE),
                y_expr=Number(y_size),
                z_expr=Number(1),
            )
        )
    return program


def perform_cooperative_matrix2(program: Program, chunk_size: int = BLOCK_SIZE) -> Program:
    blocked = False
    for stmt in program.body_stmts:
        if not isinstance(stmt, ForLoopWithConditionAndIncrement):
            continue
        if _can_block_loop(stmt, program):
            blocked = True

    program.tiled = blocked
    program.tile_block_size = BLOCK_SIZE if blocked else 1
    program.use_shared_memory_tiling = False
    program.shared_memory_chunk_size = 1
    program.reduction_chunk_size = 0
    program.reduction_chunks = 1
    program.use_cooperative_matrix2 = blocked and program.space_dim == 2
    program.cooperative_matrix2_chunk_size = chunk_size if program.use_cooperative_matrix2 else 1
    if blocked and not program.workgroups:
        program.workgroups.append(
            WorkgroupProperties(
                x_expr=Number(BLOCK_SIZE),
                y_expr=Number(BLOCK_SIZE),
                z_expr=Number(1),
            )
        )
    return program


def _can_block_loop(loop, program):
    condition = getattr(loop, "condition", None)
    if not isinstance(condition, Condition):
        return False
    upper_bound = _extract_number(condition.rhs)
    if upper_bound is None or upper_bound < BLOCK_SIZE:
        return False
    loop_var_name = _get_loop_var(loop)
    if not loop_var_name:
        return False
    accu_vars = _find_accumulators_in_loop(loop)
    if not accu_vars:
        return False
    if not _has_inner_product_pattern(loop):
        return False
    if not hasattr(program, "loop_vars") or not program.loop_vars:
        return False
    outer_bounds = _get_outer_bounds(program)
    if not outer_bounds:
        return False
    return True


def _get_loop_var(loop):
    if hasattr(loop, "loop_var_name") and loop.loop_var_name:
        return loop.loop_var_name
    condition = getattr(loop, "condition", None)
    if isinstance(condition, Condition):
        lhs = getattr(condition, "lhs", None)
        if isinstance(lhs, Identifier):
            return lhs.name
    return ""


def _get_outer_bounds(program):
    bounds = []
    for expr_attr in ("limit_expr", "dispatch_size_expr"):
        if hasattr(program, expr_attr):
            val = _extract_number(getattr(program, expr_attr))
            if val is not None:
                for name in program.loop_vars:
                    bounds.append((name, val))
                return bounds
    if hasattr(program, "params") and program.params:
        for param in program.params:
            vt = getattr(param, "var_type", None)
            row_s = getattr(vt, "row_size_expr", None) if vt else None
            col_s = getattr(vt, "col_size_expr", None) if vt else None
            row_val = _extract_number(row_s) if row_s else None
            col_val = _extract_number(col_s) if col_s else None
            if row_val and col_val and hasattr(program, "loop_vars") and program.loop_vars:
                for name in program.loop_vars[:1]:
                    bounds.append((name, row_val))
                remaining = program.loop_vars[1:] if len(program.loop_vars) > 1 else [program.loop_vars[-1]]
                for name in remaining:
                    bounds.append((name, col_val))
                return bounds
    return bounds


def _find_accumulators_in_loop(loop):
    accu = []
    seen = set()
    for s in loop.body_stmts:
        if isinstance(s, Assignment):
            lvalue = getattr(s, "lvalue", None)
            op = getattr(s, "assign_op", "")
            if isinstance(lvalue, Identifier) and op in ("+=", "-="):
                name = lvalue.name
                if name not in seen:
                    accu.append(name)
                    seen.add(name)
    return accu


def _has_inner_product_pattern(loop):
    for s in loop.body_stmts:
        if _contains_binary_mult(s):
            return True
    return False


def _contains_binary_mult(node):
    if node is None:
        return False
    if isinstance(node, BinaryExpr) and node.op == "*":
        return True
    for attr in ("left", "right", "lhs", "rhs", "base"):
        child = getattr(node, attr, None)
        if _contains_binary_mult(child):
            return True
    if hasattr(node, "init_expr") and node.init_expr:
        if isinstance(node.init_expr, BinaryExpr):
            if node.init_expr.op == "*":
                return True
            if _contains_binary_mult(node.init_expr):
                return True
    if hasattr(node, "rvalue") and node.rvalue:
        if isinstance(node.rvalue, BinaryExpr) and node.rvalue.op == "*":
            return True
        if _contains_binary_mult(node.rvalue):
            return True
    return False


def _extract_number(expr):
    if expr is None:
        return None
    if isinstance(expr, Number):
        val = getattr(expr, "value", None)
        if val is not None and isinstance(val, (int, float)):
            return int(val)
    return None
