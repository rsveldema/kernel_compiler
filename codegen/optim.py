"""Optimization passes for Program AST."""

import copy

from codegen.ast.program import Program
from codegen.ast.statement import (
    ForLoopWithConditionAndIncrement,
    Declaration,
    Assignment,
    Condition,
)
from codegen.ast.expression import BinaryExpr, Identifier, Number, ArrayAccess
from codegen.ast.type import Int


BLOCK_SIZE = 8


def perform_blocking(program: Program) -> Program:
    new_body_stmts = []
    for stmt in program.body_stmts:
        if not isinstance(stmt, ForLoopWithConditionAndIncrement):
            new_body_stmts.append(stmt)
            continue
        result = _try_block_loop(stmt, program)
        if result is None:
            new_body_stmts.append(stmt)
        else:
            new_body_stmts.extend(result)
    program.body_stmts = new_body_stmts
    # Mark as tiled so downstream visitors know to adjust dispatch dimensions
    program.tiled = True
    program.tile_block_size = BLOCK_SIZE
    return program


def _try_block_loop(loop, program):
    condition = getattr(loop, "condition", None)
    if not isinstance(condition, Condition):
        return None
    upper_bound = _extract_number(condition.rhs)
    if upper_bound is None or upper_bound < BLOCK_SIZE:
        return None
    loop_var_name = _get_loop_var(loop)
    if not loop_var_name:
        return None
    accu_vars = _find_accumulators_in_loop(loop)
    if not accu_vars:
        return None
    if not _has_inner_product_pattern(loop):
        return None
    if not hasattr(program, "loop_vars") or not program.loop_vars:
        return None
    outer_bounds = _get_outer_bounds(program)
    if not outer_bounds:
        return None
    return _build_tiled_loop(loop, loop_var_name, upper_bound, accu_vars, outer_bounds)


def _build_tiled_loop(original_loop, loop_var_name, upper_bound, accu_vars, outer_bounds):
    body_stmts = []

    # Reset accumulators using Assignment to avoid variable shadowing
    for var in accu_vars:
        body_stmts.append(Assignment(
            lvalue=Identifier(var),
            assign_op="=",
            rvalue=Number(0),
        ))

    tile_loops = []
    last_dim_idx = len(outer_bounds) - 1

    for dim_idx in range(last_dim_idx):
        outer_var, bound = outer_bounds[dim_idx]
        num_tiles = (bound + BLOCK_SIZE - 1) // BLOCK_SIZE
        tl = ForLoopWithConditionAndIncrement(
            loop_var_type=_make_int_type(),
            loop_var_name="tile_" + outer_var,
            condition=Condition(Identifier("tile_" + outer_var), "<", Number(num_tiles)),
            increment_var="tile_" + outer_var,
            increment_op="++",
            init_expr=Number(0),
            body_stmts=[],
        )
        tile_loops.append(tl)

    last_outer_var, last_bound = outer_bounds[-1]
    num_tiles_last = (last_bound + BLOCK_SIZE - 1) // BLOCK_SIZE

    inner_tile_loop = ForLoopWithConditionAndIncrement(
        loop_var_type=_make_int_type(),
        loop_var_name="tile_" + last_outer_var,
        condition=Condition(Identifier("tile_" + last_outer_var), "<", Number(num_tiles_last)),
        increment_var="tile_" + last_outer_var,
        increment_op="++",
        init_expr=Number(0),
        body_stmts=list(body_stmts),
    )

    block_var_type = Int("int")
    
    # block_start/block_end cover the FULL original reduction dimension
    # (not just BLOCK_SIZE) so each workgroup computes one output element
    orig_upper_bound = _extract_number(getattr(original_loop, "condition", None))
    if orig_upper_bound is not None:
        pass  # use upper_bound below
    block_range = upper_bound if upper_bound else 1024
    
    inner_tile_loop.body_stmts.append(Declaration(
        is_const=True,
        var_type=block_var_type,
        name="block_start",
        init_expr=Number(0),
    ))

    inner_tile_loop.body_stmts.append(Declaration(
        is_const=True,
        var_type=block_var_type,
        name="block_end",
        init_expr=Number(block_range),
    ))

    replacement_loop = _build_replacement_loop(original_loop, loop_var_name)
    inner_tile_loop.body_stmts.append(replacement_loop)

    current_innermost = inner_tile_loop
    for tl in reversed(tile_loops):
        new_tl = ForLoopWithConditionAndIncrement(
            loop_var_type=tl.loop_var_type,
            loop_var_name=tl.loop_var_name,
            condition=Condition(getattr(tl.condition, "lhs", Identifier("x")),
                              getattr(tl.condition, "op", "<"),
                              Number(getattr(tl.condition, "rhs", Number(0)).value)),
            increment_var=tl.increment_var,
            increment_op="++",
            init_expr=Number(0),
            body_stmts=[],
        )
        new_tl.body_stmts.append(copy.deepcopy(current_innermost))
        current_innermost = new_tl

    return [current_innermost]


def _build_replacement_loop(original_loop, loop_var_name):
    condition = getattr(original_loop, "condition", None)
    upper_bound = _extract_number(getattr(condition, "rhs", None)) if condition else 0
    if not upper_bound:
        upper_bound = 1024

    # Use Int("int") for the inner loop variable to avoid mixing uint64_t with int 
    # in array index expressions (e.g., A[(1024 * i) + k] where i is int)
    
    inner = ForLoopWithConditionAndIncrement(
        loop_var_type=Int("int"),
        loop_var_name="k",
        condition=Condition(Identifier("k"), "<", Identifier("block_end")),
        increment_var="k",
        increment_op="++",
        init_expr=Identifier("block_start"),
        body_stmts=[],
    )

    transformed = []
    for s in original_loop.body_stmts:
        new_s = _transform_block_stmt(s, loop_var_name)
        if new_s is not None:
            transformed.append(new_s)

    inner.body_stmts = transformed
    return inner


def _transform_block_stmt(stmt, loop_var_name):
    if stmt is None:
        return None
    new_stmt = copy.deepcopy(stmt)
    if isinstance(new_stmt, Declaration):
        init_expr = getattr(new_stmt, "init_expr", None)
        if init_expr and _refers_to_var(init_expr, loop_var_name):
            return None
        if init_expr:
            new_stmt.init_expr = _replace_refs_in_expr(init_expr, loop_var_name)
    elif isinstance(new_stmt, Assignment):
        lvalue = getattr(new_stmt, "lvalue", None)
        rvalue = getattr(new_stmt, "rvalue", None)
        if lvalue:
            new_stmt.lvalue = _replace_refs_in_expr(lvalue, loop_var_name)
        if rvalue:
            new_stmt.rvalue = _replace_refs_in_expr(rvalue, loop_var_name)
    return new_stmt


def _replace_refs_in_expr(expr, old_name):
    if expr is None:
        return None
    if isinstance(expr, Identifier) and getattr(expr, "name", "") == old_name:
        return Identifier("k")
    new_expr = copy.deepcopy(expr)
    for attr in ("lhs", "rhs", "left", "right", "operand", "base"):
        if hasattr(new_expr, attr):
            child = getattr(new_expr, attr)
            setattr(new_expr, attr, _replace_refs_in_expr(child, old_name))
    if hasattr(new_expr, "indices"):
        new_indices = []
        for idx in new_expr.indices:
            new_indices.append(_replace_refs_in_expr(idx, old_name))
        new_expr.indices = new_indices
    return new_expr


def _refers_to_var(expr, var_name):
    if expr is None:
        return False
    if isinstance(expr, Identifier) and getattr(expr, "name", "") == var_name:
        return True
    for attr in ("lhs", "rhs", "left", "right", "base"):
        child = getattr(expr, attr, None)
        if _refers_to_var(child, var_name):
            return True
    if hasattr(expr, "indices"):
        for idx in expr.indices:
            if _refers_to_var(idx, var_name):
                return True
    return False


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


def _make_int_type():
    from codegen.ast.type import Int
    return Int("int")


def _extract_number(expr):
    if expr is None:
        return None
    if isinstance(expr, Number):
        val = getattr(expr, "value", None)
        if val is not None and isinstance(val, (int, float)):
            return int(val)
    return None
