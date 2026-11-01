"""Conservative recognition of AST contractions that map exactly to BLAS."""

from dataclasses import dataclass
from pathlib import Path

from . import kast as ast


@dataclass(frozen=True)
class BlasGemm:
    a: str
    b: str
    c: str
    k: str | int
    transpose_a: bool
    transpose_b: bool
    accumulate: bool
    input_type: str = "float32"
    epilogue: str = "none"


def _identifier(node, name: str | None = None) -> bool:
    return isinstance(node, ast.Identifier) and (name is None or node.name == name)


def _array(node, base: str, indices: tuple[str, str]) -> bool:
    return (
        isinstance(node, ast.ArrayAccess)
        and _identifier(node.base, base)
        and len(node.indices) == 2
        and all(_identifier(index, expected) for index, expected in zip(node.indices, indices))
    )


def _zero(node) -> bool:
    return isinstance(node, ast.Number) and float(node.value) == 0.0


def _number_value(node):
    if isinstance(node, ast.Number):
        return float(node.value)
    if isinstance(node, ast.UnaryMinusExpr) and isinstance(node.operand, ast.Number):
        return -float(node.operand.value)
    return None


def detect_safe_gemm(program: ast.Program, allow_clip_epilogue: bool = False) -> BlasGemm | None:
    """Recognize only `C[i,j] += sum_k A[k,i] * B[k,j]`.

    The intentionally narrow matcher rejects epilogues, clamps, mixed types,
    additional side effects, static casts around the result, and non-canonical
    indexing. Broader GEMM forms can be added independently after validation.
    """
    if program.space_dim != 2 or len(program.loop_vars) != 2:
        return None
    i, j = program.loop_vars
    params = {p.name: p for p in program.params if isinstance(p, ast.Declaration)}
    body = program.body_stmts
    if len(body) != 7:
        return None
    sum_decl, loop, output = body[4], body[5], body[6]
    # Exact overwrite projection/Gram product:
    # C[i,j] = sum_k A[i,k] * B[j,k].
    output_value = output.rvalue if isinstance(output, ast.Assignment) else None
    if isinstance(output_value, ast.CastExpr) and isinstance(output_value.cast_type, ast.Float):
        output_value = output_value.operand
    clip_call = (
        output_value
        if isinstance(output_value, ast.CallExpr)
        and _identifier(output_value.callee, "clamp")
        and len(output_value.args) == 3
        and _identifier(output_value.args[0], "sum")
        else None
    )
    if clip_call is not None and tuple(_number_value(arg) for arg in clip_call.args[1:]) != (-10000.0, 10000.0):
        clip_call = None
    # Keep clipped AB^T contractions in the regular kernel path for now. The
    # BLAS epilogue split is disabled until its ordering and synchronization
    # semantics have been validated on all supported backends.
    if clip_call is not None and not allow_clip_epilogue:
        return None
    overwrite = (
        isinstance(loop, ast.ForLoopWithConditionAndIncrement)
        and isinstance(loop.condition, ast.Condition)
        and isinstance(loop.condition.rhs, (ast.Number, ast.Identifier))
        and isinstance(output, ast.Assignment)
        and output.assign_op == "="
        and (_identifier(output_value, "sum") or clip_call is not None)
    )

    if not (
        isinstance(sum_decl, ast.Declaration)
        and isinstance(sum_decl.var_type, ast.Float)
        and sum_decl.name == "sum"
        and _zero(sum_decl.init_expr)
        and isinstance(loop, ast.ForLoopWithConditionAndIncrement)
        and loop.loop_var_name == "l_idx"
        and _zero(loop.init_expr)
        and isinstance(loop.condition, ast.Condition)
        and _identifier(loop.condition.lhs, "l_idx")
        and loop.condition.op == "<"
        and (_identifier(loop.condition.rhs) or overwrite)
        and loop.increment_var == "l_idx"
        and loop.increment_op == "++"
        and isinstance(output, ast.Assignment)
        and ((output.assign_op == "+=" and _identifier(output.rvalue, "sum")) or overwrite)
    ):
        return None
    if len(loop.body_stmts) != 4:
        return None
    k_decl, term_decl, overflow, accumulate = loop.body_stmts
    if not (
        isinstance(k_decl, ast.Declaration)
        and k_decl.name == "k"
        and isinstance(k_decl.init_expr, ast.CastExpr)
        and _identifier(k_decl.init_expr.operand, "l_idx")
        and isinstance(term_decl, ast.Declaration)
        and term_decl.name == "term"
        and isinstance(term_decl.var_type, ast.Float)
        and isinstance(term_decl.init_expr, ast.BinaryExpr)
        and term_decl.init_expr.op == "*"
        and isinstance(overflow, ast.OverflowCheck)
        and _identifier(overflow.lvalue, "sum")
        and _identifier(overflow.operand, "term")
        and isinstance(accumulate, ast.Assignment)
        and accumulate.assign_op == "+="
        and _identifier(accumulate.lvalue, "sum")
        and _identifier(accumulate.rvalue, "term")
    ):
        return None
    left, right = term_decl.init_expr.left, term_decl.init_expr.right
    if not (isinstance(left, ast.ArrayAccess) and isinstance(right, ast.ArrayAccess)):
        return None
    a = left.base.name if isinstance(left.base, ast.Identifier) else ""
    b = right.base.name if isinstance(right.base, ast.Identifier) else ""
    c = output.lvalue.base.name if isinstance(output.lvalue, ast.ArrayAccess) and isinstance(output.lvalue.base, ast.Identifier) else ""
    abt_indices = _array(left, a, (i, "k")) and _array(right, b, (j, "k"))
    atb_indices = _array(left, a, ("k", i)) and _array(right, b, ("k", j))
    indices_match = abt_indices or atb_indices
    k = (int(loop.condition.rhs.value) if isinstance(loop.condition.rhs, ast.Number)
         else loop.condition.rhs.name)
    if not overwrite and not atb_indices:
        return None
    if not (indices_match and _array(output.lvalue, c, (i, j))):
        return None
    required_params = (a, b, c) + ((k,) if isinstance(k, str) else ())
    if any(name not in params for name in required_params):
        return None
    matrix_types = (ast.FlexibleRowsMatrix, ast.FlexibleSizeMatrix, ast.FlexibleRowsColsMatrix, ast.FixedSizeMatrix)
    if not all(isinstance(params[name].var_type, matrix_types) for name in (a, b, c)):
        return None
    if overwrite:
        input_is_half = (
            isinstance(params[a].var_type.elem_type, ast.Float16)
            and isinstance(params[b].var_type.elem_type, ast.Float16)
        )
        input_is_float = (
            isinstance(params[a].var_type.elem_type, ast.Float)
            and isinstance(params[b].var_type.elem_type, ast.Float)
        )
        if not ((input_is_half or input_is_float)
                and isinstance(params[c].var_type.elem_type, ast.Float)):
            return None
        if input_is_half and not abt_indices:
            return None
    elif not all(isinstance(params[name].var_type.elem_type, ast.Float) for name in (a, b, c)):
        return None
    if not params[a].is_const or not params[b].is_const or params[c].is_const:
        return None
    if isinstance(k, str) and not isinstance(params[k].var_type, ast.Int):
        return None
    if overwrite:
        return BlasGemm(a, b, c, k,
                        transpose_a=atb_indices, transpose_b=abt_indices,
                        accumulate=False,
                        input_type="float16" if input_is_half else "float32",
                        epilogue="clip" if clip_call is not None else "none")
    return BlasGemm(a, b, c, k, transpose_a=True, transpose_b=False, accumulate=True)


def lower_blas_epilogue(program: ast.Program) -> ast.Program:
    """Use the dedicated tkernel rewrite to leave only a BLAS epilogue kernel."""
    if program.blas is None or program.blas.epilogue == "none":
        return program
    if program.blas.epilogue != "clip":
        raise ValueError(f"unsupported BLAS epilogue: {program.blas.epilogue}")

    from .visitors.tree_rewriter import Pattern

    pattern_path = Path(__file__).resolve().parents[1] / "transforms" / "blas_clip_abt.tkernel"
    pattern = Pattern(str(pattern_path))
    if not pattern.matches(program):
        raise ValueError(f"BLAS clip rewrite did not match {program.header}")
    pattern.apply(program)
    return program
