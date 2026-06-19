"""Transform Lark trees to codegen_ast nodes.

Extracted from parser.py so that parser.py only contains grammar, parser,
parse() and a thin transform wrapper.
"""

from lark import Tree, Token
from .visitors.resolve_array_indices import resolve_array_indices
from .kast.program import *
from .kast.expression import *
from .kast.type import *
from .kast.statement import *

# ── helpers ────────────────────────────────────────────────────────

def _ast_to_str(expr):
    """Convert an AST expression to its string representation."""
    if isinstance(expr, Number):
        return str(expr.value)
    if isinstance(expr, Identifier):
        return expr.name
    if isinstance(expr, BinaryExpr):
        left = _ast_to_str(expr.left)
        right = _ast_to_str(expr.right)
        op = expr.op
        return f"({left} {op} {right})"
    if hasattr(expr, 'max_val'):  # LimitExpr
        inner = _ast_to_str(expr.max_val)
        return f"limit<{inner}>()"
    return str(expr)


def _is_token(val):
    return isinstance(val, Token)


def _token_value(t):
    if isinstance(t, Token):
        return t.value
    if isinstance(t, Tree) and len(t.children) == 1:
        child = t.children[0]
        if isinstance(child, Token):
            return child.value
    return None



_TYPE_NAMES = frozenset((
    "int", "size_t", "float", "float16",
    "fixed_size_vector", "flexible_rows_matrix", "fixed_size_matrix",
    "fixed_size_triangular_matrix",
    "flexible_size_matrix", "flexible_cols_matrix", "fixed_size_levels_rows_cols_matrix",
    "flexible_rows_cols_levels_matrix", "fixed_size_obj_vector",
))


def _is_type_name(token_val):
    """Check if a token value is a known type name (not just any IDENT)."""
    return token_val in _TYPE_NAMES


def _extract_op_token_from_tree(tree):
    """Extract an operator Token from a possibly deeply nested Tree structure.

    The grammar nests operators like: binary_operator -> arith_operator -> Token('*')
    This unwraps until it finds a Token or can't go deeper.
    Returns the Token value if found, otherwise None.
    """
    current = tree
    while isinstance(current, Tree) and len(current.children) == 1:
        child = current.children[0]
        if isinstance(child, Token):
            return child.value
        if isinstance(child, Tree):
            current = child
        else:
            break
    # Reached a Tree with children or non-Tree terminal
    if isinstance(current, Tree) and len(current.children) == 1:
        child = current.children[0]
        if isinstance(child, Token):
            return child.value
    return None

# ── extractors ─────────────────────────────────────────────────────


def extract_header(header_tree):
    """Extract string value and workgroup properties from header tree.

    Header tree has: [string, workgroup_properties*, ...]
    Returns (header_string, list_of_workgroup_properties_trees)
    """
    header_str = ""
    workgroups = []
    for child in header_tree.children:
        if isinstance(child, Tree):
            if child.data == "string":
                header_str = _token_value(child.children[0])
            elif child.data == "workgroup_properties":
                workgroups.append(child)
        elif _is_token(child) and child.type == "ESCAPED_STRING":
            header_str = _token_value(child)
    return (header_str, workgroups)


def extract_loop_vars_and_dim(space_tree):
    """Extract all loop variable names and dimensionality from a parfor_space tree.

    For 1D (OFFLOAD_PARFOR_1D_PARAM(i, ...)): loop_vars=[i], dim=1
    For 2D (OFFLOAD_PARFOR_2D_PARAM(i, j, ...)): loop_vars=[i, j], dim=2
    For 3D triangular (OFFLOAD_PARFOR_3D_TRIANGULAR_PARAM(hi, i, j, ...)):
        loop_vars=[hi, i, j], dim=3
    """
    ids = []
    for c in space_tree.children:
        if _is_token(c) and c.type == "IDENT":
            ids.append(c.value)
            if len(ids) >= 3:
                break

    dim = min(len(ids), 3)
    return ids, dim


def _extract_grid_name_from_expr(expr_tree):
    """Extract a single identifier name from an expression tree (grid/extent name in 2D)."""
    for child in expr_tree.children:
        if isinstance(child, Tree) and child.data == "base_expr":
            for sub in child.children:
                if isinstance(sub, Tree) and sub.data == "lhs":
                    lhs_children = sub.children
                    if lhs_children:
                        first = lhs_children[0]
                        if _is_token(first):
                            return first.value
    for child in expr_tree.children:
        if _is_token(child) and child.type == "IDENT":
            return child.value
    return ""


def _find_expression_trees(space_tree):
    """Find ALL expression children in a parfor_space tree (before parfor_parameters_list)."""
    results = []
    for child in space_tree.children:
        if isinstance(child, Tree) and child.data == "expression":
            results.append(child)
    return results


def _find_expression_tree(space_tree):
    """Find the first expression child in a parfor_space tree (works for 1D)."""
    trees = _find_expression_trees(space_tree)
    return trees[0] if trees else None


def _extract_expression_name(expr_tree):
    """Extract a variable name from an expression tree.

    Returns the identifier/variable name if the expression is a simple field access,
    or its numeric value if it's a number literal.
    """
    for child in expr_tree.children:
        if isinstance(child, Tree) and child.data == "base_expr":
            for sub in child.children:
                if isinstance(sub, Tree):
                    if sub.data == "number":
                        for nc in sub.children:
                            if hasattr(nc, "value"):
                                v = nc.value
                                if isinstance(v, (int, float)):
                                    return str(int(v))
                                # Handle string numeric values from tokens
                                try:
                                    fval = float(str(v).rstrip("fF"))
                                    return str(int(fval))
                                except ValueError:
                                    pass
                        return "0"
                    elif sub.data == "field_access":
                        # Return the identifier name from field_access
                        for sc in sub.children:
                            if isinstance(sc, Token) and sc.type == "IDENT":
                                return sc.value
    # Fallback: try direct tokens
    for child in expr_tree.children:
        if _is_token(child):
            if child.type == "NUMBER":
                v = child.value
                if "." in str(v):
                    return str(int(float(v)))
                return str(v)
            elif child.type == "IDENT":
                return child.value
    return None


def extract_limit_expr(space_tree):
    """Extract limit or triangular bound expressions from parfor_space.

    For 1D/2D (single expression): returns a single expression AST node.
    For 3D triangular (two expressions): returns a tuple of
        (lower_bound_expr, upper_bound_expr, raw_names_list).
        raw_names_list contains the string names/values for push constants.
    """
    expr_trees = _find_expression_trees(space_tree)
    if not expr_trees:
        return None

    if space_tree.data in {"parfor_2d_triangular_param", "parfor_2d_upper_triangular_param"}:
        upper = transform_expression(expr_trees[0])
        raw_name = _extract_expression_name(expr_trees[0])
        kind = "upper" if space_tree.data == "parfor_2d_upper_triangular_param" else "lower"
        return (Number(0), upper, ["0", raw_name], kind)

    if len(expr_trees) == 2:
        # Triangular case: two bounds (lower and upper)
        lower = transform_expression(expr_trees[0])
        upper = transform_expression(expr_trees[1])
        raw_names = [_extract_expression_name(et) for et in expr_trees]
        return (lower, upper, raw_names, "lower")

    expr_tree = expr_trees[0]

    for child in expr_tree.children:
        if isinstance(child, Tree) and child.data == "limit_expr":
            return transform_expression(child)

    for child in expr_tree.children:
        if isinstance(child, Tree) and child.data == "base_expr":
            for sub in child.children:
                if isinstance(sub, Tree) and sub.data == "limit_expr":
                    return transform_expression(sub)

    return None


def _transform_workgroup_properties(wg_tree):
    """Transform workgroup_properties tree to an AST node.

    Two grammar alternatives:
    1. workgroup { x: expr, y: expr, z: expr } -> WorkgroupProperties
    2. shared declaration ";" -> SharedDecl
    """
    if wg_tree.data == "workgroup":
        # workgroup { x: ..., y: ..., z: ... } case
        expressions = []
        for child in wg_tree.children:
            if isinstance(child, Tree):
                expr_result = transform_expression(child)
                if expr_result is not None:
                    expressions.append(expr_result)
        x_expr = expressions[0] if len(expressions) > 0 else None
        y_expr = expressions[1] if len(expressions) > 1 else None
        z_expr = expressions[2] if len(expressions) > 2 else None
        return WorkgroupProperties(x_expr, y_expr, z_expr)

    # shared declaration case
    if isinstance(wg_tree.children[0], Tree):
        decl_tree = wg_tree.children[0]
        is_const = decl_tree.data == "const_decl"
        is_constexpr = decl_tree.data == "constexpr_decl"
        var_type = None
        name = ""
        init_expr = None

        for child in decl_tree.children:
            if isinstance(child, Tree):
                data = child.data
                if data in ("int", "float"):
                    var_type = _resolve_nested_type(child) or Int()
                elif data == "type":
                    resolved = _resolve_nested_type(child)
                    if resolved is not None:
                        var_type = resolved
                    else:
                        var_type = _transform_type(child)
                elif data == "expression":
                    init_expr = transform_expression(child)
                elif data in ("index_expr", "lhs"):
                    lvalue = _transform_lvalue(child)
                    if isinstance(lvalue, Identifier):
                        name = lvalue.name
            elif isinstance(child, Token):
                name = child.value

    return SharedDecl(is_const, var_type or Int(), name, init_expr, is_constexpr=is_constexpr)

    # workgroup { x: ..., y: ..., z: ... } case (from grammar tree)
    expressions = []
    for child in wg_tree.children:
        if isinstance(child, Tree):
            expr_result = transform_expression(child)
            if expr_result is not None:
                expressions.append(expr_result)
    x_expr = expressions[0] if len(expressions) > 0 else None
    y_expr = expressions[1] if len(expressions) > 1 else None
    z_expr = expressions[2] if len(expressions) > 2 else None
    return WorkgroupProperties(x_expr, y_expr, z_expr)


def transform_expression(expr_tree):
    """Convert an expression Tree to an AST node."""
    if expr_tree.data == "wildcard_expression":
        for child in expr_tree.children:
            if _is_token(child) and child.type == "IDENT":
                return WildcardExpression(child.value)

    if expr_tree.data == "inc_call":
        for child in expr_tree.children:
            if isinstance(child, Tree) and child.data == "expression":
                operand = transform_expression(child)
                if operand is not None:
                    return BinaryExpr(operand, "+", Number(1))

    # Direct limit_expr
    if expr_tree.data == "limit_expr":
        expr_children = []
        for child in expr_tree.children:
            if isinstance(child, Tree) and child.data == "expression":
                expr_children.append(child)
            elif isinstance(child, Tree) and child.data == "limit_args":
                expr_children.extend(
                    sub for sub in child.children
                    if isinstance(sub, Tree) and sub.data == "expression"
                )
        max_val = transform_expression(expr_children[0]) if expr_children else None
        if len(expr_children) >= 3:
            return LimitExpr(
                max_val,
                transform_expression(expr_children[1]),
                transform_expression(expr_children[2]),
            )
        if len(expr_children) >= 2:
            return LimitExpr(max_val, transform_expression(expr_children[1]))
        return LimitExpr(max_val, max_val)

    # 'expression' wrapper with binary ops
    if expr_tree.data == "expression":
        children = expr_tree.children
        op_val = None
        ternary_tree = None
        if children and isinstance(children[-1], Tree) and children[-1].data == "ternary_trailer":
            ternary_tree = children[-1]
            children = children[:-1]

        if len(children) == 3 and _is_token(children[1]):
            op_val = children[1].value
        elif (
            len(children) == 3
            and isinstance(children[1], Tree)
        ):
            # Unwrap nested operator tokens (e.g. binary_operator -> arith_operator -> Token('*'))
            token_val = _extract_op_token_from_tree(children[1])
            if token_val is not None:
                op_val = token_val

        if op_val is not None:
            left = transform_expression(children[0])
            right = transform_expression(children[2])
            if left is not None and right is not None:
                expr = BinaryExpr(left, op_val, right)
                if ternary_tree is not None and len(ternary_tree.children) >= 2:
                    return TernaryExpr(
                        expr,
                        transform_expression(ternary_tree.children[0]),
                        transform_expression(ternary_tree.children[1]),
                    )
                return expr

        for child in children:
            if isinstance(child, Tree):
                result = transform_expression(child)
                if result is not None:
                    if ternary_tree is not None and len(ternary_tree.children) >= 2:
                        return TernaryExpr(
                            result,
                            transform_expression(ternary_tree.children[0]),
                            transform_expression(ternary_tree.children[1]),
                        )
                    return result

    # Direct number tree
    if expr_tree.data == "number":
        val_token = expr_tree.children[0]
        try:
            val_str = val_token.value
            unsigned = isinstance(val_str, str) and val_str.endswith(("u", "U"))
            if isinstance(val_str, str):
                val_str = val_str.rstrip("uU")
            if isinstance(val_str, str) and ("." in val_str or "e" in val_str.lower()):
                core = val_str.rstrip("fF")
                if core.endswith("."):
                    core = core + "0"
                return Number(float(core), unsigned=unsigned)
            return Number(int(val_str), unsigned=unsigned)
        except (ValueError, TypeError):
            return None

    # Handle base_expr directly
    if expr_tree.data == "base_expr":
        return _transform_from_base(expr_tree)

    # Bare lhs trees directly
    if expr_tree.data == "lhs":
        return _transform_lvalue(expr_tree)

    return None


def _transform_from_base(base_tree):
    """Transform a base_expr tree into an AST expression."""
    if (
        len(base_tree.children) == 2
        and isinstance(base_tree.children[0], Tree)
        and base_tree.children[0].data == "type"
        and isinstance(base_tree.children[1], Tree)
    ):
        cast_type = _resolve_nested_type(base_tree.children[0])
        operand = transform_expression(base_tree.children[1])
        if operand is not None:
            return CastExpr(cast_type, operand)

    for child in base_tree.children:
        if isinstance(child, Tree):
            data = child.data
            if data == "limit_expr":
                return transform_expression(child)
            elif data == "lhs":
                return _transform_lvalue(child)
            elif data == "field_access":
                # Check for cast expression pattern: type_name(...) where
                # the base IDENT is a known type name and call_args exists.
                # This catches `int(l_idx)` which grammar prefers as field_access.
                wildcard_result = _maybe_parse_wildcard_from_field_access(child)
                if wildcard_result is not None:
                    return wildcard_result
                fa_result = _maybe_parse_cast_from_field_access(child)
                if fa_result is not None:
                    return fa_result
                result = _transform_lvalue(child)
                if result is not None:
                    return result
            elif data == "cast":
                cast_type = _resolve_nested_type(child.children[0])
                operand = transform_expression(child.children[1])
                if operand is not None:
                    return CastExpr(cast_type, operand)
            elif data == "unary_expression":
                op = None
                operand = None
                for sub in child.children:
                    if isinstance(sub, Tree) and sub.data == "unary_operator":
                        op = _extract_op_token_from_tree(sub)
                    elif isinstance(sub, Tree):
                        operand = transform_expression(sub)
                if operand is not None:
                    if op == "-":
                        return UnaryMinusExpr(operand)
                    return NegationExpr(operand)
            elif data == "inc_call":
                for sub in child.children:
                    if isinstance(sub, Tree) and sub.data == "expression":
                        operand = transform_expression(sub)
                        if operand is not None:
                            return BinaryExpr(operand, "+", Number(1))
            else:
                result = transform_expression(child)
                if result is not None:
                    return result

        elif _is_token(child):
            return Identifier(child.value)

    return None


def _maybe_parse_cast_from_field_access(fa_tree):
    """Detect cast pattern in field_access (e.g. int(l_idx)) and return CastExpr if found.

    The grammar prefers `field_access` over the inline cast form
    `type "(" expression ")"` for expressions like `int(l_idx)`.
    This helper detects when the base IDENT is a known type name
    with call_args present, treating it as a cast expression.
    """
    children = fa_tree.children
    if not children:
        return None

    # First child must be a token whose value is a known type name
    first = children[0]
    if not _is_token(first) or not _is_type_name(first.value):
        return None

    # Look for call_args among remaining children
    call_arg_children = []
    for c in children[1:]:
        if isinstance(c, Tree) and c.data == "call_args":
            for cc in c.children:
                if isinstance(cc, Tree) and cc.data == "expression":
                    call_arg_children.append(cc)

    if not call_arg_children:
        return None

    # Get the operand expression from call_args
    operand = transform_expression(call_arg_children[0])
    if operand is None:
        return None

    # Resolve the type name
    cast_type = _resolve_nested_type(fa_tree.children[0]) or Int(first.value)
    return CastExpr(cast_type, operand)


def _maybe_parse_wildcard_from_field_access(fa_tree):
    """Detect wildcard(name) parsed through field_access/call_args."""
    children = fa_tree.children
    if not children:
        return None

    first = children[0]
    is_wildcard = isinstance(first, Tree) and first.data == "wildcard"
    is_wildcard = is_wildcard or (_is_token(first) and first.type == "IDENT" and first.value == "wildcard")
    if not is_wildcard:
        return None

    for child in children[1:]:
        if isinstance(child, Tree) and child.data == "call_args":
            for arg_child in child.children:
                if isinstance(arg_child, Tree) and arg_child.data == "expression":
                    operand = transform_expression(arg_child)
                    if isinstance(operand, Identifier):
                        return WildcardExpression(operand.name)

    return None


def _transform_lvalue(lhs_tree):
    """Transform an LHS tree into an AST lvalue expression.

    Produces FieldAccess for a.b chains, ArrayAccess for a[b,c] arrays,
    or bare Identifier when there's just a name.
    """
    if _is_token(lhs_tree):
        return Identifier(lhs_tree.value)

    data = lhs_tree.data

    # lhs always has exactly one child: a field_access node (grammar: lhs -> field_access)
    if data == "lhs":
        child = lhs_tree.children[0]
        if isinstance(child, Tree) and child.data == "field_access":
            return _parse_field_access_for_lhs(child)
        # fallback for unexpected structure
        return _transform_lvalue(child)

    # Handle field_access nodes directly (can appear in base_expr context)
    if data == "field_access":
        return _parse_field_access_for_lhs(lhs_tree)

    if data == "base_expr":
        return _transform_from_base(lhs_tree)

    return None


def _parse_field_access_for_lhs(fa_tree):
    """Parse a field_access tree into an AST FieldAccess or ArrayAccess.

    The grammar is: IDENT (array_index)? (DOT IDENT)*
    - Bare identifier ``x``  ->  FieldAccess(base=Identifier("x"), fields=[])
    - Simple chain ``obj.x`` ->  FieldAccess(base=Identifier("obj"), fields=["x"])
    - Array access ``a[b,c]`` ->  ArrayAccess(base=FieldAccess(base=Identifier("a")), indices=[...])
    """
    children = fa_tree.children

    # First child is always the base IDENT token
    first = children[0]
    if not _is_token(first) or first.type != "IDENT":
        return None

    base_name = first.value
    field = None
    index_groups = []
    call_args = None

    for child in children[1:]:
        if isinstance(child, Tree) and child.data == "array_index":
            indices = []
            for idx_child in child.children:
                if isinstance(idx_child, Tree):
                    expr_result = transform_expression(idx_child)
                    if expr_result is not None:
                        indices.append(expr_result)
            index_groups.append(indices)
        elif _is_token(child) and child.type == "DOT":
            pass  # separator between field names
        elif _is_token(child) and child.type == "IDENT":
            field = child.value
        elif isinstance(child, Tree) and child.data == "call_args":
            call_args = []
            for arg_child in child.children:
                if isinstance(arg_child, Tree) and arg_child.data == "expression":
                    arg_expr = transform_expression(arg_child)
                    if arg_expr is not None:
                        call_args.append(arg_expr)

    base = Identifier(base_name)

    if call_args is not None:
        assert field is None and not index_groups
        if base_name == "inc" and call_args:
            return BinaryExpr(call_args[0], "+", Number(1))
        return CallExpr(base, call_args)

    if index_groups:
        expr = base
        if field:
            # Pure field chain (no array indexing)
            expr = FieldAccess(base=base, field=field)

        for indices in index_groups:
            expr = ArrayAccess(base=expr, indices=indices)
        return expr

    if field:
        # Pure field chain (no array indexing)
        return FieldAccess(base=base, field=field)

    # Bare identifier - return a bare Identifier for simplicity
    # (FieldAccess with empty fields is equivalent but less clean)
    return base


# ── type transform ─────────────────────────────────────────────────


def _resolve_nested_type(node):
    """Resolve a nested type from a tree node (TYPE_LITERAL or compound type).

    Handles both Tree nodes (from grammar) and direct Token inputs (e.g., from
    field_access detection in cast expressions).
    """
    # Handle direct Token input (e.g., when called from _maybe_parse_cast_from_field_access)
    if isinstance(node, Token):
        val = node.value
        if val in ("int", "size_t"):
            return Int(val)
        elif val in ("float", "rlmm_float"):
            return Float()
        elif val in ("float16", "rlmm_float_small"):
            return Float16()
        return None

    if isinstance(node, Tree):
        data = node.data
        # Handle the simple type case: single TYPE_LITERAL token child
        if data == "type" and len(node.children) == 1 and _is_token(node.children[0]):
            token = node.children[0]
            val = token.value
            if val in ("int", "size_t"):
                return Int(val)
            elif val in ("float", "rlmm_float"):
                return Float()
            elif val in ("float16", "rlmm_float_small"):
                return Float16()
        # Handle compound types (multi-param matrix/vector types)
        if data == "type" and len(node.children) >= 2:
            first_child = node.children[0]
            if _is_token(first_child):
                type_name = first_child.value
                return _transform_type(node, default_name=type_name)

    # Explicitly reject unknown types instead of silently falling back to int
    return None


def _transform_type(type_tree, default_name="int"):
    """Transform a 'type' Lark tree into an AST Type node.

    After the grammar rewrite with explicit type name terminals (FSV, FRM, FSM,
    FSLRMC, FRCLM), each multi-param type produces a Tree(data='type', children=[...])
    where the first child is the type name token and remaining children are:
      elem_type + expression params.
    """
    data = type_tree.data
    children = type_tree.children
    if data != "type" or len(children) < 2:
        return None

    first_child = children[0]

    # Check if this is a multi-param type (first child is terminal token)
    if _is_token(first_child):
        type_name = first_child.value

        expr_children = [
            c for c in children[1:] if isinstance(c, Tree) and c.data == "expression"
        ]

        if "fixed_size_obj_vector" in type_name:
            if len(expr_children) >= 3:
                return FixedSizeObjVectorMatrix(
                    _resolve_nested_type(children[2]),
                    transform_expression(expr_children[2]),
                    transform_expression(expr_children[0]),
                    transform_expression(expr_children[1]),
                )
        elif "fixed_size_levels_rows_cols_matrix" in type_name:
            # 4 params: elem_type + level + row + col
            if len(expr_children) >= 3:
                return FixedSizeLevelsRowsColsMatrix(
                    _resolve_nested_type(children[1]),
                    transform_expression(expr_children[0]),
                    transform_expression(expr_children[1]),
                    transform_expression(expr_children[2]),
                )
        elif "flexible_rows_cols_levels_matrix" in type_name:
            # 4 params: elem_type + level + row + col
            if len(expr_children) >= 3:
                return FlexibleRowsColsLevelsMatrix(
                    _resolve_nested_type(children[1]),
                    transform_expression(expr_children[0]),
                    transform_expression(expr_children[1]),
                    transform_expression(expr_children[2]),
                )
        elif "fixed_size_triangular_matrix" in type_name:
            if len(expr_children) >= 2:
                return FixedSizeTriangularMatrix(
                    _resolve_nested_type(children[1]),
                    transform_expression(expr_children[0]),
                    transform_expression(expr_children[1]),
                )
        elif "flexible_size_matrix" in type_name:
            if len(expr_children) >= 2:
                return FlexibleSizeMatrix(
                    _resolve_nested_type(children[1]),
                    transform_expression(expr_children[0]),
                    transform_expression(expr_children[1]),
                )
        elif "fixed_size_matrix" in type_name:
            # 3 params: elem_type + row + col
            if len(expr_children) >= 2:
                return FixedSizeMatrix(
                    _resolve_nested_type(children[1]),
                    transform_expression(expr_children[0]),
                    transform_expression(expr_children[1]),
                )
        elif "flexible_rows_matrix" in type_name:
            # 3 params: elem_type + row + col
            if len(expr_children) >= 2:
                return FlexibleRowsMatrix(
                    _resolve_nested_type(children[1]),
                    transform_expression(expr_children[0]),
                    transform_expression(expr_children[1]),
                )
        elif "flexible_rows_cols_matrix" in type_name:
            if len(expr_children) >= 2:
                return FlexibleRowsColsMatrix(
                    _resolve_nested_type(children[1]),
                    transform_expression(expr_children[0]),
                    transform_expression(expr_children[1]),
                )
        elif "flexible_cols_matrix" in type_name:
            if len(expr_children) >= 2:
                return FlexibleRowsColsMatrix(
                    _resolve_nested_type(children[1]),
                    transform_expression(expr_children[0]),
                    transform_expression(expr_children[1]),
                )
        elif "fixed_size_vector" in type_name:
            # 2 params: elem_type + size_expr
            if len(expr_children) >= 1:
                return FixedSizeVector(
                    _resolve_nested_type(children[1]),
                    transform_expression(expr_children[0]),
                )
        return None


    # Handle single-child TYPE_LITERAL terminal (scalar types)
    if len(children) == 1 and _is_token(first_child):
        name_val = first_child.value
        if name_val in ("int", "size_t"):
            return Int(name_val)
        elif name_val in ("float", "rlmm_float"):
            return Float()
        elif name_val in ("float16", "rlmm_float_small"):
            return Float16()
        else:
            # Unknown scalar type - fall through to return None for explicit handling
            pass

    # Simple types (data like 'int' or 'float')
    if hasattr(first_child, 'data') and first_child.data in ("int", "float"):
        name_val = _is_token(first_child) and first_child.value or first_child.data
        return Int(name_val) if first_child.data == "int" else Float()

    if hasattr(first_child, 'data') and first_child.data == "type":
        return _transform_type(first_child)

    return None

    # Simple types (data like 'int' or 'float')
# ── statement transform ────────────────────────────────────────────


def _is_overflow_check(stmt_tree):
    """Check if a statement tree represents an OVERFLOW_CHECK_ADD."""
    if stmt_tree.data != "statement":
        return False
    if len(stmt_tree.children) != 2:
        return False
    lhs_child, expr_child = stmt_tree.children
    if not isinstance(lhs_child, Tree) or lhs_child.data != "lhs":
        return False
    if not isinstance(expr_child, Tree):
        return False
    return True


def transform_declaration(stmt_tree):
    """Transform a declaration tree into Declaration AST."""
    is_const = stmt_tree.data == "const_decl"
    is_constexpr = stmt_tree.data == "constexpr_decl"
    var_type = None
    name = ""
    init_expr = None

    extended_type_prefixes = (
        "fixed_size_vector",
        "flexible_rows_matrix",
        "fixed_size_matrix",
        "fixed_size_triangular_matrix",
        "fixed_size_levels_rows_cols_matrix",
        "flexible_rows_cols_levels_matrix",
    )

    for child in stmt_tree.children:
        if isinstance(child, Tree):
            data = child.data
            # Simple types (int/float from grammar rewrite rules)
            if data in ("int", "float"):
                var_type = _resolve_nested_type(child) or Int()
            elif data == "type":
                resolved = _resolve_nested_type(child)
                if resolved is not None:
                    var_type = resolved
                else:
                    var_type = _transform_type(child)
            elif child.data.startswith(extended_type_prefixes):
                # Complex type as direct child (shouldn't happen normally, but handle it)
                var_type = _transform_type(child)
            elif data == "expression":
                init_expr = transform_expression(child)
            elif data in ("index_expr", "lhs"):
                lvalue = _transform_lvalue(child)
                if isinstance(lvalue, Identifier):
                    name = lvalue.name
        elif isinstance(child, Token):
            # Could be the variable name token directly
            name = child.value

    return Declaration(is_const, var_type or Int(), name, init_expr, is_constexpr=is_constexpr)


def transform_statement(stmt_tree):
    """Transform a statement Tree to an AST node."""
    data = stmt_tree.data

    if data == "statement":
        return transform_statement(stmt_tree.children[0])

    if data == "wildcard_statement":
        for child in stmt_tree.children:
            if _is_token(child) and child.type == "IDENT":
                return WildcardStatement(child.value)

    # Overflow check: 2 children [lhs, expression]
    if _is_overflow_check(stmt_tree):
        lhs_child, expr_child = stmt_tree.children
        lvalue = _transform_lvalue(lhs_child)
        operand = transform_expression(expr_child)
        if lvalue is not None and operand is not None:
            return OverflowCheck(lvalue, operand)

    if data == "atomic_op":
        op = "atomicAdd"
        exprs = []
        for child in stmt_tree.children:
            if _is_token(child) and child.type == "ATOMIC_OP":
                op = child.value
            elif isinstance(child, Tree) and child.data == "expression":
                exprs.append(transform_expression(child))
        if len(exprs) >= 2:
            return AtomicOp(exprs[0], exprs[1], op)

    # block: "{" statement* "}"
    # for_statement
    if data == "for_statement":
        # Two grammar alternatives:
        # 1. "for" "(" for_loop_var ";" condition ";" increment ")" (block | statement)
        # 2. "for" "(" "const" type IDENT ":" expression ")" (block | statement)
        body_stmts = []
        # Initialize all variables early to avoid UnboundLocalError
        loop_var_type = None
        loop_var_name = ""
        init_expr_for_var = None
        condition_tree = None
        condition = None
        inc_var = ""
        inc_op = ""

        # Handle loop variable part (children[0] = for_loop_var OR inline type "int"/"float")
        for_loop_part = stmt_tree.children[0]
        if isinstance(for_loop_part, Tree):
            for_lp_data = for_loop_part.data
            if for_lp_data == "for_loop_var":
                # for_loop_var -> (const)? type IDENT "=" expression
                loop_var_type = None
                loop_var_name = ""
                init_expr_for_var = None

                for child in for_loop_part.children:
                    if isinstance(child, Tree):
                        data = child.data
                        if data == "type":
                            loop_var_type = _resolve_nested_type(
                                child
                            ) or _transform_type(child)
                        elif data == "expression":
                            init_expr_for_var = transform_expression(child)
                    elif _is_token(child):
                        loop_var_name = child.value

                # condition from children[1]
                condition_tree = (
                    stmt_tree.children[1] if len(stmt_tree.children) > 1 else None
                )
                condition = None
                if (
                    isinstance(condition_tree, Tree)
                    and condition_tree.data == "condition"
                ):
                    lhs_t = condition_tree.children[0]
                    rhs_t = condition_tree.children[2]
                    lhs = (
                        _transform_lvalue(lhs_t)
                        if _is_token(lhs_t)
                        or (isinstance(lhs_t, Tree) and lhs_t.data in ("lhs",))
                        else None
                    )
                    op_val = ""
                    for c in condition_tree.children:
                        if _is_token(c):
                            op_val = c.value
                        elif isinstance(c, Tree) and len(c.children) == 1:
                            inner = c.children[0]
                            if _is_token(inner):
                                op_val = inner.value

                    rhs = (
                        transform_expression(rhs_t) if isinstance(rhs_t, Tree) else None
                    )
                    if lhs is not None and rhs is not None:
                        condition = Condition(lhs, op_val, rhs)

                # Extract increment from children[2] if present
                inc_tree = (
                    stmt_tree.children[2] if len(stmt_tree.children) > 2 else None
                )
                if isinstance(inc_tree, Tree):
                    for c in inc_tree.children:
                        if _is_token(c) and c.type == "INC_OP":
                            inc_op = c.value
                        elif _is_token(c) and c.type == "IDENT":
                            inc_var = c.value

                # Extract body from the last child of stmt_tree (after all parsed parts)
                body_tree = None
                for bt in reversed(stmt_tree.children):
                    if isinstance(bt, Tree) and bt.data in ("block", "statement"):
                        body_tree = bt
                        break

                if body_tree is not None:
                    result = transform_statement(body_tree)
                    if result is not None:
                        if isinstance(result, list):
                            body_stmts.extend(result)
                        else:
                            body_stmts.append(result)

                return ForLoopWithConditionAndIncrement(
                    loop_var_type=loop_var_type,
                    loop_var_name=loop_var_name,
                    condition=condition,
                    increment_var=inc_var,
                    increment_op=inc_op,
                    body_stmts=body_stmts,
                )


        # Handle inline type variant: Tree(for_statement, [Tree(int/float), IDENT, expr, body])
        if loop_var_type is None and len(stmt_tree.children) >= 4:
            first_child = stmt_tree.children[0]
            if isinstance(first_child, Tree) and (first_child.data in ("int", "float") or first_child.data == "type"):
                # type from children[0]
                # Extract type name from the TYPE_LITERAL token
                type_name = "int"  # default
                if first_child.data == "type":
                    for c in first_child.children:
                        if _is_token(c):
                            type_name = c.value
                            break
                loop_var_type = _resolve_nested_type(first_child) or Int(type_name)

                # Variable name from children[1]
                second_child = stmt_tree.children[1]
                if isinstance(second_child, Token) and second_child.type == "IDENT":
                    loop_var_name = second_child.value

                # Init expr from children[2]
                third_child = stmt_tree.children[2]
                if isinstance(third_child, Tree) and third_child.data == "expression":
                    init_expr_for_var = transform_expression(third_child)

                # Body from children[3] (can be deeply nested statement/block)
                fourth_child = stmt_tree.children[3]
                if isinstance(fourth_child, Tree):

                    def extract_stmts(t):
                        results = []
                        if isinstance(t, Tree):
                            if t.data == "block":
                                for inner in t.children:
                                    results.extend(extract_stmts(inner))
                            elif t.data == "statement":
                                for inner in t.children:
                                    results.extend(extract_stmts(inner))
                            else:
                                results.append(t)
                        return results

                    extracted = extract_stmts(fourth_child)
                    for eb in extracted:
                        if isinstance(eb, Tree):
                            s = transform_statement(eb)
                            if s is not None:
                                if isinstance(s, list):
                                    body_stmts.extend(s)
                                else:
                                    body_stmts.append(s)

                return ForLoopRange(
                    loop_var_type=loop_var_type,
                    loop_var_name=loop_var_name or "",
                    init_expr=init_expr_for_var,
                    body_stmts=body_stmts,
                )

                # increment from children[2]
                inc_tree = (
                    stmt_tree.children[2] if len(stmt_tree.children) > 2 else None
                )
                inc_var = ""
                inc_op = ""
                if isinstance(inc_tree, Tree) and inc_tree.data == "increment":
                    for c in inc_tree.children:
                        if _is_token(c) and c.type == "IDENT":
                            inc_var = c.value
                        elif _is_token(c) and c.type == "INC_OP":
                            inc_op = c.value

                # body from children[3] (block or statement)
                body_tree = (
                    stmt_tree.children[3] if len(stmt_tree.children) > 3 else None
                )
                if isinstance(body_tree, Tree):
                    if body_tree.data == "block":
                        for bc in body_tree.children:
                            s = transform_statement(bc)
                            if s is not None:
                                body_stmts.append(s)
                    else:
                        s = transform_statement(body_tree)
                        if s is not None:
                            body_stmts.append(s)

                return ForLoopWithConditionAndIncrement(
                    loop_var_type=loop_var_type,
                    loop_var_name=loop_var_name,
                    condition=condition,
                    increment_var=inc_var,
                    increment_op=inc_op,
                    body_stmts=body_stmts,
                )

        # Handle inline for loop variant (from "for" "(" <type> IDENT ":" expr ")" body)
        if not loop_var_type and len(stmt_tree.children) >= 2:
            first_child = stmt_tree.children[0]

            is_inline_type = False
            type_node = None

            # Direct inline type at for_statement level
            if isinstance(first_child, Tree) and (first_child.data in ("int", "float") or first_child.data == "type"):
                is_inline_type = True
                type_node = first_child
                loop_var_name = (
                    str(stmt_tree.children[1].value)
                    if len(stmt_tree.children) > 1
                    and hasattr(stmt_tree.children[1], "value")
                    else ""
                )

            elif isinstance(first_child, Tree) and first_child.data == "for_statement":
                # Nested for_statement (shouldn't normally happen but handle it)
                inner = first_child
                if len(inner.children) > 0 and isinstance(inner.children[0], Tree):
                    inner_first = inner.children[0]
                    if inner_first.data in ("int", "float"):
                        is_inline_type = True
                        type_node = inner_first
                        loop_var_name = (
                            str(inner.children[1].value)
                            if len(inner.children) > 1
                            and hasattr(inner.children[1], "value")
                            else ""
                        )

            # Also check: if first_child.data == 'int'/'float', also get name from stmt_tree children
            if isinstance(first_child, Tree) and (first_child.data in ("int", "float") or first_child.data == "type"):
                is_inline_type = True
                type_node = first_child
                if len(stmt_tree.children) > 1:
                    second = stmt_tree.children[1]
                    if hasattr(second, "type") and second.type == "IDENT":
                        loop_var_name = str(second.value)

            if is_inline_type and type_node is not None:
                is_inline_type = True
                # Extract type name from the TYPE_LITERAL token
                type_name = "int"  # default
                if first_child.data == "type":
                    for c in first_child.children:
                        if _is_token(c):
                            type_name = c.value
                            break
                loop_var_type = _resolve_nested_type(first_child) or Int(type_name)

                # Variable name - look at appropriate child
                var_child = None
                if is_inline_type and type_node is not None:
                    # Check if first_child was a for_statement wrapper
                    if (
                        isinstance(stmt_tree.children[0], Tree)
                        and stmt_tree.children[0].data == "for_statement"
                    ):
                        inner_fs = stmt_tree.children[0]
                        var_child = (
                            inner_fs.children[1] if len(inner_fs.children) > 1 else None
                        )
                    else:
                        var_child = (
                            stmt_tree.children[1]
                            if len(stmt_tree.children) > 1
                            else None
                        )
                else:
                    var_child = (
                        stmt_tree.children[1] if len(stmt_tree.children) > 1 else None
                    )

                if isinstance(var_child, Token) and var_child.type == "IDENT":
                    loop_var_name = var_child.value

                # Init expression from third child
                if len(stmt_tree.children) > 2:
                    third_child = stmt_tree.children[2]
                    if isinstance(third_child, Tree):
                        init_expr_for_var = (
                            transform_expression(third_child)
                            if third_child.data == "expression"
                            else None
                        )
                        condition_tree = (
                            third_child if third_child.data == "condition" else None
                        )

                # Condition from fourth child
                if len(stmt_tree.children) > 3:
                    fourth_child = stmt_tree.children[3]
                    if isinstance(fourth_child, Tree):
                        if fourth_child.data == "expression":
                            init_expr_for_var = transform_expression(fourth_child)
                        elif fourth_child.data == "condition":
                            condition_tree = fourth_child

                # Body from fifth child (can be deeply nested statement/block)
                if len(stmt_tree.children) > 4:
                    fifth_child = stmt_tree.children[4]
                    if isinstance(fifth_child, Tree):

                        def extract_stmts(t):
                            results = []
                            if isinstance(t, Tree):
                                if t.data == "block":
                                    for inner in t.children:
                                        results.extend(extract_stmts(inner))
                                elif t.data == "statement":
                                    for inner in t.children:
                                        results.extend(extract_stmts(inner))
                                else:
                                    results.append(t)
                            return results

                        extracted = extract_stmts(fifth_child)
                        for eb in extracted:
                            if isinstance(eb, Tree):
                                s = transform_statement(eb)
                                if s is not None:
                                    if isinstance(s, list):
                                        body_stmts.extend(s)
                                    else:
                                        body_stmts.append(s)

                # Parse condition if found
                if condition_tree is not None and len(condition_tree.children) >= 3:
                    cond_children = condition_tree.children
                    lhs_t = cond_children[0]
                    rhs_t = cond_children[2]
                    op_val = ""
                    for c in cond_children:
                        if _is_token(c):
                            op_val = c.value
                        elif isinstance(c, Tree) and len(c.children) == 1:
                            inner = c.children[0]
                            if _is_token(inner):
                                op_val = inner.value

                    lhs = None
                    rhs = (
                        transform_expression(rhs_t) if isinstance(rhs_t, Tree) else None
                    )
                    if hasattr(lhs_t, "type") and _is_token(lhs_t):
                        lhs_name = getattr(lhs_t, "value", "")
                        if lhs_name:
                            lhs = Identifier(lhs_name)

                    if lhs is not None and rhs is not None:
                        condition = Condition(lhs, op_val, rhs)

                return ForLoopWithConditionAndIncrement(
                    loop_var_type=loop_var_type,
                    loop_var_name=loop_var_name or "",
                    condition=condition,
                    increment_var=inc_var,
                    increment_op=inc_op,
                    body_stmts=body_stmts,
                )
            elif for_lp_data == "for":
                # Alternative 2: "for" "(" <type> IDENT ":" expression ")" (block | statement)
                # First child is type (could be 'int', 'float', or 'type' tree)
                first_child = (
                    for_loop_part.children[0]
                    if len(for_loop_part.children) > 0
                    else None
                )

                loop_var_type = None
                init_expr_for_var = None
                condition_tree = None

                if isinstance(first_child, Tree):
                    fdata = first_child.data
                    if fdata in ("int", "float"):
                        loop_var_type = (
                            _resolve_nested_type(first_child) or Int()
                            if fdata == "int"
                            else Float()
                        )
                    elif fdata == "type":
                        loop_var_type = _transform_type(first_child)
                    # First child could also be a condition tree (for alternate grammar variant)
                    elif first_child.data == "condition":
                        condition_tree = first_child

                # Second child is the variable name (IDENT token)
                if len(for_loop_part.children) > 1:
                    second = for_loop_part.children[1]
                    if _is_token(second):
                        loop_var_name = second.value

                # Third child could be init expression or condition
                if len(for_loop_part.children) > 2:
                    third = for_loop_part.children[2]
                    if isinstance(third, Tree) and third.data == "expression":
                        init_expr_for_var = transform_expression(third)
                    elif isinstance(third, Tree) and third.data == "condition":
                        condition_tree = third

                # Body from remaining children (handle both block and statement bodies)
                body_found = False
                for bc_idx in range(3, len(for_loop_part.children)):
                    bc = for_loop_part.children[bc_idx]
                    if isinstance(bc, Tree):
                        extracted_blocks = []

                        # Unwrap nested statement/block layers to get actual statements
                        def extract_from_tree(t):
                            results = []
                            if isinstance(t, Tree):
                                if t.data == "block":
                                    for inner in t.children:
                                        results.extend(extract_from_tree(inner))
                                elif t.data == "statement":
                                    # statement has children which might contain the block
                                    for inner in t.children:
                                        results.extend(extract_from_tree(inner))
                                else:
                                    results.append(t)
                            return results

                        extracted_blocks = extract_from_tree(bc)

                        for eb in extracted_blocks:
                            if isinstance(eb, Tree):
                                s = transform_statement(eb)
                                if s is not None:
                                    if isinstance(s, list):
                                        body_stmts.extend(s)
                                    else:
                                        body_stmts.append(s)
                        body_found = True

                # Also check for condition in remaining children
                if not body_found:
                    for cond_idx in range(3, len(for_loop_part.children)):
                        cond_child = for_loop_part.children[cond_idx]
                        if (
                            isinstance(cond_child, Tree)
                            and cond_child.data == "condition"
                        ):
                            condition_tree = cond_child
                            # Deeply nested block - extract inner statements
                            inner_block = (
                                fourth.children[0] if len(fourth.children) > 0 else None
                            )
                            if inner_block and isinstance(inner_block, Tree):
                                for bc in inner_block.children:
                                    s = transform_statement(bc)
                                    if s is not None:
                                        body_stmts.append(s)
                        elif fourth.data == "statement":
                            # Statement containing block - extract from it
                            sub_block = (
                                fourth.children[0] if len(fourth.children) > 0 else None
                            )
                            if (
                                sub_block
                                and isinstance(sub_block, Tree)
                                and sub_block.data == "block"
                            ):
                                for bc in sub_block.children:
                                    s = transform_statement(bc)
                                    if s is not None:
                                        body_stmts.append(s)
                        elif fourth.data == "condition":
                            # Handle condition parsing (IDENT compare_operator expression)
                            cond_children = fourth.children
                            if len(cond_children) >= 3:
                                lhs_t = cond_children[0]
                                rhs_t = cond_children[2]
                                op_val = ""
                                for c in cond_children:
                                    if _is_token(c):
                                        op_val = c.value
                                    elif isinstance(c, Tree) and len(c.children) == 1:
                                        inner = c.children[0]
                                        if _is_token(inner):
                                            op_val = inner.value
                                lhs = None
                                rhs = (
                                    transform_expression(rhs_t)
                                    if isinstance(rhs_t, Tree)
                                    else None
                                )
                                if hasattr(lhs_t, "type") and _is_token(lhs_t):

                                    # Try to get identifier for comparison
                                    lhs_name = getattr(lhs_t, "value", "")
                                    if lhs_name:
                                        pass  # Use string directly in condition

                return ForLoopRange(
                    loop_var_type=loop_var_type,
                    loop_var_name=loop_var_name or "",
                    init_expr=init_expr_for_var if 'init_expr_for_var' in locals() else None,
                    body_stmts=body_stmts,
                )

    # if_statement
    if data == "if_statement":
        condition_tree = stmt_tree.children[0] if len(stmt_tree.children) > 0 else None
        condition = None
        if isinstance(condition_tree, Tree) and condition_tree.data == "condition":
            lhs_t = condition_tree.children[0]
            rhs_t = condition_tree.children[2]
            op_val = ""
            for c in condition_tree.children:
                if _is_token(c):
                    op_val = c.value
                elif isinstance(c, Tree) and len(c.children) == 1:
                    inner = c.children[0]
                    if _is_token(inner):
                        op_val = inner.value

            lhs = (
                _transform_lvalue(lhs_t)
                if _is_token(lhs_t)
                or (isinstance(lhs_t, Tree) and (lhs_t.data == "lhs" or True))
                else None
            )
            rhs = transform_expression(rhs_t) if isinstance(rhs_t, Tree) else None
            if lhs is not None and rhs is not None:
                condition = Condition(lhs, op_val, rhs)

        body_stmts = []
        if len(stmt_tree.children) > 1:
            body_tree = stmt_tree.children[1]
            if isinstance(body_tree, Tree):
                if body_tree.data == "block":
                    result = transform_statement(body_tree)
                    if isinstance(result, list):
                        body_stmts.extend(result)
                    elif result is not None:
                        body_stmts.append(result)
                else:
                    s = transform_statement(body_tree)
                    if s is not None:
                        if isinstance(s, list):
                            body_stmts.extend(s)
                        else:
                            body_stmts.append(s)

        return If(condition, body_stmts)

    # declaration;
    if data == "decl" or data == "const_decl" or data == "constexpr_decl":
        decl = transform_declaration(stmt_tree)
        return decl

    # assignment;
    if data == "assignment":
        assign_op = ""
        lvalue = None
        rvalue = None

        for child in stmt_tree.children:
            if isinstance(child, Tree):
                if child.data == "lhs":
                    lvalue = _transform_lvalue(child)
                elif child.data == "expression":
                    rvalue = transform_expression(child)
                elif child.data == "assign_operator":
                    # Extract the actual ASSIGN_OP token from within assign_operator
                    for sub in child.children:
                        if _is_token(sub):
                            assign_op = sub.value
            elif _is_token(child):
                assign_op = child.value

        return Assignment(lvalue, assign_op, rvalue)

    # SharedDecl from workgroup "shared" alternative (handled in _transform_workgroup_properties)
    # block statement within body - collect all statements
    if data == "block":
        all_stmts = []
        for stmt_child in stmt_tree.children:
            result = transform_statement(stmt_child)
            if result is not None:
                if isinstance(result, list):
                    all_stmts.extend(result)
                else:
                    all_stmts.append(result)
        return all_stmts

    return None


# ── public transform entry point ───────────────────────────────────


def transform(t: Tree) -> Program:
    """Top-level: turn a parsed Lark tree into a Program AST."""
    p = Program()

    # children: [header, parfor_space, params_list, body_list]
    header_str, wg_trees = extract_header(t.children[0])
    p.header = header_str

    for wg_tree in wg_trees:
        p.workgroups.append(_transform_workgroup_properties(wg_tree))

    p.loop_vars, p.space_dim = extract_loop_vars_and_dim(t.children[1])

    limit_result = extract_limit_expr(t.children[1])
    if isinstance(limit_result, tuple):
        p.lower_bound_expr = limit_result[0]
        p.upper_bound_expr = limit_result[1]
        if len(limit_result) >= 4:
            p.triangular_kind = limit_result[3]

    if not hasattr(p, "lower_bound_expr"):
        if p.space_dim >= 2:
            for c in t.children[1].children:
                if isinstance(c, Tree) and c.data == "expression":
                    p.grid_name = _extract_grid_name_from_expr(c)
                    break

    if not hasattr(p, "lower_bound_expr"):
        expr_tree = _find_expression_tree(t.children[1])
        if expr_tree is not None:
            p.dispatch_size_expr = transform_expression(expr_tree)

    # Build name-to-declaration map from parameters_list
    param_names = []
    for c in t.children[1].children:
        if isinstance(c, Tree) and c.data == "parfor_parameters_list":
            for pc in c.children:
                if isinstance(pc, Tree) and pc.data == "parameter":
                    for tc in pc.children:
                        if isinstance(tc, Token) and tc.type == "IDENT":
                            param_names.append(tc.value)

    # Build name-to-type map from the type declarations
    extended_type_prefixes = (
        "fixed_size_vector",
        "flexible_rows_matrix",
        "fixed_size_matrix",
        "fixed_size_triangular_matrix",
        "fixed_size_levels_rows_cols_matrix",
        "flexible_rows_cols_levels_matrix",
    )

    param_type_map = {}
    # Handle new grammar: parfor_parameters now contains a decl_list wrapper
    decls_node = t.children[2].children[0]
    if isinstance(decls_node, Tree) and decls_node.data == "decl_list":
        decl_trees = decls_node.children
    else:
        decl_trees = [decls_node]
    
    for decl_tree in decl_trees:
        var_name = ""
        if isinstance(decl_tree, Tree):
            for dc in decl_tree.children:
                if isinstance(dc, Token) and dc.type == "IDENT":
                    idx = decl_tree.children.index(dc)
                    if idx > 0:
                        prev = decl_tree.children[idx - 1]
                        if isinstance(prev, Tree):
                            prev_data = prev.data
                            if prev_data in (
                                "type",
                                "int",
                                "float",
                            ) or prev_data.startswith(extended_type_prefixes):
                                var_name = dc.value
        param_type_map[var_name] = transform_declaration(decl_tree)

    # Collect constexpr parameter declarations for type resolution
    param_constexpr_defines = []
    if hasattr(decls_node, 'children'):
        for decl_tree in decls_node.children:
            if isinstance(decl_tree, Tree) and decl_tree.data == "constexpr_decl":
                init_expr = None
                name = ""
                for child in decl_tree.children:
                    if isinstance(child, Token) and child.type == "IDENT":
                        name = child.value
                    elif isinstance(child, Tree) and child.data == "expression":
                        init_expr = transform_expression(child)
                if init_expr is not None:
                    # Convert AST expression to string representation
                    expr_str = _ast_to_str(init_expr)
                    param_constexpr_defines.append((name, expr_str))
    p._param_constexpr_defines = param_constexpr_defines

    # Store triangular bound expressions and names (for RLLM-style push constants)
    if isinstance(limit_result, tuple) and len(limit_result) >= 3:
        lower, upper, raw_bounds = limit_result[:3]
        p.lower_bound_expr = lower
        p.upper_bound_expr = upper
        p.triangular_bounds_raw = raw_bounds
        if len(limit_result) >= 4:
            p.triangular_kind = limit_result[3]

    for bound_name in getattr(p, "triangular_bounds_raw", []) or []:
        if bound_name in param_type_map and bound_name not in param_names:
            param_names.append(bound_name)

    # Add any parameter names declared in the PARAMETERS block that were not
    # listed in the OFFLOAD_PARFOR_*_PARAM line (local variable aliases).
    # Only add declarations whose types are matrices/vectors (not scalars like
    # limit expressions or other push-constant-only values).
    matrix_type_classes = frozenset((
        "FixedSizeVector",
        "FlexibleRowsMatrix",
        "FixedSizeMatrix",
        "FixedSizeTriangularMatrix",
        "FlexibleSizeMatrix",
        "FixedSizeLevelsRowsColsMatrix",
        "FlexibleRowsColsLevelsMatrix",
        "FixedSizeObjVectorMatrix",
    ))
    for decl_name, decl in param_type_map.items():
        if decl_name not in param_names and decl_name.strip():
            vt = getattr(decl, "var_type", None)
            if vt is not None and str(type(vt).__name__) in matrix_type_classes:
                param_names.append(decl_name)

    p.params = [param_type_map.get(n) for n in param_names]

    for stmt_tree in t.children[3].children:
        result = transform_statement(stmt_tree)
        if result is not None:
            if isinstance(result, list):
                p.body_stmts.extend(result)
            else:
                p.body_stmts.append(result)

    return p
