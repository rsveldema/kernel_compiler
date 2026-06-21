"""Vulkan-compatible code generator (GLSL compute shader output).

Produces RLLM-style GLSL with:
- std430 SSBOs using compile-time sized arrays for matrix types with known dimensions
- push_constant block for all scalar params and triangular bounds
- Proper variable initialization from gl_GlobalInvocationID and rllm_push
"""


from codegen.visitors.pretty_printer import prettyprint

from .visitor import Visitor

from ..kast.program import *
from ..kast.type import *
from ..kast.expression import *
from ..kast.statement import *
from ..kast.workgroup import WorkgroupProperties
import math
import re


def _substitute_in_expr(expr, replacements):
    """Recursively replace identifier names in an expression tree."""
    if expr is None:
        return None

    # Recurse into children first
    for attr in ("left", "right"):
        child = getattr(expr, attr, None)
        if hasattr(child, "accept") and child is not None:
            setattr(expr, attr, _substitute_in_expr(child, replacements))
    
    for attr in ("lhs", "rhs"):
        child = getattr(expr, attr, None)
        if hasattr(child, "accept") and child is not None:
            setattr(expr, attr, _substitute_in_expr(child, replacements))

    base = getattr(expr, "base", None)
    if hasattr(base, "accept") and base is not None:
        expr.base = _substitute_in_expr(base, replacements)

    callee = getattr(expr, "callee", None)
    if hasattr(callee, "accept") and callee is not None:
        expr.callee = _substitute_in_expr(callee, replacements)

    args = getattr(expr, "args", [])
    new_args = []
    for arg in (args or []):
        if hasattr(arg, "accept"):
            new_args.append(_substitute_in_expr(arg, replacements))
        else:
            new_args.append(arg)
    expr.args = new_args

    condition = getattr(expr, "condition", None)
    if hasattr(condition, "accept"):
        setattr(expr, "condition", _substitute_in_expr(condition, replacements))
    
    true_expr = getattr(expr, "true_expr", None)
    if hasattr(true_expr, "accept"):
        expr.true_expr = _substitute_in_expr(true_expr, replacements)

    false_expr = getattr(expr, "false_expr", None)
    if hasattr(false_expr, "accept"):
        expr.false_expr = _substitute_in_expr(false_expr, replacements)

    operand = getattr(expr, "operand", None)
    if hasattr(operand, "accept") and operand is not None:
        setattr(expr, "operand", _substitute_in_expr(operand, replacements))

    return expr


_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")
_GLSL_RESERVED = {"None", "static_cast"}


def _is_push_identifier(value: str | None) -> bool:
    return bool(value and _IDENT_RE.match(value) and value not in _GLSL_RESERVED)


def parse_literal(s: str) -> Expression:
    """Convert a literal string to an AST Number expression."""
    s = s.strip()
    if s.isdigit():
        stripped = s.lstrip("0") or "0"
        return Number(stripped)
    return Number(s)


def _extract_size_from_expr(expr: Expression, constants: dict[str, str] | None = None) -> int:
    assert expr is not None
    if constants is None:
        constants = {}

    if isinstance(expr, Number):
        return int(expr.value)

    if isinstance(expr, Identifier):
        val_str = constants.get(expr.name)
        if val_str is not None:
            return _extract_size_from_expr(parse_literal(val_str), constants)
        return -1

    if isinstance(expr, BinaryExpr):
        lv = _extract_size_from_expr(expr.left, constants)
        rv = _extract_size_from_expr(expr.right, constants)
        if lv < 0 or rv < 0:
            return -1
        match expr.op:
            case "+":
                return lv + rv
            case "-":
                return lv - rv
            case "*":
                return lv * rv
            case "/":
                return lv // rv
            case _:
                assert False, f"no binary operator for constant folding: {expr.op}"

    if isinstance(expr, LimitExpr):
        return _extract_size_from_expr(expr.max_val, constants)

    assert False, f"cannot constant fold: {prettyprint(expr)}"
    # s = self._to_str(expr)
    # if s and re.match(r'^\d+$', s):
    #    return s
    # return None


def _compute_matrix_size(ty: Type, constants: dict[str, str] | None = None) -> int:
    parts = 1
    if isinstance(ty, (FixedSizeLevelsRowsColsMatrix, FlexibleRowsColsLevelsMatrix, FixedSizeObjVectorMatrix)):
        for attr in ("level_expr", "row_size_expr", "col_size_expr"):
            expr = getattr(ty, attr, None)
            if expr:
                s = _extract_size_from_expr(expr, constants)
                if s > 0:
                    parts *= s
        return parts
    elif isinstance(ty, FixedSizeTriangularMatrix):
        row = _extract_size_from_expr(ty.row_size_expr, constants)
        col = _extract_size_from_expr(ty.col_size_expr, constants)
        if row <= 0 or col <= 0:
            return -1
        if row != col:
            raise ValueError("fixed_size_triangular_matrix must be square")
        return row * (row + 1) // 2
    elif isinstance(ty, (FixedSizeMatrix, FlexibleSizeMatrix, FlexibleRowsColsMatrix)):
        for attr in ("row_size_expr", "col_size_expr"):
            expr = getattr(ty, attr, None)
            if expr:
                s = _extract_size_from_expr(expr, constants)
                if s > 0:
                    parts *= s
        return parts
    elif isinstance(ty, FixedSizeVector):
        if ty.size_expr:
            s = _extract_size_from_expr(ty.size_expr, constants)
            if s > 0:
                parts *= s
                return parts
        return -1
    return -1


class VulkanKernelVisitor(Visitor):
    """Transforms the parsed AST into a Vulkan GLSL compute shader string."""

    def __init__(self, use_bfloat16: bool = False):
        self._use_bfloat16 = use_bfloat16
        self._lines: list[str] = []
        self._indent_level: int = 0
        self._binding_counter: int = 0
        self._ssbo_map: dict[str, tuple] = {}  # param_name -> (is_3d, type_node)
        self._push_constant_fields: list[tuple] = []  # (name, type_str) pairs
        self._push_constant_map: dict[str, bool] = {}
        self._triangular_upper_bound_name: str | None = None
        self._uses_reduction_chunks: bool = False
        self._constexpr_defines: list[tuple[str, str]] = []
        self._constexpr_map: dict[str, str] = {}

    def _type_uses_float16(self, ty: Type | None) -> bool:
        if ty is None:
            return False
        if isinstance(ty, Float16):
            return True
        elem_type = getattr(ty, "elem_type", None)
        return self._type_uses_float16(elem_type)

    def _program_uses_float16(self, node: Program) -> bool:
        for param in node.params:
            if isinstance(param, Declaration) and self._type_uses_float16(param.var_type):
                return True
        return False

    def _array_root_identifier(self, node: Expression | None) -> str | None:
        while isinstance(node, ArrayAccess):
            node = node.base
        if isinstance(node, Identifier):
            return node.name
        return None

    def _lvalue_uses_float16_storage(self, node: Expression | None) -> bool:
        root_name = self._array_root_identifier(node)
        if root_name is None:
            return False
        mapped = self._ssbo_map.get(root_name)
        return mapped is not None and self._type_uses_float16(mapped[1])

    def _to_arithmetic_operand(self, node: Expression | None) -> str:
        value = self._visit_expr_child(node)
        if isinstance(node, ArrayAccess) and self._lvalue_uses_float16_storage(node):
            return f"float({value})"
        return value


    def _resolve_constexpr_value(self, expr_str: str) -> str | None:
        """Try to resolve an expression string to a plain literal using existing constexpr_map values.
        
        Returns the resolved value as a string (e.g., "36") or None if it can't be resolved.
        Handles simple numeric literals and binary expressions with operator splitting.
        """
        expr_str = expr_str.strip()
        # Already a plain literal?
        if re.match(r'^\d+$', expr_str):
            return expr_str
        
        # Strip outer parentheses for evaluation
        stripped = expr_str
        while len(stripped) > 1 and stripped[0] == '(' and stripped[-1] == ')':
            stripped = stripped[1:-1].strip()
        
        if stripped == expr_str:
            return None
        
        # Try splitting on binary operators (left-to-right, outermost level)
        depth = 0
        for op in ['+', '-', '*', '/']:
            i = len(stripped) - 2  # Don't match last char in case of trailing paren
            while i > 0:
                if stripped[i] == ')':
                    depth += 1
                elif stripped[i] == '(':
                    depth -= 1
                elif depth == 0 and stripped[i] == op and i + 1 < len(stripped):
                    # Check it's a standalone operator (not part of >, <, etc.)
                    before_ok = stripped[i-1].isspace() if i > 0 else True
                    after_ok = stripped[i+1].isspace() if i + 1 < len(stripped) else False
                    if before_ok and after_ok:
                        left_str = stripped[:i].strip()
                        right_str = stripped[i+1:].strip()
                        lv = self._resolve_constexpr_value(left_str)
                        rv = self._resolve_constexpr_value(right_str)
                        if lv is not None and rv is not None:
                            try:
                                if op == '+':
                                    return str(int(lv) + int(rv))
                                elif op == '-':
                                    return str(int(lv) - int(rv))
                                elif op == '*':
                                    return str(int(lv) * int(rv))
                                elif op == '/':
                                    result = int(lv) // int(rv) if int(rv) != 0 else None
                                    return str(result) if result is not None else None
                            except (ValueError, OverflowError):
                                pass
                        break
                i -= 1
        
        # Check for size_t(...) cast - unwrap to inner expression
        if stripped.startswith('size_t(') and stripped.endswith(')'):
            inner = stripped[7:-1].strip()
            return self._resolve_constexpr_value(inner)
        
        return None

    def _is_shared_memory_multi_arg(self, node: Program) -> bool:
        if not node.use_shared_memory_tiling:
            return False
        names = [p.name for p in node.params if isinstance(p, Declaration)]
        return names == ["A1", "B1", "A2", "B2", "A3", "B3", "C"]

    # ── helpers ────────────────────────────────────────────────────────

    def _indent(self) -> str:
        return "    " * self._indent_level

    def _emit(self, line: str = "") -> None:
        self._lines.append(f"{self._indent()}{line}")

    def _push(self) -> None:
        self._indent_level += 1

    def _pop(self) -> None:
        self._indent_level -= 1

    def result(self) -> str:
        # Emit constexpr defines as preprocessor directives at the top of the shader.
        defines = []
        for name, expr in self._constexpr_defines:
            defines.append(f"#define {name} {expr}")
        if defines:
            return "\n".join(defines) + "\n" + "\n".join(self._lines) + "\n"
        return "\n".join(self._lines) + "\n"

    # ── type helpers (GLSL) ────────────────────────────────────────────

    def _glsl_elem_type(self, ty) -> str:
        """Map an AST Type node to a GLSL element type string."""
        if hasattr(ty, "elem_type") and ty.elem_type is not None:
            return self._glsl_type_name(ty.elem_type)
        return self._glsl_type_name(ty) if ty else "float"

    def _glsl_type_name(self, ty) -> str:
        """Map an AST Type node to a GLSL type name."""
        if isinstance(ty, Int):
            # size_t → uint64_t, other integers → int
            return "uint64_t" if ty.name == "size_t" else "int"
        if isinstance(ty, Float):
            return "float"
        if isinstance(ty, Float16):
            return "bfloat_t" if self._use_bfloat16 else "float16_t"
        # Compound types (vectors/matrices) default to float for element type
        return "float"

    def _to_str(self, node) -> str:
        """Dispatcher: convert an AST node to a GLSL string via visitor dispatch."""
        if node is None:
            return ""
        return node.accept(self)

    def _format_constant(self, value, value_type: str) -> str:
        if value_type == "int":
            return str(int(value))
        formatted = f"{float(value):.9g}"
        if "." not in formatted and "e" not in formatted.lower():
            formatted += ".0"
        return formatted

    def _callee_name(self, node: Expression) -> str:
        if isinstance(node, Identifier):
            return node.name or ""
        if isinstance(node, FieldAccess):
            base = self._callee_name(node.base)
            return f"{base}.{node.field}" if base else node.field
        return self._to_str(node)

    def _const_eval_expr(self, node: Expression):
        if node is None:
            return None

        if isinstance(node, Number):
            if isinstance(node.value, float):
                return (float(node.value), "float")
            return (int(node.value), "int")

        if isinstance(node, Identifier):
            if node.name in self._constexpr_map:
                value = self._constexpr_map[node.name]
                try:
                    if "." in value or "e" in value.lower():
                        return (float(value), "float")
                    return (int(value), "int")
                except ValueError:
                    return None
            return None

        if isinstance(node, UnaryMinusExpr):
            inner = self._const_eval_expr(node.operand)
            if inner is None:
                return None
            value, value_type = inner
            return (-value, value_type)

        if isinstance(node, CastExpr):
            inner = self._const_eval_expr(node.operand)
            if inner is None:
                return None
            value, _value_type = inner
            cast_type = node.cast_type.accept(self)
            if cast_type == "float":
                return (float(value), "float")
            return (int(value), "int")

        if isinstance(node, BinaryExpr):
            left = self._const_eval_expr(node.left)
            right = self._const_eval_expr(node.right)
            if left is None or right is None:
                return None
            lv, lt = left
            rv, rt = right
            value_type = "float" if lt == "float" or rt == "float" else "int"
            try:
                if node.op == "+":
                    value = lv + rv
                elif node.op == "-":
                    value = lv - rv
                elif node.op == "*":
                    value = lv * rv
                elif node.op == "/":
                    if rv == 0:
                        return None
                    value = (int(lv) // int(rv)) if value_type == "int" else (float(lv) / float(rv))
                else:
                    return None
            except (ValueError, OverflowError, ZeroDivisionError):
                return None
            return (value, value_type)

        if isinstance(node, CallExpr):
            callee = self._callee_name(node.callee)
            if callee in {"sqrt", "std.sqrt", "std::sqrt"} and len(node.args) == 1:
                arg = self._const_eval_expr(node.args[0])
                if arg is None:
                    return None
                value, _value_type = arg
                if value < 0:
                    return None
                return (math.sqrt(float(value)), "float")
            return None

        return None

    # ── expression visitors ────────────────────────────────────────────

    def visit_type(self, node: Type) -> str:
        return self._glsl_elem_type(node)

    def visit_int(self, node: Int) -> str:
        if node.name == "size_t":
            return "uint64_t"
        return "int"

    def visit_float(self, node: Float) -> str:
        return "float"

    def visit_float16(self, node: Float16) -> str:
        if self._use_bfloat16:
            return "bfloat_t"
        return "float16_t"

    def visit_coop_mat(self, node: CoopMat) -> str:
        elem = self._to_str(node.elem_type)
        scope = self._to_str(node.scope_expr)
        rows = self._to_str(node.row_size_expr)
        cols = self._to_str(node.col_size_expr)
        use = self._to_str(node.use_expr)
        return f"coopmat<{elem}, {scope}, {rows}, {cols}, {use}>"

    def visit_fixed_size_vector(self, node: FixedSizeVector) -> str:
        return self._glsl_elem_type(node)

    def visit_flexible_rows_matrix(self, node: FlexibleRowsMatrix) -> str:
        return self._glsl_elem_type(node)

    def visit_fixed_size_matrix(self, node: FixedSizeMatrix) -> str:
        return self._glsl_elem_type(node)

    def visit_fixed_size_triangular_matrix(self, node: FixedSizeTriangularMatrix) -> str:
        return self._glsl_elem_type(node)

    def visit_flexible_size_matrix(self, node: FlexibleSizeMatrix) -> str:
        return self._glsl_elem_type(node)

    def visit_fixed_size_obj_vector_matrix(self, node: FixedSizeObjVectorMatrix) -> str:
        return self._glsl_elem_type(node)

    def visit_fixed_size_levels_rows_cols_matrix(
        self, node: FixedSizeLevelsRowsColsMatrix
    ) -> str:
        return self._glsl_elem_type(node)

    def visit_flexible_rows_cols_levels_matrix(
        self, node: FlexibleRowsColsLevelsMatrix
    ) -> str:
        return self._glsl_elem_type(node)

    def visit_expression(self, node: Expression) -> str:
        return node.accept(self)

    def visit_flexible_rows_cols_matrix(
        self, node: FlexibleRowsColsMatrix
    ) -> str:
        return self._glsl_elem_type(node)

    def visit_ternary_expr(self, node: TernaryExpr) -> str:
        cond = self._to_str(node.condition)
        true_val = self._to_str(node.true_expr)
        false_val = self._to_str(node.false_expr)
        return f"({cond} ? {true_val} : {false_val})"

    def visit_unary_minus_expr(self, node: UnaryMinusExpr) -> str:
        folded = self._const_eval_expr(node)
        if folded is not None:
            return self._format_constant(*folded)
        operand = self._to_str(node.operand)
        return f"-{operand}"

    def visit_expression(self, node: Expression) -> str:
        return node.accept(self)

    def visit_number(self, node: Number) -> str:
        val = node.value
        if isinstance(val, float):
            s = f"{val}"
            if "." not in s and "e" not in s.lower():
                s += ".0"
            return s
        s = str(int(val))
        if getattr(node, "unsigned", False):
            if abs(val) > 4294967295:  # UINT_MAX — need uint64
                return s + "UL"
            return s + "U"
        return s

    def visit_identifier(self, node: Identifier) -> str:
        name = node.name
        if name in self._push_constant_map:
            return f"rllm_push.{name}"
        # Check constexpr map for constant-folded values
        if name in self._constexpr_map:
            return self._constexpr_map[name]
        return name or "unknown"


    def _visit_expr_child(self, node):
        """Helper for visiting expression children (returns '?' for None)."""
        if node is None:
            return "?"
        return node.accept(self)

    def visit_array_access(self, node: ArrayAccess) -> str:
        if isinstance(node.base, ArrayAccess) and isinstance(node.base.base, Identifier):
            base_name = node.base.base.name
            mapped = self._ssbo_map.get(base_name)
            if mapped is not None and isinstance(mapped[1], FixedSizeObjVectorMatrix):
                level = self._visit_expr_child(node.base.indices[0]) if node.base.indices else "0"
                row = self._visit_expr_child(node.indices[0]) if len(node.indices) > 0 else "0"
                col = self._visit_expr_child(node.indices[1]) if len(node.indices) > 1 else "0"
                return f"rllm_load_{base_name}({level}, {row}, {col})"
        base = node.base.accept(self) if node.base else "?"
        parts = []
        for idx in node.indices:
            idx_str = self._visit_expr_child(idx) if idx else "?"
            parts.append(idx_str)
        return f"{base}[{', '.join(parts)}]"

    def visit_field_access(self, node: FieldAccess) -> str:
        base = ""
        if isinstance(node.base, Identifier):
            base = node.base.name or "unknown"
        else:
            base = self._visit_expr_child(node.base)
        assert base != None
        base += "." + node.field
        return base

    def visit_call_expr(self, node: CallExpr) -> str:
        folded = self._const_eval_expr(node)
        if folded is not None:
            return self._format_constant(*folded)
        callee = self._visit_expr_child(node.callee)
        args = ", ".join(self._visit_expr_child(arg) for arg in node.args)
        return f"{callee}({args})"

    def visit_limit_expr(self, node: LimitExpr) -> str:
        max_val = self._visit_expr_child(node.max_val)
        start = self._visit_expr_child(getattr(node, "start", None))
        end = self._visit_expr_child(getattr(node, "end", None))
        if start and start != "0":
            return f"limit<{max_val}>({start}, {end})"
        return f"limit<{max_val}>({end})"

    def visit_binary_expr(self, node: BinaryExpr) -> str:
        folded = self._const_eval_expr(node)
        if folded is not None:
            return self._format_constant(*folded)
        left = self._to_arithmetic_operand(node.left)
        right = self._to_arithmetic_operand(node.right)
        op = node.op or "+"
        return f"({left} {op} {right})"

    def visit_cast_expr(self, node: CastExpr) -> str:
        folded = self._const_eval_expr(node)
        if folded is not None:
            return self._format_constant(*folded)
        operand = self._visit_expr_child(node.operand)
        cast_type = node.cast_type.accept(self)
        return f"{cast_type}({operand})"

    def visit_negation_expr(self, node: NegationExpr) -> str:
        operand = self._visit_expr_child(node.operand)
        return f"!{operand}"

    def visit_wildcard_expression(self, node: WildcardExpression) -> str:
        return f"wildcard({node.name})"

    # ── condition visitor ──────────────────────────────────────────────

    def visit_condition(self, node: Condition) -> str:
        lhs = self._visit_expr_child(node.lhs)
        rhs = self._visit_expr_child(node.rhs)
        return f"{lhs} {node.op} {rhs}"

    # ── statement visitors (return string, caller emits with indent) ──

    def visit_statement(self, node: Statement) -> str:
        return node.accept(self)

    def visit_for_loop_range(self, node: ForLoopRange) -> str:
        """Generate a GLSL for loop from range-style (for i in range(n))."""
        lines = []
        ind = "    " * (self._indent_level + 1)

        lower_bound = "0"
        upper_bound = "?"
        if node.init_expr and isinstance(node.init_expr, LimitExpr):
            max_val = self._to_str(node.init_expr.max_val) if node.init_expr.max_val else "?"
            start_val = self._to_str(getattr(node.init_expr, "start", None)) if getattr(node.init_expr, "start", None) else "0"
            end_val = self._to_str(getattr(node.init_expr, "end", None)) if getattr(node.init_expr, "end", None) else "?"
            lower_bound = start_val
            upper_bound = end_val if end_val != "?" else max_val
            if self._triangular_upper_bound_name and upper_bound.lstrip("-").isdigit():
                upper_bound = f"rllm_push.{self._triangular_upper_bound_name}"

        lines.append(
            f"{ind}for (int {node.loop_var_name} = {lower_bound}; "
            f"{node.loop_var_name} < {upper_bound}; ++{node.loop_var_name}) {{"
        )

        for stmt in node.body_stmts:
            if hasattr(stmt, "accept"):
                result = stmt.accept(self)
                if isinstance(result, str):
                    lines.append(f"{ind}{result}")

        lines.append(f"{ind}}}")
        return "\n".join(lines) + "\n"

    def visit_for_loop_with_condition_and_increment(
        self, node: ForLoopWithConditionAndIncrement
    ) -> str:
        """Generate a GLSL for loop from condition+increment style."""
        lines = []
        ind = "    " * (self._indent_level + 1)

        lower_bound = "0"
        upper_bound = "?"

        if node.condition:
            op = node.condition.op or ">="
            rhs = self._to_str(node.condition.rhs) if node.condition.rhs else "?"
            lower_bound = (
                self._to_str(node.init_expr) if node.init_expr else "0"
            )
            upper_bound = rhs

        inc_var = node.increment_var if node.increment_var else node.loop_var_name
        inc_op = node.increment_op if node.increment_op else "++"

        loop_type = self._glsl_type_name(node.loop_var_type) if node.loop_var_type else "int"

        current_program = getattr(self, "_current_program", None)
       

        lines.append(
            f"{ind}for ({loop_type} {node.loop_var_name} = {lower_bound}; "
            f"{node.loop_var_name} < {upper_bound}; {inc_op}{inc_var}) {{"
        )

        for stmt in node.body_stmts:
            if hasattr(stmt, "accept"):
                result = stmt.accept(self)
                if isinstance(result, str):
                    lines.append(f"{ind}{result}")

        lines.append(f"{ind}}}")
        return "\n".join(lines) + "\n"

    def visit_if(self, node: If) -> str:
        cond_str = self._to_str(node.condition) if node.condition else "?"
        ind = "    " * (self._indent_level + 1)
        lines = [f"{ind}if ({cond_str}) {{"]

        for stmt in node.body_stmts:
            if hasattr(stmt, "accept"):
                result = stmt.accept(self)
                if isinstance(result, str):
                    lines.append(f"{ind}{result}")

        lines.append(f"{ind}}}")
        
        if node.else_stmts:
            else_lines = [f"{ind}else {{"]
            for stmt in node.else_stmts:
                if hasattr(stmt, "accept"):
                    result = stmt.accept(self)
                    if isinstance(result, str):
                        else_lines.append(f"{ind}{result}")
            else_lines.append(f"{ind}}}")
            lines.extend(else_lines)
        
        return "\n".join(lines) + "\n"

    def visit_declaration(self, node: Declaration) -> str:
        # constexpr declarations are emitted as preprocessor defines, not GLSL variables.
        if node.is_constexpr and node.init_expr is not None:
            expr_str = self._to_str(node.init_expr)
            self._constexpr_defines.append((node.name, expr_str))
            return ""
        prefix = "const " if node.is_const else ""
        var_type = self._to_str(node.var_type) if node.var_type else "float"
        init_str = ""
        if node.init_expr is not None:
            init_str = f" = {self._to_str(node.init_expr)}"
        return f"{prefix}{var_type} {node.name}{init_str};"

    def visit_assignment(self, node: Assignment) -> str:
        if (
            isinstance(node.lvalue, ArrayAccess)
            and isinstance(node.lvalue.base, ArrayAccess)
            and isinstance(node.lvalue.base.base, Identifier)
        ):
            base_name = node.lvalue.base.base.name
            mapped = self._ssbo_map.get(base_name)
            if mapped is not None and isinstance(mapped[1], FixedSizeObjVectorMatrix):
                level = self._visit_expr_child(node.lvalue.base.indices[0]) if node.lvalue.base.indices else "0"
                row = self._visit_expr_child(node.lvalue.indices[0]) if len(node.lvalue.indices) > 0 else "0"
                col = self._visit_expr_child(node.lvalue.indices[1]) if len(node.lvalue.indices) > 1 else "0"
                rvalue = self._to_str(node.rvalue) if node.rvalue else "?"
                if node.assign_op != "=":
                    op = node.assign_op[0]
                    rvalue = f"rllm_load_{base_name}({level}, {row}, {col}) {op} ({rvalue})"
                return f"rllm_store_{base_name}({level}, {row}, {col}, {rvalue});"
        lvalue = self._to_str(node.lvalue) if node.lvalue else "?"
        rvalue = self._to_str(node.rvalue) if node.rvalue else "?"
        if (
            self._uses_reduction_chunks
            and node.assign_op == "+="
            and isinstance(node.lvalue, ArrayAccess)
        ):
            return f"atomicAdd({lvalue}, {rvalue});"
        if self._lvalue_uses_float16_storage(node.lvalue):
            if node.assign_op == "=":
                return f"{lvalue} = float16_t({rvalue});"
            op = node.assign_op[0]
            return f"{lvalue} = float16_t({lvalue} {op} ({rvalue}));"
        return f"{lvalue} {node.assign_op} {rvalue};"

    def visit_overflow_check(self, node: OverflowCheck) -> str:
        lvalue = self._to_str(node.lvalue) if node.lvalue else "?"
        operand = self._to_str(node.operand) if node.operand else "?"
        return f"// OVERFLOW_CHECK_ADD({lvalue}, {operand});"

    def visit_shared_decl(self, node: SharedDecl) -> str:
        # constexpr shared declarations are emitted as preprocessor defines.
        if node.is_constexpr and node.init_expr is not None:
            expr_str = self._to_str(node.init_expr)
            self._constexpr_defines.append((node.name, expr_str))
            return ""
        prefix = "const " if node.is_const else ""
        var_type = self._to_str(node.var_type) if node.var_type else "float"
        init_str = ""
        if node.init_expr is not None:
            init_str = f" = {self._to_str(node.init_expr)}"
        dim_strs = [self._to_str(d) for d in getattr(node, "dimensions", []) or []]
        dims = "".join(f"[{d}]" for d in dim_strs)
        return f"shared {prefix}{var_type} {node.name}{dims}{init_str};"

    def visit_tensor_layout_decl(self, node) -> str:
        """Emit a tensor_layout declaration (tensorLayout<N> variable)."""
        prefix = "const " if getattr(node, "is_const", False) else ""
        dim_str = self._to_str(getattr(node, "dim_expr", None))
        name = getattr(node, "name", "?")
        init_str = ""
        if getattr(node, "init_expr", None) is not None:
            init_str = f" = {self._to_str(node.init_expr)}"
        return f"{prefix}tensorLayoutNV<{dim_str}> {name}{init_str};"

    def visit_raw_statement(self, node: RawStatement) -> str:
        return node.text.rstrip()

    def visit_call_statement(self, node: CallStatement) -> str:
        return f"{self._to_str(node.call_expr)};"

    def visit_wildcard_statement(self, node: WildcardStatement) -> str:
        return f"wildcard({node.name})"

    def visit_workgroup_properties(self, node: WorkgroupProperties) -> str:
        parts = []
        if node.x_expr is not None:
            parts.append(f"x: {self._to_str(node.x_expr)}")
        if node.y_expr is not None:
            parts.append(f"y: {self._to_str(node.y_expr)}")
        if node.z_expr is not None:
            parts.append(f"z: {self._to_str(node.z_expr)}")
        return f"workgroup {{ {', '.join(parts)} }}"

    def visit_return_statement(self, node: ReturnStatement) -> str:
        ind = "    " * self._indent_level
        return f"{ind}return;"

    def visit_atomic_op(self, node: AtomicOp) -> str:
        lhs = self._to_str(node.lhs)
        rhs = self._to_str(node.rhs)
        ind = "    " * self._indent_level
        return f"{ind}{node.op}({lhs}, {rhs});"

    def _workgroup_size(self, node: Program) -> tuple[str, str, str]:
        # Default workgroup sizes: 16x16x1 for 2D/3D, 16x1x1 for 1D
        if node.space_dim >= 2:
            wg_x, wg_y, wg_z = "16", "16", "1"
        else:
            wg_x, wg_y, wg_z = "16", "1", "1"
        for wg in node.workgroups:
            if isinstance(wg, WorkgroupProperties):
                wg_x = self._to_str(wg.x_expr) if wg.x_expr else wg_x
                wg_y = self._to_str(wg.y_expr) if wg.y_expr else wg_y
                wg_z = self._to_str(wg.z_expr) if wg.z_expr else wg_z
        return wg_x, wg_y, wg_z

        self._emit("}")


    # ── Program visitor (main entry point) ────────────────────────────


    def _emit_tile_vars(self, node):
        """Map each loop variable index to its name (used for gl_GlobalInvocationID init).
        
        When tiled, tile indices replace original loop variables as the outer dimensions.
        Each workgroup handles one tile so loop_var names map directly to 
        gl_GlobalInvocationID.x/y/z coordinates.
        """
        self._tile_var_name = {}
        for idx, var_name in enumerate(node.loop_vars):
            self._tile_var_name[idx] = var_name  # use original name (i, j)

    def _flatten_tiled_body(self, stmts):
        """Recursively flatten tile loops out of body statements.
        
        When tiling is applied, tile_i/tile_j loops should not appear in the GLSL body.
        Instead their content is flattened into the main() function body, and the 
        tile indices come from gl_GlobalInvocationID.x/y.
        """
        result = []
        for stmt in stmts:
            #assert False, "shoult not be needed. a tkernel should have flattened the loop"
            if isinstance(stmt, ForLoopWithConditionAndIncrement) and                getattr(stmt, "loop_var_name", "").startswith("tile_"):
                # Skip this tile loop, flatten its body instead
                inner_stmts = getattr(stmt, "body_stmts", [])
                if inner_stmts:
                    result.extend(self._flatten_tiled_body(inner_stmts))
            else:
                result.append(stmt)
        return result


    def _find_loop_var_metadata(self, var_name):
        """Find parallelization metadata for a loop variable."""
        for stmt in getattr(self, "_parallel_loop_vars", []):
            lv_name = getattr(stmt, "loop_var_name", "")
            if lv_name == var_name:
                offset = getattr(stmt, "_parallel_offset_var", f"{var_name}_offset")
                global_var = f"global_{var_name}"
                bound = str(getattr(
                    getattr(stmt, "_parallel_upper_bound", None), "value", 0
                ))
                return (offset, global_var, bound)
        return None


    def _emit_parallel_stmt(self, stmt, node):
        """Emit a single body statement for a parallelized program.

        Handles variable substitution and early-exit guards for loops.
        """
        if not hasattr(stmt, "accept"):
            return ""

        sub_map = getattr(self, "_parallel_global_vars", {})
        wg_size = getattr(node, 'workgroup_size', 8)
        parallelized = getattr(node, 'parallelized', False)

        # For parallelized programs, loops whose variables are tracked by
        # _parallel_global_vars are NOT emitted as loop constructs. Instead
        # their body statements are processed with variable substitution so that
        # all references to loop variables resolve to the global indices computed
        # from gl_GlobalInvocationID and workgroup partitioning.
        if parallelized:
            #assert False, "tkernel files should have replaced this loop construct"
            return self._emit_parallel_body_stmt(stmt, node)

        # ── Non-parallelized path: emit normally with substitution. ───────

        if isinstance(stmt, ForLoopWithConditionAndIncrement):
            cond = getattr(stmt, "condition", None)
            lv_name = self._get_loop_var_from_cond(stmt, cond)

            if lv_name and lv_name in sub_map:
                global_var = sub_map[lv_name]

                old_bound = None
                if hasattr(cond, "rhs") and isinstance(cond.rhs, Number):
                    old_bound = cond.rhs.value
                    cond.rhs.value = wg_size

                if hasattr(cond, "lhs"):
                    lvalue = getattr(cond, "lhs", None)
                    if isinstance(lvalue, Identifier):
                        lvalue.name = global_var

                stmt.accept(self)  # Emit the modified for-loop with bound change

                if old_bound is not None and hasattr(cond, "rhs"):
                    cond.rhs.value = old_bound

                return ""

            # Non-parallelizable loop variable – just substitute and emit.
            self._substitute_stmt(stmt, sub_map)
            return stmt.accept(self)

        if isinstance(stmt, ForLoopRange):
            lv_name = getattr(stmt, "loop_var_name", "")

            if lv_name and lv_name in sub_map:
                global_var = sub_map[lv_name]

                ie = getattr(stmt, "init_expr", None)
                old_max_val = None
                if hasattr(ie, "max_val") and isinstance(ie.max_val, Number):
                    old_max_val = ie.max_val.value
                    ie.max_val.value = wg_size

                self._substitute_stmt(stmt, sub_map)
                result = stmt.accept(self)

                if old_max_val is not None and hasattr(ie, "max_val") and isinstance(ie.max_val, Number):
                    ie.max_val.value = old_max_val

                return result

            self._substitute_stmt(stmt, sub_map)
            return stmt.accept(self)

        # Non-loop statement: just substitute identifiers in expressions.
        self._substitute_stmt(stmt, sub_map)
        return stmt.accept(self)

    def _emit_parallel_body_stmt(self, stmt, node):
        """Process a body statement for a parallelized program.

        For parallelized programs where iteration is handled by gl_GlobalInvocationID
        and workgroup partitioning, loop constructs in the original AST are not emitted
        as loops. Instead their body statements are processed with variable substitution
        so that all references to loop variables use their global indices.
        """
        if not hasattr(stmt, "accept"):
            return ""

        sub_map = getattr(self, "_parallel_global_vars", {})
        parallelized = getattr(node, 'parallelized', False)
        workgroup_count = getattr(node, 'workgroup_count', 1)

        # If the statement is a ForLoop/ForLoopRange whose variable is in sub_map,
        # process its body statements directly (not as a loop construct).
        if parallelized:
            if isinstance(stmt, ForLoopWithConditionAndIncrement):
                cond = getattr(stmt, "condition", None)
                lv_name = self._get_loop_var_from_cond(stmt, cond)
                if lv_name and lv_name in sub_map:
                    parts = []
                    for inner_stmt in (getattr(stmt, "body_stmts", []) or []):
                        self._substitute_stmt(inner_stmt, sub_map)
                        r = inner_stmt.accept(self)
                        if isinstance(r, str):
                            parts.append(r.rstrip())
                    return "\n".join(parts)
            elif isinstance(stmt, ForLoopRange):
                lv_name = getattr(stmt, "loop_var_name", "")
                if lv_name and lv_name in sub_map:
                    parts = []
                    for inner_stmt in (getattr(stmt, "body_stmts", []) or []):
                        self._substitute_stmt(inner_stmt, sub_map)
                        r = inner_stmt.accept(self)
                        if isinstance(r, str):
                            parts.append(r.rstrip())
                    return "\n".join(parts)

        # Default: substitute identifiers in expressions and emit the statement normally.
        self._substitute_stmt(stmt, sub_map)
        return stmt.accept(self)

    def _get_loop_var_from_cond(self, stmt, cond):
        """Extract loop variable name from a ForLoopWithConditionAndIncrement."""
        lv = getattr(stmt, "loop_var_name", "")
        if lv:
            return lv
        if isinstance(cond, Condition):
            lhs = getattr(cond, "lhs", None)
            if isinstance(lhs, Identifier):
                return lhs.name
        return ""

    def _substitute_stmt(self, stmt, sub_map):
        """Substitute loop variable identifiers in a statement's expression tree."""
        if not hasattr(stmt, "accept"):
            return
        
        # lvalue (for assignments)
        lvalue = getattr(stmt, "lvalue", None)
        if isinstance(lvalue, Expression):
            self._substitute_expr(lvalue, sub_map)

        rvalue = getattr(stmt, "rvalue", None)
        if isinstance(rvalue, Expression):
            self._substitute_expr(rvalue, sub_map)

        lhs = getattr(stmt, "lhs", None)
        if isinstance(lhs, Expression):
            self._substitute_expr(lhs, sub_map)

        rhs = getattr(stmt, "rhs", None)
        if isinstance(rhs, Expression):
            self._substitute_expr(rhs, sub_map)

        init_expr = getattr(stmt, "init_expr", None)
        if isinstance(init_expr, Expression):
            self._substitute_expr(init_expr, sub_map)

    def _substitute_expr(self, expr, sub_map):
        """Substitute loop variable identifiers in an expression tree."""
        if expr is None:
            return
        
        lhs = getattr(expr, "lhs", None)
        if isinstance(lhs, Identifier) and lhs.name in sub_map:
            lhs.name = sub_map[lhs.name]
        
        rhs = getattr(expr, "rhs", None)
        if isinstance(rhs, Identifier) and rhs.name in sub_map:
            rhs.name = sub_map[rhs.name]

        base = getattr(expr, "base", None)
        if isinstance(base, Identifier) and base.name in sub_map:
            base.name = sub_map[base.name]

        lvalue = getattr(expr, "lvalue", None)
        if isinstance(lvalue, Identifier) and lvalue.name in sub_map:
            lvalue.name = sub_map[lvalue.name]

        # Recurse into children
        for child_attr in ("left", "right", "operand"):
            child = getattr(expr, child_attr, None)
            if isinstance(child, Expression):
                self._substitute_expr(child, sub_map)

        # Handle ArrayAccess indices (e.g., dst[i] -> dst[global_i])
        indices = getattr(expr, "indices", [])
        for idx in range(len(indices)):
            ind = indices[idx]
            if isinstance(ind, Identifier):
                self._substitute_expr(ind, sub_map)



    def visit_program(self, node: Program) -> str:
        self._current_program = node
        self._lines = []
        self._indent_level = 0
        self._binding_counter = 0
        self._ssbo_map = {}
        self._push_constant_fields = []
        self._push_constant_map = {}
        self._triangular_upper_bound_name = None
        self._uses_reduction_chunks = node.reduction_chunks > 1

        self._emit("#version 450")
        self._emit("#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require")
        if self._program_uses_float16(node):
            self._emit("#extension GL_EXT_shader_16bit_storage : require")
            self._emit("#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require")
        self._emit("#extension GL_KHR_shader_subgroup_arithmetic : require")
        self._emit("#extension GL_KHR_shader_subgroup_clustered : require")
        self._emit("#extension GL_EXT_shader_atomic_float : require")
        self._emit("#extension GL_EXT_shader_atomic_float2 : require")
        if node.use_cooperative_matrix2:
            self._emit("#extension GL_KHR_memory_scope_semantics : require")
            self._emit("#extension GL_KHR_cooperative_matrix : require")
            self._emit("#extension GL_NV_cooperative_matrix2 : require")
        wg_x, wg_y, wg_z = self._workgroup_size(node)
        self._emit(f"layout(local_size_x = {wg_x}, local_size_y = {wg_y}, local_size_z = {wg_z}) in;")
        self._emit("")

        # ── Classify parameters ──
        all_matrix_params = []
        triangular_bounds_raw = node.triangular_bounds_raw
        triangular_kind = node.triangular_kind
        is_triangular = bool(triangular_kind) or len(triangular_bounds_raw) >= 2
        if is_triangular and _is_push_identifier(triangular_bounds_raw[1]) and not triangular_bounds_raw[1].lstrip("-").isdigit():
            self._triangular_upper_bound_name = triangular_bounds_raw[1]

        for param in node.params:
            if param is None or not isinstance(param, Declaration):
                continue

            vt = param.var_type
            is_matrix = isinstance(
                vt,
                (
                    FlexibleRowsMatrix,
                    FlexibleSizeMatrix,
                    FlexibleRowsColsMatrix,
                    FixedSizeMatrix,
                    FixedSizeTriangularMatrix,
                    FlexibleRowsColsLevelsMatrix,
                    FixedSizeLevelsRowsColsMatrix,
                    FixedSizeObjVectorMatrix,
                ),
            )
            is_vector = isinstance(vt, FixedSizeVector)

            if is_matrix or is_vector:
                all_matrix_params.append(param)
            else:
                if param.name not in {f[0] for f in self._push_constant_fields}:
                    self._push_constant_fields.append((param.name, "float" if isinstance(vt, Float) else "int"))
                    self._push_constant_map[param.name] = True

        def _is_literal(s):
            return s.lstrip("-").isdigit() if s else False

        for tb in triangular_bounds_raw:
            if _is_push_identifier(tb) and not _is_literal(tb) and tb not in {
                f[0] for f in self._push_constant_fields
            }:
                self._push_constant_fields.append((tb, "int"))
                self._push_constant_map[tb] = True

        bound_fields = [("rllm_bound_x", "int")]
        if node.space_dim >= 2:
            bound_fields.append(("rllm_bound_y", "int"))
        if node.space_dim >= 3:
            bound_fields.append(("rllm_bound_z", "int"))
        for name, vtype in reversed(bound_fields):
            if name not in {f[0] for f in self._push_constant_fields}:
                self._push_constant_fields.insert(0, (name, vtype))
                self._push_constant_map[name] = True

        # When parallelized, pass workgroup_count K to the shader for stride-based iteration.
        if node.parallelized and node.workgroup_count > 1:
            self._push_constant_fields.insert(0, ('rllm_wg_count', 'int'))
            self._push_constant_map['rllm_wg_count'] = True

        # ── Emit push_constant block before helper functions that reference it ──
        if self._push_constant_fields:
            self._emit("")
            self._emit("layout(push_constant) uniform RllmPushConstants {")
            self._push()
            for name, vtype in self._push_constant_fields:
                self._emit(f"{vtype} {name};")
            self._pop()
            self._emit("} rllm_push;")

        # ── Build constant map from constexpr declarations ──
        # Collect constexpr params first (so they're available for SSBO type resolution)
        constexpr_map = {}
        
        # Add param-level constexpr defines (for unresolved identifier resolution in types)
        for name, expr in node._param_constexpr_defines:
            constexpr_map[name] = expr
        
        # Then merge with body-level constexpr defines
        for name, expr in self._constexpr_defines:
            if name not in constexpr_map:
                constexpr_map[name] = expr
        
        # Resolve body-level constexpr values that may reference other constexpr names
        for name in list(constexpr_map.keys()):
            val = constexpr_map[name]
            if not re.match(r'^\d+$', val):
                resolved = self._resolve_constexpr_value(val)
                if resolved is not None:
                    constexpr_map[name] = resolved
        
        # Store the fully-resolved map on the visitor for body expression constant folding
        self._constexpr_map = {k: v for k, v in constexpr_map.items() if re.match(r'^\d+$', v)}

        
        # ── Emit SSBO buffers with compile-time sized arrays ──
        for param in all_matrix_params:
            vt = param.var_type
            is_obj_vector_matrix = isinstance(vt, FixedSizeObjVectorMatrix)
            is_3d = hasattr(vt, "level_expr") and vt.level_expr is not None
            inner = self._glsl_elem_type(vt)

            size = _compute_matrix_size(vt, constexpr_map)
            if size >= 2147483648:
                size_str = f" /* too large for int: {size} */ "
            elif size > 0:
                size_str = str(size)
            else:
                size_str = " /* unknown */ "

            if is_obj_vector_matrix:
                levels = _extract_size_from_expr(vt.level_expr, constexpr_map)
                flat_size = _extract_size_from_expr(vt.row_size_expr, constexpr_map) * _extract_size_from_expr(vt.col_size_expr, constexpr_map)
                flat_size_str = "" if flat_size >= 2147483648 else str(flat_size)
                for level in range(levels):
                    self._emit(
                        f"layout(std430, set = 0, binding = {self._binding_counter}) buffer RllmBuffer_{param.name}_{level} {{"
                    )
                    self._push()
                    self._emit(f"{inner} rllm_{param.name}_load_store_{level}[{flat_size_str}];")
                    self._pop()
                    self._emit("};")
                    self._binding_counter += 1
                self._emit(f"{inner} rllm_load_{param.name}(int level, int row, int col) {{")
                self._push()
                self._emit("int idx = row * rllm_push.rllm_bound_y + col;")
                self._emit("switch (level) {")
                self._push()
                for level in range(levels):
                    self._emit(f"case {level}: return rllm_{param.name}_load_store_{level}[idx];")
                self._emit("default: return 0;")
                self._pop()
                self._emit("}")
                self._pop()
                self._emit("}")
                self._emit(f"void rllm_store_{param.name}(int level, int row, int col, {inner} value) {{")
                self._push()
                self._emit("int idx = row * rllm_push.rllm_bound_y + col;")
                self._emit("switch (level) {")
                self._push()
                for level in range(levels):
                    self._emit(f"case {level}: rllm_{param.name}_load_store_{level}[idx] = value; return;")
                self._emit("default: return;")
                self._pop()
                self._emit("}")
                self._pop()
                self._emit("}")
            else:
                self._emit(
                    f"layout(std430, set = 0, binding = {self._binding_counter}) buffer RllmBuffer_{param.name} {{"
                )
                self._push()
                self._emit(f"{inner} {param.name}[{size_str}];")
                self._pop()
                self._emit("};")
                self._binding_counter += 1

            self._ssbo_map[param.name] = (is_3d, vt)

        # ── Main function ──
        self._emit("")
        self._emit("void main() {")
        self._push()

        # Tile variable names (when tiling is applied)
        if node.tiled:
            self._emit_tile_vars(node)

        # ── Parallelized initialization (GPU-wide parallel dispatch) ──
        parallelized = node.parallelized
        if parallelized and node.space_dim >= 1 and node.loop_vars:
            workgroup_count = node.workgroup_count
            
            # Compute global index using gl_GlobalInvocationID for true GPU-wide parallelism.
            # Each thread handles one element per invocation. When K > 1, the dispatch
            # dimensions cover all elements and threads use stride-based iteration if needed.
            self._parallel_global_vars = {}
            for idx, var_name in enumerate(node.loop_vars):
                dim_idx = min(idx, 2)
                dim = ['x', 'y', 'z'][dim_idx]
                global_var = f"global_{var_name}"
                self._emit(f"const int {global_var} = int(gl_GlobalInvocationID.{dim});")
                
                # Create an alias so that the original loop variable name is available in scope
                # for triangular guards and other code that references i, j etc. directly.
                self._emit(f"const int {var_name} = {global_var};")
                self._parallel_global_vars[var_name] = global_var

        # -- Implicit loop variable initialization --
        if not parallelized and node.loop_vars:
            self._parallel_global_vars = {}
            for idx, var_name in enumerate(node.loop_vars):
                dim_idx = min(idx, 2)
                dim = ['x', 'y', 'z'][dim_idx]
                global_var = f'global_{var_name}'
                self._emit(f'const int {global_var} = int(gl_GlobalInvocationID.{dim});')
                self._emit(f'const int {var_name} = {global_var};')
                self._parallel_global_vars[var_name] = global_var

        # 2. Initialize params from push constants
        for name, vtype in self._push_constant_fields:
            self._emit(f"{vtype} {name} = rllm_push.{name};")

        if not parallelized and node.loop_vars:
            parts = []
            for idx, var in enumerate(node.loop_vars):
                bound_name = f"rllm_bound_{'xyz'[idx]}"
                parts.append(f"{var} >= rllm_push.{bound_name}")
            if parts:
                self._emit(f"if ({' || '.join(parts)}) return;")

        # 3. Triangular guard (if applicable)
        if is_triangular:
            parts = []
            for idx, var in enumerate(node.loop_vars):
                bound_name = f"rllm_bound_{'xyz'[idx]}"
                parts.append(f"{var} >= rllm_push.{bound_name}")
            if len(node.loop_vars) >= 2:
                row_var = node.loop_vars[-2]
                col_var = node.loop_vars[-1]
                if triangular_kind == "upper":
                    parts.append(f"{row_var} > {col_var}")
                else:
                    parts.append(f"{col_var} > {row_var}")

            self._emit(f"if ({' || '.join(parts)}) return;")

        old_indent = self._indent_level
        self._indent_level += 1

        if getattr(node, 'parallelized', False) and node.space_dim >= 1 and node.loop_vars:
            # For parallelized programs, process each body statement with variable substitution
            for stmt in (node.body_stmts or []):
                result = self._emit_parallel_stmt(stmt, node)
                if isinstance(result, str):
                    self._emit(result.rstrip())
        elif getattr(node, 'tiled', False):
            # When tiling is applied, tile loops should come from gl_GlobalInvocationID.x/y
            # Skip explicit tile loop constructs and flatten their content into main() body
            body_stmts = self._flatten_tiled_body(node.body_stmts)
            for stmt in body_stmts:
                if hasattr(stmt, "accept"):
                    result = stmt.accept(self)
                    if isinstance(result, str):
                        self._emit(result.rstrip())
        else:
            for stmt in node.body_stmts:
                if hasattr(stmt, "accept"):
                    result = stmt.accept(self)
                    if isinstance(result, str):
                        self._emit(result.rstrip())

        self._indent_level = old_indent
        self._pop()
        self._emit("}")

        return self.result()
