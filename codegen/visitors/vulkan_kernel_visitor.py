"""Vulkan-compatible code generator (GLSL compute shader output).

Produces RLLM-style GLSL with:
- std430 SSBOs using compile-time sized arrays for matrix types with known dimensions
- push_constant block for all scalar params and triangular bounds
- Proper variable initialization from gl_GlobalInvocationID and rllm_push
- Triangular guard at start of main() for 3D triangular loops
"""

import re
from typing import Optional, Tuple

from .. import ast
from .visitor import Visitor


class VulkanKernelVisitor(Visitor):
    """Transforms the parsed AST into a Vulkan GLSL compute shader string."""

    def __init__(self):
        self._lines: list[str] = []
        self._indent_level: int = 0
        self._binding_counter: int = 0
        self._ssbo_map: dict[str, tuple] = {}   # param_name -> (is_3d, type_node)
        self._push_constant_fields: list[tuple] = []  # (name, type_str) pairs
        self._push_constant_map: dict[str, bool] = {}

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
        return "\n".join(self._lines) + "\n"

    # ── type helpers (GLSL) ────────────────────────────────────────────

    def _glsl_elem_type(self, ty) -> str:
        if hasattr(ty, 'elem_type') and ty.elem_type is not None:
            t = ty.elem_type
            if isinstance(t, (ast.Int, ast.Float)):
                return "float"
        return "float"

    def _extract_size_from_expr(self, expr) -> str:
        if isinstance(expr, ast.Number):
            return str(int(expr.value))
        s = self._to_str(expr)
        if s and re.match(r'^\d+$', s):
            return s
        return None

    def _compute_matrix_size(self, ty) -> str:
        if isinstance(ty, (ast.FixedSizeLevelsRowsColsMatrix, ast.FlexibleRowsColsLevelsMatrix)):
            parts = []
            for attr in ('level_expr', 'row_size_expr', 'col_size_expr'):
                expr = getattr(ty, attr, None)
                if expr:
                    s = self._extract_size_from_expr(expr)
                    if s:
                        parts.append(s)
            return "*".join(parts) if parts else "1"
        elif isinstance(ty, ast.FixedSizeMatrix):
            parts = []
            for attr in ('row_size_expr', 'col_size_expr'):
                expr = getattr(ty, attr, None)
                if expr:
                    s = self._extract_size_from_expr(expr)
                    if s:
                        parts.append(s)
            return "*".join(parts) if parts else "1"
        elif isinstance(ty, ast.FixedSizeVector):
            if ty.size_expr:
                s = self._extract_size_from_expr(ty.size_expr)
                if s:
                    return s
            return "1"
        return "1"

    def _to_str(self, node) -> str:
        """Convert an AST node to a GLSL string representation."""
        if node is None:
            return ""
        if isinstance(node, ast.Number):
            val = node.value
            if isinstance(val, float):
                s = f"{val}"
                if "." not in s:
                    s += ".0"
                return s
            return str(int(val))
        if isinstance(node, ast.Identifier):
            name = node.name
            if name in self._push_constant_map:
                return f"rllm_push.{name}"
            return name or "unknown"
        if isinstance(node, ast.LimitExpr):
            max_val = self._to_str(node.max_val) if node.max_val else "?"
            body = self._to_str(node.body) if node.body else "?"
            return f"limit<{max_val}>({body})"
        if isinstance(node, ast.BinaryExpr):
            left = self._to_str(node.left) if node.left else "?"
            right = self._to_str(node.right) if node.right else "?"
            op = node.op or "+"
            return f"({left} {op} {right})"
        if isinstance(node, ast.CastExpr):
            operand = self._to_str(node.operand) if node.operand else "?"
            return f"int({operand})"
        if isinstance(node, ast.ArrayAccess):
            base = node.base.accept(self) if node.base else "?"
            parts = []
            for idx in node.indices:
                idx_str = self._to_str(idx) if idx else "?"
                parts.append(idx_str)
            return f"{base}[{', '.join(parts)}]"
        if isinstance(node, ast.FieldAccess):
            base = ""
            if isinstance(node.base, ast.Identifier):
                base = node.base.name or "unknown"
            else:
                base = self._to_str(node.base)
            for field in node.fields:
                base += "." + field
            return base
        if isinstance(node, ast.IncCall):
            arg = self._to_str(node.operand) if hasattr(node, 'operand') and node.operand else "?"
            return f"inc({arg})"
        return str(node)

    # ── expression visitors ────────────────────────────────────────────

    def visit_type(self, node: ast.Type) -> str:
        return self._glsl_elem_type(node)

    def visit_int(self, node: ast.Int) -> str:
        return "int"

    def visit_float(self, node: ast.Float) -> str:
        return "float"

    def visit_fixed_size_vector(self, node: ast.FixedSizeVector) -> str:
        return self._glsl_elem_type(node)

    def visit_flexible_rows_matrix(self, node: ast.FlexibleRowsMatrix) -> str:
        return self._glsl_elem_type(node)

    def visit_fixed_size_matrix(self, node: ast.FixedSizeMatrix) -> str:
        return self._glsl_elem_type(node)

    def visit_fixed_size_levels_rows_cols_matrix(self, node: ast.FixedSizeLevelsRowsColsMatrix) -> str:
        return self._glsl_elem_type(node)

    def visit_flexible_rows_cols_levels_matrix(self, node: ast.FlexibleRowsColsLevelsMatrix) -> str:
        return self._glsl_elem_type(node)

    def visit_expression(self, node: ast.Expression) -> str:
        return node.accept(self)

    def visit_number(self, node: ast.Number) -> str:
        return self._to_str(node)

    def visit_identifier(self, node: ast.Identifier) -> str:
        name = node.name
        if name in self._push_constant_map:
            return f"rllm_push.{name}"
        return name or "unknown"

    def visit_array_access(self, node: ast.ArrayAccess) -> str:
        return self._to_str(node)

    def visit_field_access(self, node: ast.FieldAccess) -> str:
        return self._to_str(node)

    def visit_limit_expr(self, node: ast.LimitExpr) -> str:
        max_val = self._to_str(node.max_val) if node.max_val else "?"
        body = self._to_str(node.body) if node.body else "?"
        return f"limit<{max_val}>({body})"

    def visit_binary_expr(self, node: ast.BinaryExpr) -> str:
        left = self._to_str(node.left) if node.left else "?"
        right = self._to_str(node.right) if node.right else "?"
        op = node.op or "+"
        return f"({left} {op} {right})"

    def visit_cast_expr(self, node: ast.CastExpr) -> str:
        operand = self._to_str(node.operand) if node.operand else "?"
        return f"int({operand})"

    def visit_negation_expr(self, node: ast.NegationExpr) -> str:
        op = "!" + (self._to_str(node.operand) if node.operand else "?")
        return op

    # ── condition visitor ──────────────────────────────────────────────

    def visit_condition(self, node: ast.Condition) -> str:
        lhs = self._to_str(node.lhs) if node.lhs else "?"
        rhs = self._to_str(node.rhs) if node.rhs else "?"
        return f"{lhs} {node.op} {rhs}"

    # ── statement visitors (return string, caller emits with indent) ──

    def visit_statement(self, node: ast.Statement) -> str:
        return node.accept(self)

    def visit_for(self, node: ast.For) -> str:
        """Generate a GLSL for loop as a string."""
        lines = []
        ind = "    " * (self._indent_level + 1)

        # Determine upper bound
        upper_bound = "?"
        lower_bound = "0"
        
        if node.init_expr and isinstance(node.init_expr, ast.LimitExpr):
            max_val = self._to_str(node.init_expr.max_val) if node.init_expr.max_val else "?"
            upper_bound = max_val
        elif node.condition:
            op = node.condition.op or ">="
            rhs = self._to_str(node.condition.rhs) if node.condition.rhs else "?"
            lower_bound = self._to_str(node.condition.lhs) if node.condition.lhs else "0"
            upper_bound = rhs

        inc_var = node.increment_var if node.increment_var else node.loop_var_name
        inc_op = node.increment_op if node.increment_op else "++"

        lines.append(f"{ind}for (int {node.loop_var_name} = {lower_bound}; {node.loop_var_name} < {upper_bound}; {inc_op}{inc_var}) {{")
        
        for stmt in node.body_stmts:
            if hasattr(stmt, 'accept'):
                result = stmt.accept(self)
                if isinstance(result, str):
                    lines.append(f"{ind}{result}")

        lines.append(f"{ind}}}")
        return "\n".join(lines) + "\n"

    def visit_if(self, node: ast.If) -> str:
        cond_str = self.visit_condition(node.condition) if node.condition else "?"
        ind = "    " * (self._indent_level + 1)
        lines = [f"{ind}if ({cond_str}) {{"]
        
        for stmt in node.body_stmts:
            if hasattr(stmt, 'accept'):
                result = stmt.accept(self)
                if isinstance(result, str):
                    lines.append(f"{ind}{result}")

        lines.append(f"{ind}}}")
        return "\n".join(lines) + "\n"

    def visit_declaration(self, node: ast.Declaration) -> str:
        prefix = "const " if node.is_const else ""
        var_type = "float"
        init_str = ""
        if node.init_expr is not None:
            init_str = f" = {self._to_str(node.init_expr)}"
        return f"{prefix}{var_type} {node.name}{init_str};"

    def visit_assignment(self, node: ast.Assignment) -> str:
        lvalue = self._to_str(node.lvalue) if node.lvalue else "?"
        rvalue = self._to_str(node.rvalue) if node.rvalue else "?"
        return f"{lvalue} {node.assign_op} {rvalue};"

    def visit_overflow_check(self, node: ast.OverflowCheck) -> str:
        lvalue = self._to_str(node.lvalue) if node.lvalue else "?"
        operand = self._to_str(node.operand) if node.operand else "?"
        return f"OVERFLOW_CHECK_ADD({lvalue}, {operand});"

    def visit_shared_decl(self, node: ast.SharedDecl) -> str:
        prefix = "const " if node.is_const else ""
        var_type = "float"
        init_str = ""
        if node.init_expr is not None:
            init_str = f" = {self._to_str(node.init_expr)}"
        return f"shared {prefix}{var_type} {node.name}{init_str};"

    def visit_workgroup_properties(self, node: ast.WorkgroupProperties) -> str:
        parts = []
        if node.x_expr is not None:
            parts.append(f"x: {self._to_str(node.x_expr)}")
        if node.y_expr is not None:
            parts.append(f"y: {self._to_str(node.y_expr)}")
        if node.z_expr is not None:
            parts.append(f"z: {self._to_str(node.z_expr)}")
        return f"workgroup {{ {', '.join(parts)} }}"

    # ── Program visitor (main entry point) ────────────────────────────

    def visit_program(self, node: ast.Program) -> str:
        self._lines = []
        self._indent_level = 0
        self._binding_counter = 0
        self._ssbo_map = {}
        self._push_constant_fields = []
        self._push_constant_map = {}

        self._emit("#version 450")
        self._emit("")

        # ── Classify parameters ──
        all_matrix_params = []
        triangular_bounds_raw = node.triangular_bounds_raw if getattr(node, 'triangular_bounds_raw', None) else []
        is_triangular = len(triangular_bounds_raw) >= 2

        for param in node.params:
            if param is None or not isinstance(param, ast.Declaration):
                continue
            
            vt = param.var_type
            is_matrix = isinstance(vt, (ast.FlexibleRowsMatrix, ast.FixedSizeMatrix,
                                        ast.FlexibleRowsColsLevelsMatrix, ast.FixedSizeLevelsRowsColsMatrix))
            is_vector = isinstance(vt, ast.FixedSizeVector)
            
            if is_matrix or is_vector:
                all_matrix_params.append(param)
            else:
                if param.name not in {f[0] for f in self._push_constant_fields}:
                    self._push_constant_fields.append((param.name, "int"))
                    self._push_constant_map[param.name] = True

        def _is_literal(s):
            return s.lstrip('-').isdigit() if s else False
        
        for tb in triangular_bounds_raw:
            if not _is_literal(tb) and tb not in {f[0] for f in self._push_constant_fields}:
                self._push_constant_fields.append((tb, "int"))
                self._push_constant_map[tb] = True

        # ── Emit SSBO buffers with compile-time sized arrays ──
        for param in all_matrix_params:
            vt = param.var_type
            is_3d = hasattr(vt, 'level_expr') and vt.level_expr is not None
            inner = self._glsl_elem_type(vt)
            size_str = self._compute_matrix_size(vt)
            
            self._emit(f"layout(std430, set = 0, binding = {self._binding_counter}) buffer RllmBuffer_{param.name} {{")
            self._push()
            self._emit(f"{inner} {param.name}[{size_str}];")
            self._pop()
            self._emit(f"}} {param.name};")
            
            self._ssbo_map[param.name] = (is_3d, vt)
            self._binding_counter += 1

        # ── Emit push_constant block ──
        if self._push_constant_fields:
            self._emit("")
            self._emit("layout(push_constant) uniform RllmPushConstants {")
            self._push()
            for name, vtype in self._push_constant_fields:
                self._emit(f"{vtype} {name};")
            self._pop()
            self._emit("} rllm_push;")

        # ── Main function ──
        self._emit("")
        self._emit("void main() {")
        self._push()

        # 1. Initialize loop variables from gl_GlobalInvocationID
        if node.space_dim >= 1 and node.loop_vars:
            for idx, var_name in enumerate(node.loop_vars):
                coord = 'xyz'[idx]
                self._emit(f"int {var_name} = int(gl_GlobalInvocationID.{coord});")

        # 2. Initialize params from push constants
        for name, vtype in self._push_constant_fields:
            if vtype == "int":
                self._emit(f"int {name} = rllm_push.{name};")

        # 3. Triangular guard (if applicable)
        if is_triangular and triangular_bounds_raw and len(triangular_bounds_raw) >= 2:
            lower_name = triangular_bounds_raw[0]
            upper_name = triangular_bounds_raw[1]
            
            parts = []
            for i, var in enumerate(node.loop_vars):
                if _is_literal(lower_name):
                    parts.append(f"{var} >= {lower_name}")
                else:
                    parts.append(f"{var} >= rllm_push.{lower_name}")
            
            for var in node.loop_vars[1:]:
                if _is_literal(upper_name):
                    parts.append(f"{var} >= {upper_name}")
                else:
                    parts.append(f"{var} >= rllm_push.{upper_name}")
            
            self._emit(f"if ({' || '.join(parts)}) return;")

        # 4. Body statements - just emit directly (no double-emission)
        old_indent = self._indent_level
        self._indent_level += 1
        
        for stmt in node.body_stmts:
            if hasattr(stmt, 'accept'):
                result = stmt.accept(self)
                if isinstance(result, str):
                    # Strip trailing newline and emit with current indent
                    self._emit(result.rstrip())

        self._indent_level = old_indent
        self._pop()
        self._emit("}")

        return self.result()
