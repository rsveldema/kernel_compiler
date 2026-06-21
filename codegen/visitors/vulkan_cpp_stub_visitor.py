"""C++ stub generator for Vulkan compute kernels.

Generates compilable C++ header with:
- Struct wrappers matching the SSBO layout (RllmBuffer_<name>)
- Push constants struct for scalar params and triangular bounds
- Descriptor set struct for buffer bindings
- A dispatch function matching the kernel's parameters
"""

from typing import Any, Dict, List, Tuple
from .. import kast as ast
from .visitor import Visitor

import re

_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")
_CPP_RESERVED = {"None", "static_cast"}


def _is_push_identifier(value: str | None) -> bool:
    return bool(value and _IDENT_RE.match(value) and value not in _CPP_RESERVED)

def type_to_cpp_string(ty, use_bfloat16: bool = False) -> str:
    """Convert a Type AST node to its C++ string representation."""
    if ty is None:
        return "unknown_type"
    if isinstance(ty, ast.Int):
        return "int32_t"
    if isinstance(ty, ast.Float):
        return "float"
    if isinstance(ty, ast.Float16):
        return "bfloat_t" if use_bfloat16 else "float16_t"
    if isinstance(ty, ast.FixedSizeVector) and ty.elem_type is not None:
        elem = type_to_cpp_string(ty.elem_type, use_bfloat16)
        size = ty.size_expr.accept(CastExprTypeConverter()) if hasattr(ty.size_expr, 'accept') else "?"
        return f"{elem}[{size}]"
    # For all other types (vectors/matrices) used as casts, fall back to unknown
    return "unknown_type"


# Minimal converter for size expressions inside type_to_cpp_string
class CastExprTypeConverter(Visitor):
    def _visit_expr_child(self, node):
        if node is None:
            return "?"
        return node.accept(self)

    def visit_number(self, node):
        return str(node.value)

    def visit_identifier(self, node):
        return node.name or "?"

    def visit_binary_expr(self, node):
        left = self._visit_expr_child(node.left)
        right = self._visit_expr_child(node.right)
        op = node.op or "+"
        return f"({left} {op} {right})"

    def accept(self, visitor):
        raise NotImplementedError


class VulkanCppStubVisitor(Visitor):
    """Transforms the parsed AST into a C++ stub for calling the Vulkan kernel."""

    def __init__(self, use_bfloat16: bool = False):
        self._use_bfloat16 = use_bfloat16
        self._lines: List[str] = []
        self._indent_level: int = 0
        self._buffer_structs: Dict[str, str] = {}  # name -> struct_name
        self._kernel_name: str = ""
        self._namespace_name: str = ""
        self._display_name: str = ""
        self._space_dim: int = 1


    def _cpp_dimension(self) -> str:
        mapping = {1: "OneD", 2: "TwoD", 3: "ThreeD"}
        return mapping.get(self._space_dim, "OneD")

    def _cpp_type(self) -> str:
        return self._kernel_type
    # ── helpers ────────────────────────────────────────────────────────

    def _emit(self, line: str = "") -> None:
        self._lines.append("    " * self._indent_level + line)

    def _push(self) -> None:
        self._indent_level += 1

    def _pop(self) -> None:
        self._indent_level -= 1

    def result(self) -> str:
        return "\n".join(self._lines) + "\n"
    
    def _generate_type_aliases_header(self) -> tuple[str, str]:
        """Generate the kfloat type aliases header filename and content."""
        kfloat_type = "bfloat_t" if self._use_bfloat16 else "float"
        filename = "_type_aliases.hpp"
        content = (
            "#pragma once\n"
            "#include <cstdint>\n"
            "using float16_t = _Float16;\n"
            "using bfloat_t = uint16_t;\n"
            f"using kfloat = {kfloat_type};\n"
        )
        return (filename, content)


    def _cpp_type_str(self, ty) -> str:
        if isinstance(ty, ast.Int):
            return "int32_t"
        if isinstance(ty, ast.Float):
            return "float"
        if isinstance(ty, ast.Float16):
            return "bfloat_t" if self._use_bfloat16 else "float16_t"
        if isinstance(ty, ast.CoopMat):
            return "void"
        return "unknown_type"

    def _class_name(self, kernel_name: str) -> str:
        return "".join(part[:1].upper() + part[1:] for part in kernel_name.split("_") if part) + "Kernel"

    # ── expression visitors ────────────────────────────────────────────

    def visit_type(self, node: ast.Type) -> str:
        return self._cpp_type_str(node)

    def visit_int(self, node: ast.Int) -> str:
        return "int32_t"

    def visit_float(self, node: ast.Float) -> str:
        return "float"

    def visit_float16(self, node: ast.Float16) -> str:
        return "bfloat_t" if self._use_bfloat16 else "float16_t"

    def visit_coop_mat(self, node: ast.CoopMat) -> str:
        return "void"

    def visit_fixed_size_vector(self, node: ast.FixedSizeVector) -> str:
        elem = self._cpp_type_str(node.elem_type) if node.elem_type else "float"
        return f"{elem}"

    def visit_flexible_rows_matrix(self, node: ast.FlexibleRowsMatrix) -> str:
        return ""  # handled in visit_program

    def visit_flexible_size_matrix(self, node: ast.FlexibleSizeMatrix) -> str:
        return ""  # handled in visit_program

    def visit_fixed_size_matrix(self, node: ast.FixedSizeMatrix) -> str:
        return ""

    def visit_fixed_size_triangular_matrix(self, node: ast.FixedSizeTriangularMatrix) -> str:
        return ""

    def visit_fixed_size_obj_vector_matrix(self, node: ast.FixedSizeObjVectorMatrix) -> str:
        return ""  # handled in visit_program

    def visit_expression(self, node: ast.Expression) -> str:
        return node.accept(self)

    def visit_number(self, node: ast.Number) -> str:
        num_value = int(node.value)
        s = str(num_value)
        if getattr(node, "unsigned", False):
            if abs(num_value) > 4294967295:  # UINT_MAX — need uint64
                return s + "ULL"
            return s + "U"
        return s

    def visit_identifier(self, node: ast.Identifier) -> str:
        return node.name or "unknown"

    def visit_array_access(self, node: ast.ArrayAccess) -> str:
        base = node.base.accept(self) if node.base else ""
        indices = ", ".join(i.accept(self) for i in node.indices)
        return f"{base}[{indices}]"

    def visit_field_access(self, node: ast.FieldAccess) -> str:
        base = node.base.accept(self) if node.base else ""
        base += "." + node.field
        return base

    def visit_call_expr(self, node: ast.CallExpr) -> str:
        callee = node.callee.accept(self) if node.callee else ""
        args = ", ".join(arg.accept(self) for arg in node.args)
        return f"{callee}({args})"

    def _to_str(self, node) -> str:
        if node is None:
            return ""
        return node.accept(self)

    def _visit_expr_child(self, node):
        """Helper for visiting expression children (returns '?' for None)."""
        if node is None:
            return "?"
        return node.accept(self)

    def visit_binary_expr(self, node: ast.BinaryExpr) -> str:
        left = self._visit_expr_child(node.left)
        right = self._visit_expr_child(node.right)
        op = node.op or "+"
        return f"({left} {op} {right})"

    def visit_cast_expr(self, node: ast.CastExpr) -> str:
        cast_type = type_to_cpp_string(node.cast_type, self._use_bfloat16)
        operand = self._visit_expr_child(node.operand)
        return f"static_cast<{cast_type}>({operand})"

    def visit_negation_expr(self, node: ast.NegationExpr) -> str:
        operand = self._visit_expr_child(node.operand)
        return f"!{operand}"

    def visit_wildcard_expression(self, node: ast.WildcardExpression) -> str:
        return f"wildcard({node.name})"

    # ── condition visitor ──────────────────────────────────────────────

    def visit_condition(self, node: ast.Condition) -> str:
        lhs = (
            node.lhs.accept(self)
            if isinstance(node.lhs, ast.Expression)
            else (node.lhs or "")
        )
        rhs = (
            node.rhs.accept(self)
            if isinstance(node.rhs, ast.Expression)
            else (node.rhs or "")
        )
        return f"{lhs} {node.op} {rhs}"

    # ── statement visitors ─────────────────────────────────────────────

    def visit_statement(self, node: ast.Statement) -> str:
        return node.accept(self)

    def visit_for_loop_range(self, node: ast.ForLoopRange) -> str:
        return ""

    def visit_for_loop_with_condition_and_increment(
        self, node: ast.ForLoopWithConditionAndIncrement
    ) -> str:
        return ""

    def visit_raw_statement(self, node: ast.RawStatement) -> str:
        return node.text.rstrip()

    def visit_call_statement(self, node: ast.CallStatement) -> str:
        return f"{node.call_expr.accept(self)};"

    def visit_wildcard_statement(self, node: ast.WildcardStatement) -> str:
        return f"wildcard({node.name})"

    # ── workgroup and program visitors ────────────────────────────────

    def _is_matrix_type(self, ty) -> bool:
        """Check if a type is a matrix or vector that maps to an SSBO."""
        return isinstance(
            ty,
            (
                ast.FlexibleRowsMatrix,
                ast.FlexibleSizeMatrix,
                ast.FlexibleRowsColsMatrix,
                ast.FixedSizeMatrix,
                ast.FixedSizeTriangularMatrix,
                ast.FlexibleRowsColsLevelsMatrix,
                ast.FixedSizeLevelsRowsColsMatrix,
                ast.FixedSizeObjVectorMatrix,
                ast.FixedSizeVector,
            ),
        )

    def _compute_matrix_size(self, ty) -> str:
        """Compute compile-time array size for a matrix/vector type.
        
        Uses unsigned 64-bit literals when the total would overflow signed 32-bit int.
        """
        if isinstance(
            ty, (ast.FixedSizeLevelsRowsColsMatrix, ast.FlexibleRowsColsLevelsMatrix)
        ):
            parts = []
            for attr in ("level_expr", "row_size_expr", "col_size_expr"):
                expr = getattr(ty, attr, None)
                if expr and isinstance(expr, ast.Number):
                    val = int(expr.value)
                    parts.append(str(val))
            if not parts:
                return "1"
            total = 1
            for p in parts:
                total *= int(p)
            return str(total)
        elif isinstance(ty, ast.FixedSizeTriangularMatrix):
            row_expr = getattr(ty, "row_size_expr", None)
            col_expr = getattr(ty, "col_size_expr", None)
            row = int(row_expr.value) if isinstance(row_expr, ast.Number) else 1
            col = int(col_expr.value) if isinstance(col_expr, ast.Number) else row
            if row != col:
                raise ValueError("fixed_size_triangular_matrix must be square")
            return str(row * (row + 1) // 2)
        elif isinstance(ty, (ast.FlexibleRowsMatrix, ast.FlexibleSizeMatrix, ast.FlexibleRowsColsMatrix, ast.FixedSizeMatrix, ast.FixedSizeObjVectorMatrix)):
            parts = []
            for attr in ("row_size_expr", "col_size_expr"):
                expr = getattr(ty, attr, None)
                if expr and isinstance(expr, ast.Number):
                    val = int(expr.value)
                    parts.append(str(val))
            if not parts:
                return "1"
            total = 1
            for p in parts:
                total *= int(p)
            return str(total)
        elif (
            isinstance(ty, ast.FixedSizeVector)
            and ty.size_expr
            and isinstance(ty.size_expr, ast.Number)
        ):
            val = int(ty.size_expr.value)
            if val > 2147483647:
                return f"{val}ULL"
            return str(val)
        return "1"

    def _buffer_element_cpp_type(self, ty) -> str:
        elem_type = getattr(ty, "elem_type", None)
        if elem_type is None:
            return "float"
        return self._cpp_type_str(elem_type)

    def _number_literal(self, expr) -> str | None:
        if expr and isinstance(expr, ast.Number):
            return str(int(expr.value))
        return None

    def _buffer_dimension_constants(self, name: str, ty) -> List[Tuple[str, str]]:
        if isinstance(
            ty, (ast.FixedSizeLevelsRowsColsMatrix, ast.FlexibleRowsColsLevelsMatrix)
        ):
            dims = [
                ("X", self._number_literal(getattr(ty, "level_expr", None))),
                ("Y", self._number_literal(getattr(ty, "row_size_expr", None))),
                ("Z", self._number_literal(getattr(ty, "col_size_expr", None))),
            ]
        elif isinstance(ty, (ast.FlexibleRowsMatrix, ast.FlexibleSizeMatrix, ast.FlexibleRowsColsMatrix, ast.FixedSizeMatrix, ast.FixedSizeTriangularMatrix, ast.FixedSizeObjVectorMatrix)):
            dims = [
                ("X", self._number_literal(getattr(ty, "row_size_expr", None))),
                ("Y", self._number_literal(getattr(ty, "col_size_expr", None))),
            ]
        elif isinstance(ty, ast.FixedSizeVector):
            dims = [("X", self._number_literal(getattr(ty, "size_expr", None)))]
        else:
            dims = []
        return [(f"{name}_{axis}", value) for axis, value in dims if value is not None]

    def visit_workgroup_properties(self, node: ast.WorkgroupProperties) -> dict:
        return {
            "x": self._to_str(node.x_expr) if node.x_expr else "8",
            "y": self._to_str(node.y_expr) if node.y_expr else "8",
            "z": self._to_str(node.z_expr) if node.z_expr else "1",
        }

    def _initialize_program_emit_state(self) -> None:
        self._lines = []
        self._indent_level = 0
        self._buffer_structs = {}
        self._kernel_name = ""
        self._namespace_name = ""

    def _configure_kernel_identity(self, node: ast.Program) -> None:
        basename = "kernel"
        self._display_name = "unknown"
        if node.header:
            hdr_raw = node.header.strip('"')
            colon_idx = hdr_raw.rfind(":")
            if colon_idx >= 0:
                raw_class = hdr_raw[:colon_idx]
                dot_idx = raw_class.rfind(".")
                if dot_idx > 0:
                    raw_class = raw_class[:dot_idx]
                lineno_str = hdr_raw[colon_idx + 1:]
                self._display_name = f"{raw_class}:{lineno_str}"
            parts = node.header.replace('"', "").split("/")
            basename = parts[-1].rsplit(".", 1)[0] if "." in parts[-1] else parts[-1]

        src_name = node._source_filename
        if src_name:
            src_stem = src_name.rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("-", "_")
        else:
            src_stem = "kernel"

        self._kernel_name = f"{src_stem}_{basename}" if basename != "kernel" else src_stem
        self._namespace_name = f"rllm_{src_stem}"

    def _classify_program_params(
        self, node: ast.Program
    ) -> tuple[List[Tuple[ast.Declaration, str, str]], List[ast.Declaration], bool]:
        buffer_params: List[Tuple[ast.Declaration, str, str]] = []
        scalar_params: List[ast.Declaration] = []
        has_matrix_params = False

        for param in node.params:
            if not isinstance(param, ast.Declaration):
                continue

            vt = param.var_type
            if self._is_matrix_type(vt):
                if isinstance(vt, ast.FixedSizeObjVectorMatrix):
                    levels = int(vt.level_expr.value) if isinstance(vt.level_expr, ast.Number) else 1
                    for level in range(levels):
                        sname = f"RllmBuffer_{param.name}_{level}"
                        buffer_params.append((param, sname, f"{param.name}_{level}"))
                    has_matrix_params = True
                else:
                    sname = f"RllmBuffer_{param.name}"
                    self._buffer_structs[param.name] = sname
                    has_matrix_params = True
                    buffer_params.append((param, sname, param.name))
            else:
                scalar_params.append(param)

        return buffer_params, scalar_params, has_matrix_params

    # Vulkan spec minimum required limits (Vulkan 1.4 Table A.29)
    _MAX_WORKGROUP_SIZE = 1024        # maxWorkGroupSize — per dimension
    _MAX_WORKGROUP_INVOCATIONS = 1024 # maxWorkGroupInvocations — total

    def _resolve_workgroup_sizes(self, node: ast.Program) -> tuple[str, str, str]:
        # Default workgroup sizes: 16x16x1 for 2D/3D, 16x1x1 for 1D
        if node.space_dim >= 2:
            wg_x, wg_y, wg_z = "16", "16", "1"
        else:
            wg_x, wg_y, wg_z = "16", "1", "1"
        found_explicit = False
        for wg in node.workgroups:
            if isinstance(wg, ast.WorkgroupProperties):
                found_explicit = True
                x_val = self._to_str(wg.x_expr) if wg.x_expr else "1"
                y_val = self._to_str(wg.y_expr) if wg.y_expr else "1"
                z_val = self._to_str(wg.z_expr) if wg.z_expr else "1"
                if x_val.isdigit():
                    wg_x = x_val
                if y_val.isdigit():
                    wg_y = y_val
                if z_val.isdigit():
                    wg_z = z_val
        if not found_explicit and node.space_dim >= 2 and node.tile_size_x > 1 and node.tile_size_y > 1:
            wg_x, wg_y, wg_z = str(node.tile_size_x), str(node.tile_size_y), "1"
        if node.reduction_chunks > 1:
            wg_z = str(node.reduction_chunks)
            if node.space_dim >= 2:
                wg_x = "8"
                wg_y = "8"
        # Validate against Vulkan spec minimum required limits
        try:
            x_int, y_int, z_int = int(wg_x), int(wg_y), int(wg_z)
        except ValueError:
            # Non-numeric workgroup sizes (e.g. expressions) — can't validate at compile time
            return wg_x, wg_y, wg_z
        if x_int > self._MAX_WORKGROUP_SIZE:
            raise ValueError(
                f"workgroup x size {x_int} exceeds Vulkan maxWorkGroupSize "
                f"({self._MAX_WORKGROUP_SIZE})"
            )
        if y_int > self._MAX_WORKGROUP_SIZE:
            raise ValueError(
                f"workgroup y size {y_int} exceeds Vulkan maxWorkGroupSize "
                f"({self._MAX_WORKGROUP_SIZE})"
            )
        if z_int > self._MAX_WORKGROUP_SIZE:
            raise ValueError(
                f"workgroup z size {z_int} exceeds Vulkan maxWorkGroupSize "
                f"({self._MAX_WORKGROUP_SIZE})"
            )
        total = x_int * y_int * z_int
        if total > self._MAX_WORKGROUP_INVOCATIONS:
            raise ValueError(
                f"workgroup total size {x_int}x{y_int}x{z_int}={total} "
                f"exceeds Vulkan maxWorkGroupInvocations "
                f"({self._MAX_WORKGROUP_INVOCATIONS})"
            )
        return wg_x, wg_y, wg_z

    def _build_push_constant_fields(
        self,
        scalar_params: List[ast.Declaration],
        has_2d: bool,
        has_3d: bool,
        is_triangular: bool,
        node: ast.Program,
    ) -> List[Any]:
        pc_field_names = set()
        all_pc_fields: List[Any] = []

        if "rllm_bound_x" not in pc_field_names:
            pc_field_names.add("rllm_bound_x")
            all_pc_fields.append(type("", (), {"name": "rllm_bound_x", "var_type": ast.Int()})())
        if has_2d and "rllm_bound_y" not in pc_field_names:
            pc_field_names.add("rllm_bound_y")
            all_pc_fields.append(type("", (), {"name": "rllm_bound_y", "var_type": ast.Int()})())
        if has_3d and "rllm_bound_z" not in pc_field_names:
            pc_field_names.add("rllm_bound_z")
            all_pc_fields.append(type("", (), {"name": "rllm_bound_z", "var_type": ast.Int()})())

        for param in scalar_params:
            if param.name not in pc_field_names:
                pc_field_names.add(param.name)
                all_pc_fields.append(param)

        if is_triangular:
            for tb in node.triangular_bounds_raw:
                if tb is None:
                    continue
                if _is_push_identifier(tb) and tb not in pc_field_names and not tb.lstrip("-").isdigit():
                    pc_field_names.add(tb)
                    all_pc_fields.append(type("", (), {"name": tb, "var_type": ast.Int()})())

        return all_pc_fields

    def _build_method_params(
        self,
        has_2d: bool,
        has_3d: bool,
        buffer_params: List[Tuple[ast.Declaration, str, str]],
        all_pc_fields: List[Any],
    ) -> List[str]:
        method_params: List[str] = ["        VulkanComputeContext& context"]

        if has_3d:
            method_params.append("        uint32_t dispatch_rows")
            method_params.append("        uint32_t dispatch_cols")
            method_params.append("        uint32_t dispatch_levels")
        elif has_2d:
            method_params.append("        uint32_t dispatch_rows")
            method_params.append("        uint32_t dispatch_cols")
        else:
            method_params.append("        uint32_t dispatch_rows")

        for param, _sname, arg_name in buffer_params:
            const_prefix = "const " if getattr(param, "is_const", False) else ""
            method_params.append(f"        {const_prefix}VBaseDeviceBuffer& {arg_name}")

        if all_pc_fields:
            method_params.append(f"        const {self._kernel_name}_PushConstants& push_constants")

        return method_params

    def _emit_header_prelude(self, node: ast.Program) -> None:
        self._emit("")
        hdr_text = node.header.strip('"') if node.header else "unknown"
        self._emit(f"// ── Kernel dispatch stub: {hdr_text} ─────────────")
        self._emit("#include <cstdint>")
        self._emit("#include <string>")
        self._emit("#include <vulkan/vulkan_core.h>")
        self._emit('#include "vulkan_session.hpp"')
        self._emit('#include "_type_aliases.hpp"')
        self._emit("")
        self._emit(f"namespace {self._namespace_name} {{")
        self._emit("")

    def _emit_buffer_structs(
        self, buffer_params: List[Tuple[ast.Declaration, str, str]]
    ) -> None:
        emitted_struct_names = set()
        emitted_dim_constants = set()
        for param, sname, _arg_name in buffer_params:
            if sname in emitted_struct_names:
                continue
            emitted_struct_names.add(sname)
            vt = param.var_type
            size_str = self._compute_matrix_size(vt)
            dim_constants = self._buffer_dimension_constants(param.name, vt)
            new_constants = False
            for cname, cvalue in dim_constants:
                if cname in emitted_dim_constants:
                    continue
                emitted_dim_constants.add(cname)
                new_constants = True
                self._emit(f"inline constexpr uint32_t {cname} = {cvalue};")
            if new_constants:
                self._emit("")

            self._emit(f"struct {sname} {{")
            self._push()
            elem_type = self._buffer_element_cpp_type(vt)
            self._emit(f"    {elem_type} data[{size_str}];")
            self._pop()
            self._emit("};")
            self._emit("")

    def _emit_push_constant_struct(self, all_pc_fields: List[Any]) -> None:
        if not all_pc_fields:
            return
        pc_name = f"{self._kernel_name}_PushConstants"
        self._emit(f"struct {pc_name} {{")
        self._push()
        for field in all_pc_fields:
            ctype = "float" if isinstance(getattr(field, "var_type", None), ast.Float) else "int32_t"
            self._emit(f"    {ctype} {field.name};")
        self._pop()
        self._emit("};")
        self._emit("")

    def _compute_dispatch_dims(
        self,
        node: ast.Program,
        has_2d: bool,
        has_3d: bool,
        wg_x: str,
        wg_y: str,
        wg_z: str,
    ) -> tuple[str, str, str]:
        if has_3d:
            x_dim = f"(dispatch_rows + {wg_x} - 1) / {wg_x}"
            y_dim = f"(dispatch_cols + {wg_y} - 1) / {wg_y}"
            z_dim = f"(dispatch_levels + {wg_z} - 1) / {wg_z}"
        elif has_2d:
            x_dim = f"(dispatch_rows + {wg_x} - 1) / {wg_x}"
            y_dim = f"(dispatch_cols + {wg_y} - 1) / {wg_y}"
            z_dim = "1"
        else:
            x_dim = f"(dispatch_rows + {wg_x} - 1) / {wg_x}"
            y_dim = "1"
            z_dim = "1"
        return x_dim, y_dim, z_dim

    def _emit_kernel_class(
        self,
        node: ast.Program,
        class_name: str,
        method_params: List[str],
        all_pc_fields: List[Any],
        buffer_params: List[Tuple[ast.Declaration, str, str]],
        has_2d: bool,
        has_3d: bool,
        wg_x: str,
        wg_y: str,
        wg_z: str,
    ) -> None:
        push_size = f"sizeof({self._kernel_name}_PushConstants)" if all_pc_fields else "0"
        tile_bs = node.tile_block_size
        if tile_bs is None:
            tile_bs = 1
        wg_count = node.workgroup_count
        if wg_count is None:
            wg_count = 1
        descriptor = (
            f"tile_block_size={tile_bs};"
            f"tile_size_x={node.tile_size_x};"
            f"tile_size_y={node.tile_size_y};"
            f"tile_chunk_size={node.tile_chunk_size};"
            f"workgroup_count={wg_count};"
            f"tree_transformed={'yes' if node.tree_transformed else 'no'};"
        )

        self._emit(f"class {class_name} : public AbstractKernel {{")
        self._emit("public:")
        self._push()
        self._emit(f"{class_name}(VulkanSession& session, const std::string& glsl_file)")
        kernel_name_str = '"' + self._namespace_name + '::' + class_name + '|' + self._display_name + '", '
        kernel_type_str = 'KernelDimension::' + self._cpp_dimension() + ', KernelType::' + self._cpp_type()
        descriptor_str = '"' + descriptor + '"'
        self._emit('    : AbstractKernel(' + kernel_name_str + kernel_type_str + ', ' + descriptor_str + ')')
        self._emit(f"    , kernel_(session, glsl_file, {push_size}, {len(buffer_params)})")
        self._emit("{")
        self._emit("}")
        self._emit("")

        self._emit(f"static constexpr const char* generated_descriptor() {{ return \"{descriptor}\"; }}")
        self._emit("")

        self._emit("void dispatch(")
        for i, fp in enumerate(method_params):
            comma = "," if i < len(method_params) - 1 else ""
            self._emit(fp + comma)
        self._emit(") {")
        self._push()

        x_dim, y_dim, z_dim = self._compute_dispatch_dims(node, has_2d, has_3d, wg_x, wg_y, wg_z)

        self._emit("ComputeKernelRegistry::ScopedActiveKernel active_kernel(*this);")
        self._emit("VkCommandBuffer command_buffer = context.begin_command_buffer();")

        if buffer_params:
            self._emit("VkDescriptorSet desc_set = kernel_.desc_set();")
            self._emit(f"VkDescriptorBufferInfo buffer_infos[{len(buffer_params)}]{{}};")
            self._emit(f"VkWriteDescriptorSet writes[{len(buffer_params)}]{{}};")
            for i, (_param, _sname, arg_name) in enumerate(buffer_params):
                self._emit(f"buffer_infos[{i}].buffer = {arg_name}.get();")
                self._emit(f"buffer_infos[{i}].offset = 0;")
                self._emit(f"buffer_infos[{i}].range = VK_WHOLE_SIZE;")
                self._emit(f"writes[{i}].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;")
                self._emit(f"writes[{i}].dstSet = desc_set;")
                self._emit(f"writes[{i}].dstBinding = {i};")
                self._emit(f"writes[{i}].descriptorCount = 1;")
                self._emit(f"writes[{i}].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;")
                self._emit(f"writes[{i}].pBufferInfo = &buffer_infos[{i}];")
            self._emit(f"vkUpdateDescriptorSets(context.get_device(), {len(buffer_params)}, writes, 0, nullptr);")
        if all_pc_fields:
            self._emit(
                f"vkCmdPushConstants(command_buffer, kernel_.pipeline_layout(), "
                f"VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);"
            )
        if buffer_params:
            self._emit(
                "vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, "
                "kernel_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);"
            )
        self._emit("vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel_.pipeline());")

        self._emit(f"vkCmdDispatch(command_buffer, {x_dim}, {y_dim}, {z_dim});")
        self._emit("context.submit_and_wait();")

        self._emit("}")
        self._pop()
        self._emit("")
        self._emit("private:")
        self._push()
        self._emit("VulkanComputeKernel kernel_;")
        self._pop()
        self._emit("};")

    def visit_program(self, node: ast.Program) -> str:
        """Generate the complete C++ stub header for RLLM-style kernels."""
        self._initialize_program_emit_state()
        self._configure_kernel_identity(node)

        buffer_params, scalar_params, has_matrix_params = self._classify_program_params(node)
        is_triangular = len(node.triangular_bounds_raw) >= 2
        has_2d = node.space_dim >= 2 and len(node.loop_vars) >= 2
        has_3d = node.space_dim >= 3 and len(node.loop_vars) >= 3

        self._space_dim = node.space_dim
        if is_triangular:
            self._kernel_type = "Triangular"
        elif has_matrix_params:
            self._kernel_type = "Matrix"
        else:
            self._kernel_type = "Vector"

        wg_x, wg_y, wg_z = self._resolve_workgroup_sizes(node)
        all_pc_fields = self._build_push_constant_fields(
            scalar_params=scalar_params,
            has_2d=has_2d,
            has_3d=has_3d,
            is_triangular=is_triangular,
            node=node,
        )
        method_params = self._build_method_params(
            has_2d=has_2d,
            has_3d=has_3d,
            buffer_params=buffer_params,
            all_pc_fields=all_pc_fields,
        )

        self._emit_header_prelude(node)
        self._emit_buffer_structs(buffer_params)
        self._emit_push_constant_struct(all_pc_fields)

        class_name = self._class_name(self._kernel_name)
        self._emit_kernel_class(
            node=node,
            class_name=class_name,
            method_params=method_params,
            all_pc_fields=all_pc_fields,
            buffer_params=buffer_params,
            has_2d=has_2d,
            has_3d=has_3d,
            wg_x=wg_x,
            wg_y=wg_y,
            wg_z=wg_z,
        )

        self._emit("}")
        return self.result()

    # ── additional type visitors for grammar updates ───────────────────

    def visit_flexible_rows_cols_matrix(self, node: ast.FlexibleRowsColsMatrix) -> str:
        return ""  # handled in visit_program

    def visit_ternary_expr(self, node: ast.TernaryExpr) -> str:
        cond = self._visit_expr_child(node.condition)
        true_val = self._visit_expr_child(node.true_expr)
        false_val = self._visit_expr_child(node.false_expr)
        return f"({cond} ? {true_val} : {false_val})"

    def visit_unary_minus_expr(self, node: ast.UnaryMinusExpr) -> str:
        operand = self._visit_expr_child(node.operand)
        return f"-{operand}"

    # ── statement visitors for grammar updates ─────────────────────────

    def visit_return_statement(self, node: ast.ReturnStatement) -> str:
        return "return;"

    def visit_atomic_op(self, node: ast.AtomicOp) -> str:
        lhs = self._visit_expr_child(node.lhs)
        rhs = self._visit_expr_child(node.rhs)
        return f"{node.op}({lhs}, {rhs});"

    def visit_if(self, node: ast.If) -> str:
        cond = (
            node.condition.accept(self)
            if isinstance(node.condition, ast.Expression)
            else str(node.condition)
        )
        body = "\n".join(s.accept(self) for s in node.body_stmts)
        result = f"if ({cond}) {{\n{body}\n}}"
        if node.else_stmts:
            else_body = "\n".join(s.accept(self) for s in node.else_stmts)
            result += f" else {{\n{else_body}\n}}"
        return result

    def visit_declaration(self, node: ast.Declaration) -> str:
        type_str = node.var_type.accept(self) if hasattr(node.var_type, "accept") else ""
        name = node.name
        init = f" = {node.init_expr.accept(self)}" if node.init_expr else ""
        const_prefix = "const " if node.is_const else ""
        return f"{const_prefix}{type_str} {name}{init};"

    def visit_assignment(self, node: ast.Assignment) -> str:
        lvalue = node.lvalue.accept(self) if hasattr(node.lvalue, "accept") else str(node.lvalue)
        rvalue = node.rvalue.accept(self) if node.rvalue and hasattr(node.rvalue, "accept") else ""
        op = node.assign_op or "="
        return f"{lvalue} {op} {rvalue};"

    def visit_overflow_check(self, node: ast.OverflowCheck) -> str:
        lvalue = node.lvalue.accept(self) if hasattr(node.lvalue, "accept") else str(node.lvalue)
        operand = node.operand
        return f"OVERFLOW_CHECK_ADD({lvalue}, {operand});"

    def visit_shared_decl(self, node: ast.SharedDecl) -> str:
        type_str = node.var_type.accept(self) if hasattr(node.var_type, "accept") else ""
        name = node.name
        init = f" = {node.init_expr.accept(self)}" if node.init_expr else ""
        const_prefix = "const " if node.is_const else ""
        return f"shared {const_prefix}{type_str} {name}{init};"

    def visit_limit_expr(self, node: ast.LimitExpr) -> str:
        max_v = self._visit_expr_child(node.max_val)
        if hasattr(node, 'end') and getattr(node.end, 'value', None) is not None:
            start = self._visit_expr_child(node.start) if hasattr(node.start, "accept") else ""
            end = self._visit_expr_child(node.end) if hasattr(node.end, "accept") else ""
            return f"limit<{max_v}>({start}, {end})"
        end = self._visit_expr_child(node.end) if hasattr(node.end, "accept") else str(getattr(node.end, "value", ""))
        return f"limit<{max_v}>({end})"
