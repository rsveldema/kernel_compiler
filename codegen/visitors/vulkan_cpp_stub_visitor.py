"""C++ stub generator for Vulkan compute kernels.

Generates compilable C++ header with:
- Struct wrappers matching the SSBO layout (RllmBuffer_<name>)
- Push constants struct for scalar params and triangular bounds
- Descriptor set struct for buffer bindings
- A dispatch function matching the kernel's parameters
"""

from typing import Dict, List, Tuple
from .. import ast
from .visitor import Visitor


class VulkanCppStubVisitor(Visitor):
    """Transforms the parsed AST into a C++ stub for calling the Vulkan kernel."""

    def __init__(self):
        self._lines: List[str] = []
        self._indent_level: int = 0
        self._buffer_structs: Dict[str, str] = {}  # name -> struct_name
        self._kernel_name: str = ""

    # ── helpers ────────────────────────────────────────────────────────

    def _emit(self, line: str = "") -> None:
        self._lines.append("    " * self._indent_level + line)

    def _push(self) -> None:
        self._indent_level += 1

    def _pop(self) -> None:
        self._indent_level -= 1

    def result(self) -> str:
        return "\n".join(self._lines) + "\n"

    def _cpp_type_str(self, ty) -> str:
        if isinstance(ty, ast.Int):
            return "int32_t"
        if isinstance(ty, ast.Float):
            return "float"
        return "unknown_type"

    # ── expression visitors ────────────────────────────────────────────

    def visit_type(self, node: ast.Type) -> str:
        return self._cpp_type_str(node)

    def visit_int(self, node: ast.Int) -> str:
        return "int32_t"

    def visit_float(self, node: ast.Float) -> str:
        return "float"

    def visit_fixed_size_vector(self, node: ast.FixedSizeVector) -> str:
        elem = self._cpp_type_str(node.elem_type) if node.elem_type else "float"
        return f"{elem}"

    def visit_flexible_rows_matrix(self, node: ast.FlexibleRowsMatrix) -> str:
        return ""  # handled in visit_program

    def visit_fixed_size_matrix(self, node: ast.FixedSizeMatrix) -> str:
        return ""

    def visit_expression(self, node: ast.Expression) -> str:
        return node.accept(self)

    def visit_number(self, node: ast.Number) -> str:
        s = str(node.value)
        if getattr(node, "unsigned", False):
            if abs(node.value) > 4294967295:  # UINT_MAX — need uint64
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

    def visit_limit_expr(self, node: ast.LimitExpr) -> str:
        max_v = self._visit_expr_child(node.max_val)
        body_v = self._visit_expr_child(node.body)
        return f"limit<{max_v}>({body_v})"

    def visit_cast_expr(self, node: ast.CastExpr) -> str:
        cast_type = "int32_t"
        operand = self._visit_expr_child(node.operand)
        return f"static_cast<{cast_type}>({operand})"

    def visit_negation_expr(self, node: ast.NegationExpr) -> str:
        operand = self._visit_expr_child(node.operand)
        return f"!{operand}"

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

    def visit_if(self, node: ast.If) -> str:
        raise NotImplementedError("visit_if")

    def visit_declaration(self, node: ast.Declaration) -> str:
        raise NotImplementedError("visit_declaration")

    def visit_assignment(self, node: ast.Assignment) -> str:
        raise NotImplementedError("visit_assignment")

    def visit_overflow_check(self, node: ast.OverflowCheck) -> str:
        raise NotImplementedError("visit_overflow_check")

    def visit_shared_decl(self, node: ast.SharedDecl) -> str:
        raise NotImplementedError("visit_shared_decl")

    # ── workgroup and program visitors ────────────────────────────────

    def _is_matrix_type(self, ty) -> bool:
        """Check if a type is a matrix or vector that maps to an SSBO."""
        return isinstance(
            ty,
            (
                ast.FlexibleRowsMatrix,
                ast.FixedSizeMatrix,
                ast.FlexibleRowsColsLevelsMatrix,
                ast.FixedSizeLevelsRowsColsMatrix,
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
        elif isinstance(ty, (ast.FlexibleRowsMatrix, ast.FixedSizeMatrix)):
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

    def visit_workgroup_properties(self, node: ast.WorkgroupProperties) -> dict:
        return {
            "x": self._to_str(node.x_expr) if node.x_expr else "8",
            "y": self._to_str(node.y_expr) if node.y_expr else "8",
            "z": self._to_str(node.z_expr) if node.z_expr else "1",
        }

    def visit_program(self, node: ast.Program) -> str:
        """Generate the complete C++ stub header for RLLM-style kernels."""
        self._lines = []
        self._indent_level = 0
        self._buffer_structs = {}
        self._kernel_name = ""

        # Determine kernel name from header
        basename = "kernel"
        if node.header:
            parts = node.header.replace('"', "").split("/")
            basename = parts[-1].rsplit(".", 1)[0] if "." in parts[-1] else parts[-1]
        # Use source filename stem + header basename to ensure unique dispatch names
        src_stem = getattr(node, "_source_filename", "").rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("-", "_") if hasattr(node, "_source_filename") and node._source_filename else "kernel"
        self._kernel_name = f"{src_stem}_{basename}" if basename != "kernel" else src_stem

        # Classify params
        buffer_params: List[Tuple[ast.Declaration, str]] = []
        scalar_params: List[ast.Declaration] = []

        is_triangular = len(getattr(node, "triangular_bounds_raw", [])) >= 2

        for param in node.params:
            if not isinstance(param, ast.Declaration):
                continue

            vt = param.var_type
            if self._is_matrix_type(vt):
                sname = f"RllmBuffer_{param.name}"
                self._buffer_structs[param.name] = sname
                buffer_params.append((param, sname))
            else:
                scalar_params.append(param)

        # Collect dispatch dimensions
        has_2d = node.space_dim >= 2 and len(node.loop_vars) >= 2
        has_3d = node.space_dim >= 3 and len(node.loop_vars) >= 3
        rows_param_name = node.loop_vars[0] if node.loop_vars else "dispatch_rows"
        cols_param_name = node.loop_vars[1] if has_2d else None
        levels_param_name = node.loop_vars[2] if has_3d else None

        # Workgroup sizes
        wg_x, wg_y, wg_z = "1", "1", "1"
        for wg in node.workgroups:
            if isinstance(wg, ast.WorkgroupProperties):
                x_val = self._to_str(wg.x_expr) if wg.x_expr else "1"
                y_val = self._to_str(wg.y_expr) if wg.y_expr else "1"
                z_val = self._to_str(wg.z_expr) if wg.z_expr else "1"
                if x_val.isdigit():
                    wg_x = x_val
                if y_val.isdigit():
                    wg_y = y_val
                if z_val.isdigit():
                    wg_z = z_val

        # ── Emit C++ header ────────────────────────────────────────────

        func_params: List[str] = []
        func_params.append("    VkDevice device")
        func_params.append("    VkPipelineLayout pipeline_layout")
        func_params.append("    VkDescriptorSetLayout descriptor_set_layout")
        func_params.append("    VkPipeline pipeline")
        func_params.append("    VkCommandBuffer command_buffer")
        func_params.append("    VkDescriptorSet _desc_set")

        if has_3d:
            func_params.append("    uint32_t dispatch_rows")
            func_params.append("    uint32_t dispatch_cols")
            func_params.append("    uint32_t dispatch_levels")
        elif has_2d:
            func_params.append("    uint32_t dispatch_rows")
            func_params.append("    uint32_t dispatch_cols")
        else:
            func_params.append("    uint32_t dispatch_rows")

        # Add buffer params as Vulkan buffer handles. The generated structs
        # above document SSBO layout, but callers should pass actual buffers.
        for param, sname in buffer_params:
            func_params.append(f"    VkBuffer {param.name}")

        # Build push constant field names (deduplicated)
        pc_field_names = set()
        all_pc_fields = []

        for param in scalar_params:
            if param.name not in pc_field_names:
                pc_field_names.add(param.name)
                all_pc_fields.append(param)

        if is_triangular:
            for tb in getattr(node, "triangular_bounds_raw", []):
                if tb not in pc_field_names and not tb.lstrip("-").isdigit():
                    pc_field_names.add(tb)
                    # Create a synthetic field for triangular bounds
                    all_pc_fields.append(
                        type(
                            "",
                            (),
                            {"name": tb, "is_const": True, "var_type": ast.Int()},
                        )()
                    )

        if is_triangular or scalar_params:
            func_params.append(
                f"    const {self._kernel_name}_PushConstants& push_constants"
            )

        # Emit includes
        self._emit("")
        hdr_text = node.header.strip('"') if node.header else "unknown"
        self._emit(f"// ── Kernel dispatch stub: {hdr_text} ─────────────")
        self._emit("#include <cstdint>")
        self._emit("#include <vulkan/vulkan_core.h>")
        self._emit("")

        # Buffer structs (matching SSBO layout)
        for param, sname in buffer_params:
            vt = param.var_type
            size_str = self._compute_matrix_size(vt)

            self._emit(f"struct {sname} {{")
            self._push()
            self._emit(f"    float data[{size_str}];")
            self._pop()
            self._emit("};")
            self._emit("")

        # Push constants struct for scalar params and triangular bounds
        if all_pc_fields:
            pc_name = f"{self._kernel_name}_PushConstants"
            self._emit(f"struct {pc_name} {{")
            self._push()

            for field in all_pc_fields:
                ctype = "int32_t"  # All scalar params are int-sized
                is_const = "const " if getattr(field, "is_const", False) else ""
                self._emit(f"    {is_const}{ctype} {field.name};")

            self._pop()
            self._emit("};")
            self._emit("")

        # Dispatch function signature
        self._emit(f"inline void {self._kernel_name}(")
        for i, fp in enumerate(func_params):
            comma = "," if i < len(func_params) - 1 else ""
            self._emit(fp + comma)
        self._emit(") {")
        self._push()

        # Determine tile block size (set by perform_blocking when tiling is active)
        tile_bs = getattr(node, "tile_block_size", None)

        def _div_ceil(a, b):
            return "((" + a + ") + (" + str(b) + ") - 1) / (" + str(b) + ")"

        if has_3d:
            x_dim = f"(dispatch_rows + {wg_x} - 1) / {wg_x}"
            y_dim = f"(dispatch_cols + {wg_y} - 1) / {wg_y}"
            z_dim = f"(dispatch_levels + {wg_z} - 1) / {wg_z}"
        elif has_2d:
            x_dim = f"(dispatch_rows + {wg_x} - 1) / {wg_x}"
            y_dim = f"(dispatch_cols + {wg_y} - 1) / {wg_y}"
            z_dim = str(getattr(node, "reduction_chunks", 1))
        else:
            x_dim = f"(dispatch_rows + {wg_x} - 1) / {wg_x}"
            y_dim = "1"
            z_dim = str(getattr(node, "reduction_chunks", 1))

        self._emit("(void)descriptor_set_layout;")
        if buffer_params:
            self._emit(f"VkDescriptorBufferInfo buffer_infos[{len(buffer_params)}]{{}};")
            self._emit(f"VkWriteDescriptorSet writes[{len(buffer_params)}]{{}};")
            for i, (param, _sname) in enumerate(buffer_params):
                self._emit(f"buffer_infos[{i}].buffer = {param.name};")
                self._emit(f"buffer_infos[{i}].offset = 0;")
                self._emit(f"buffer_infos[{i}].range = VK_WHOLE_SIZE;")
                self._emit(f"writes[{i}].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;")
                self._emit(f"writes[{i}].dstSet = _desc_set;")
                self._emit(f"writes[{i}].dstBinding = {i};")
                self._emit(f"writes[{i}].descriptorCount = 1;")
                self._emit(f"writes[{i}].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;")
                self._emit(f"writes[{i}].pBufferInfo = &buffer_infos[{i}];")
            self._emit(f"vkUpdateDescriptorSets(device, {len(buffer_params)}, writes, 0, nullptr);")
        else:
            self._emit("(void)device;")
        if all_pc_fields:
            self._emit(
                f"vkCmdPushConstants(command_buffer, pipeline_layout, "
                f"VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);"
            )
        if buffer_params:
            self._emit(
                "vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, "
                "pipeline_layout, 0, 1, &_desc_set, 0, nullptr);"
            )
        self._emit("vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);")
        self._emit(f"vkCmdDispatch(command_buffer, {x_dim}, {y_dim}, {z_dim});")

        self._pop()
        self._emit("}")

        return self.result()
