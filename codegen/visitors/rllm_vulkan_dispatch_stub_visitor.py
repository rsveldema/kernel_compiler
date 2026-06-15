"""RLLM dispatch stub generator for compiled Vulkan kernels."""

from __future__ import annotations

import re
from pathlib import Path

from .. import kast as ast
from .visitor import Visitor


def _sanitize_component(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", text)


def _class_name(kernel_name: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in kernel_name.split("_") if part) + "Kernel"


_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")
_CPP_RESERVED = {"None", "static_cast"}


def _is_push_identifier(value: str | None) -> bool:
    return bool(value and _IDENT_RE.match(value) and value not in _CPP_RESERVED)


class RllmVulkanDispatchStubVisitor(Visitor):
    def __init__(self, spv_path: str):
        self._spv_path = spv_path

    def _is_buffer_type(self, ty) -> bool:
        return isinstance(
            ty,
            (
                ast.FlexibleRowsMatrix,
                ast.FlexibleSizeMatrix,
                ast.FlexibleRowsColsMatrix,
                ast.FixedSizeMatrix,
                ast.FlexibleRowsColsLevelsMatrix,
                ast.FixedSizeLevelsRowsColsMatrix,
                ast.FixedSizeObjVectorMatrix,
                ast.FixedSizeVector,
            ),
        )

    def _level_count(self, ty) -> int:
        if isinstance(ty, ast.FixedSizeObjVectorMatrix) and isinstance(ty.level_expr, ast.Number):
            return int(ty.level_expr.value)
        return 1

    def _ctype(self, ty) -> str:
        return "float" if isinstance(ty, ast.Float) else "int32_t"

    def visit_program(self, node: ast.Program) -> str:
        header = node.header.strip('"') if node.header else "kernel:0"
        rel_path, _, line = header.partition(":")
        stem = _sanitize_component(Path(rel_path).stem or "kernel")
        line_part = _sanitize_component(line or "0")
        symbol = f"__rllm_vulkan_kernel_{stem}_L{line_part}"

        src_stem = (
            Path(getattr(node, "_source_filename", "")).name.rsplit(".", 1)[0].replace("-", "_")
            if getattr(node, "_source_filename", "")
            else "kernel"
        )
        stub_stem = Path(self._spv_path).name.rsplit(".", 1)[0]
        basename = stub_stem.rsplit(".", 1)[0] if "." in stub_stem else stub_stem
        kernel_name = f"{src_stem}_{basename}" if basename != "kernel" else src_stem
        namespace_name = f"rllm_{src_stem}"
        class_name = _class_name(kernel_name)
        push_name = f"{kernel_name}_PushConstants"
        stub_header = Path(self._spv_path).with_suffix(".h").name

        params = [
            p for p in node.params
            if isinstance(p, ast.Declaration) and p.name
        ]
        template_params = [f"typename P{i}" for i, _ in enumerate(params)]
        function_params = [f"P{i}&& {p.name}" for i, p in enumerate(params)]

        if node.space_dim >= 3:
            dim_values = [
                "static_cast<uint32_t>(range.outer_size())",
                "static_cast<uint32_t>(range.middle_size())",
                "static_cast<uint32_t>(range.inner_size())",
            ]
        elif node.space_dim >= 2:
            dim_values = [
                "static_cast<uint32_t>(range.outer_size())",
                "static_cast<uint32_t>(range.inner_size())",
            ]
        else:
            dim_values = ["static_cast<uint32_t>(range.inner_size())"]

        push_fields = [("rllm_bound_x", "int32_t", dim_values[0])]
        if node.space_dim >= 2:
            push_fields.append(("rllm_bound_y", "int32_t", dim_values[1]))
        if node.space_dim >= 3:
            push_fields.append(("rllm_bound_z", "int32_t", dim_values[2]))

        seen = {name for name, _, _ in push_fields}
        for param in params:
            if not self._is_buffer_type(param.var_type) and param.name not in seen:
                seen.add(param.name)
                push_fields.append((param.name, self._ctype(param.var_type), param.name))

        for tb in getattr(node, "triangular_bounds_raw", []) or []:
            if _is_push_identifier(tb) and not tb.lstrip("-").isdigit() and tb not in seen:
                seen.add(tb)
                push_fields.append((tb, "int32_t", tb))

        # When parallelized, add workgroup count and offset fields to push constants.
        _wg_count = getattr(node, "workgroup_count", 1)
        _parallelized = getattr(node, "parallelized", False)
        if _parallelized and _wg_count > 1:
            push_fields.insert(0, ("rllm_wg_count", "int32_t", str(_wg_count)))

        dispatch_args = ["rllm::vulkan_runtime::context()", *dim_values]
        mutable_buffers = []
        for param in params:
            if not self._is_buffer_type(param.var_type):
                continue
            if isinstance(param.var_type, ast.FixedSizeObjVectorMatrix):
                for level in range(self._level_count(param.var_type)):
                    dispatch_args.append(f"rllm::vulkan_runtime::device_buffer({param.name}, {level})")
                    if not getattr(param, "is_const", False):
                        mutable_buffers.append((param.name, level))
            else:
                dispatch_args.append(f"rllm::vulkan_runtime::device_buffer({param.name})")
                if not getattr(param, "is_const", False):
                    mutable_buffers.append((param.name, None))

        push_lines = []
        for name, ctype, value in push_fields:
            push_lines.append(f"        .{name} = static_cast<{ctype}>({value}),")
        dispatch_args.append("push_constants")

        template_prefix = ""
        if template_params:
            template_prefix = "template <typename Range, " + ", ".join(template_params) + ", typename... Ignored>" + chr(10)
        else:
            template_prefix = "template <typename Range, typename... Ignored>" + chr(10)

        joined_function_params = ", ".join(["Range&& range", *function_params, "Ignored&&... ignored_args"])
        joined_dispatch_args = ", ".join(dispatch_args)

        mark_lines = []
        for name, level in mutable_buffers:
            if level is None:
                mark_lines.append(f"    rllm::vulkan_runtime::mark_device_latest({name});")
            else:
                mark_lines.append(f"    rllm::vulkan_runtime::mark_device_latest({name}, {level});")

        # Parallelized kernels are still dispatched once; the GPU workgroup grid
        # and push constants control partitioning inside the shader.
        push_body = chr(10).join(push_lines)
        mark_body = chr(10).join(mark_lines) + (chr(10) if mark_lines else "")

        body_parts = []
        body_parts.append("    static_cast<void>(sizeof...(ignored_args));")
        body_parts.append("    std::lock_guard<std::recursive_mutex> vulkan_lock(rllm::vulkan_runtime::mutex());")
        spv_esc = self._spv_path.replace(chr(92), chr(92)+chr(92))
        body_parts.append('    static ' + namespace_name + "::" + class_name + ' kernel(rllm::vulkan_runtime::session(), std::string(RLLM_VULKAN_KERNEL_ROOT) + "/' + spv_esc + '");')
        body_parts.append("    ComputeKernelRegistry::ScopedActiveKernel active_kernel(kernel);")
        body_parts.append("    const " + namespace_name + "::" + push_name + " push_constants{")
        body_parts.append(push_body)
        body_parts.append("    };")
        body_parts.append("    kernel.dispatch(" + joined_dispatch_args + ");")
        if mark_body:
            body_parts.append(mark_body)
        loop_body = chr(10).join(body_parts)

        result = ""
        result += "#pragma once" + chr(10)
        result += "#include <cstdint>" + chr(10)
        result += "#include <string>" + chr(10)
        result += "#include <utility>" + chr(10)
        result += "#include <rllm_vulkan_runtime.hpp>" + chr(10)
        result += '#include "' + stub_header + '"' + chr(10)
        result += chr(10)
        result += "namespace rllm::vulkan::generated {" + chr(10)
        result += chr(10)
        result += template_prefix
        result += "inline void " + symbol + "(" + joined_function_params + ")" + chr(10)
        result += "{" + chr(10)
        result += loop_body
        result += "}" + chr(10)
        result += chr(10)
        result += "} // namespace rllm::vulkan::generated" + chr(10)
        return result
