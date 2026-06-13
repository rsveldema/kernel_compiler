"""RLLM dispatch stub generator for compiled Vulkan kernels."""

from __future__ import annotations

import re
from pathlib import Path

from .. import kast as ast
from .visitor import Visitor


def _sanitize_component(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", text)


class RllmVulkanDispatchStubVisitor(Visitor):
    def __init__(self, spv_path: str):
        self._spv_path = spv_path

    def visit_program(self, node: ast.Program) -> str:
        header = node.header.strip('"') if node.header else "kernel:0"
        rel_path, _, line = header.partition(":")
        stem = _sanitize_component(Path(rel_path).stem or "kernel")
        line_part = _sanitize_component(line or "0")
        symbol = f"__rllm_vulkan_kernel_{stem}_L{line_part}"
        launch = "launch_3d_named" if node.space_dim >= 3 else ("launch_2d_named" if node.space_dim >= 2 else "launch_1d_named")
        param_names = [
            p.name for p in node.params
            if isinstance(p, ast.Declaration) and p.name
        ]
        bound_names = ["rllm_bound_x"]
        bound_values = ["static_cast<uint32_t>(range.inner_size())"]
        if node.space_dim >= 2:
            bound_names.append("rllm_bound_y")
            bound_values.append("static_cast<uint32_t>(range.outer_size())")
        if node.space_dim >= 3:
            bound_names.append("rllm_bound_z")
            bound_values.append("static_cast<uint32_t>(range.z_size())")
        joined_names = ", ".join(f'"{name}"' for name in bound_names + param_names)
        typed_bounds = ", ".join("uint32_t" for _ in bound_names)
        if typed_bounds:
            typed_bounds += ", "
        bound_args = ", ".join(bound_values)
        if bound_args:
            bound_args += ", "

        return (
            "#pragma once\n"
            "#include <cstdint>\n"
            "#include <utility>\n"
            "#include <vulkan_kernel_calls.hpp>\n"
            "\n"
            "namespace rllm::vulkan::generated {\n"
            "\n"
            "template <typename Range, typename... Args>\n"
            f"inline void {symbol}(Range&& range, Args&&... args)\n"
            "{\n"
            f"    static rllm::vulkan::ComputeKernel<{typed_bounds}Args...> kernel(VulkanMemorySpace::get_instance(), \"{header}\", \"{self._spv_path}\");\n"
            f"    kernel.{launch}(std::forward<Range>(range), {{{joined_names}}}, {bound_args}std::forward<Args>(args)...);\n"
            "}\n"
            "\n"
            "} // namespace rllm::vulkan::generated\n"
        )
