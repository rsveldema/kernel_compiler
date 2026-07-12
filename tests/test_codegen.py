"""Regression tests for current kernel-codegen behavior."""

from codegen.parser import parse
from codegen.visitors.resolve_array_indices import resolve_array_indices
from codegen.visitors.vulkan_kernel_visitor import VulkanKernelVisitor
from offloadize_common import _defined_macros_for_backend, _eval_preprocessor_expr


def test_translator_enables_guarded_attention_diagnostics():
    macros = _defined_macros_for_backend("vulkan")

    assert _eval_preprocessor_expr("DEBUG_ATTENTION_DIAGNOSTICS", macros)


def test_triangular_dynamic_bound_does_not_replace_fixed_inner_loop_bound():
    program = parse(
        """
PROGRAM("tri.cc:2")

OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(queue, i, j, limit<8192>(seq_len), (values, seq_len))

PARAMETERS
        fixed_size_matrix<float, 8192, 512>& values,
        int seq_len

BEGIN
        float sum = 0.f;
        for (const int d : limit<64>())
            sum += values[j, d];
        values[i, 0] = sum;

END_PROGRAM
"""
    )
    program = resolve_array_indices(program)

    shader = program.accept(VulkanKernelVisitor())

    assert "for (int d = 0; d < 64; ++d)" in shader
    assert "for (int d = 0; d < rllm_push.seq_len; ++d)" not in shader
