"""Regression tests for current kernel-codegen behavior."""

from codegen.parser import parse
from codegen.visitors.resolve_array_indices import resolve_array_indices
from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor
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


def test_cpp_stub_reuses_queue_descriptor_arena():
    program = parse(
        """
PROGRAM("descriptor.cc:2")

OFFLOAD_PARFOR_1D_PARAM(queue, i, limit<16>(), (values))

PARAMETERS
        fixed_size_vector<float, 16>& values

BEGIN
        values[i] = 1.f;

END_PROGRAM
"""
    )
    program = resolve_array_indices(program)

    stub = program.accept(VulkanCppStubVisitor())

    allocation = "queue.allocate_dispatch_descriptor_set(desc_layout, 1)"
    assert allocation in stub
    assert "vkCreateDescriptorPool" not in stub
    assert "queue.defer_descriptor_pool" not in stub
    assert stub.index(allocation) < stub.index("queue.allocate_command_buffer()")
