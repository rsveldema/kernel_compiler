from unittest_offload_parfor_helper import parse_kernel

from codegen.kast.statement import Declaration
from codegen.visitors.resolve_array_indices import resolve_array_indices
from codegen.visitors.tree_rewriter import TreeRewriter
from codegen.visitors.vulkan_kernel_visitor import VulkanKernelVisitor


def assert_parse_kernel(kernel_filename):
    parse_kernel(kernel_filename)


def assert_tree_rewriter_applies(kernel_filename, reduction_chunks):
    program = parse_kernel(kernel_filename)

    program = program.accept(TreeRewriter({}))

    assert program.reduction_chunks == reduction_chunks
    assert any(
        isinstance(stmt, Declaration) and stmt.name == "rllm_reduction_chunk_size"
        for stmt in program.body_stmts
    )


def transformed_shader(kernel_filename):
    program = parse_kernel(kernel_filename)

    program = program.accept(TreeRewriter({}))
    program = resolve_array_indices(program)
    return program.accept(VulkanKernelVisitor())


def assert_single_reduction_shader(shader, reduction_chunks, orig_loop_bound=None):
    """Check GLSL for a single-sum z-partitioning transformation."""
    assert "const int rllm_reduction_chunk_size" in shader
    assert f"layout(local_size_x = 8, local_size_y = 8, local_size_z = {reduction_chunks}) in;" in shader
    assert "shared float rllm_reduction_result[1024]" in shader
    assert "gl_LocalInvocationID.z" in shader
    assert "gl_LocalInvocationID.z == 0" in shader
    assert "const int rllm_reduction_accumulate_slot" in shader
    assert shader.count("barrier();") == 3
    assert "for (int l_idx = block_start_i; l_idx < block_end_i; ++l_idx)" in shader
    if orig_loop_bound is not None:
        assert f"for (int l_idx = 0; l_idx < {orig_loop_bound}; ++l_idx)" not in shader


def assert_common_reduction_shader(shader, reduction_chunks):
    assert "const int rllm_reduction_chunk_size" in shader
    assert f"layout(local_size_x = 8, local_size_y = 8, local_size_z = {reduction_chunks}) in;" in shader
    assert "shared float rllm_reduction_resultA[1024]" in shader
    assert "shared float rllm_reduction_resultB[1024]" in shader
    assert "gl_LocalInvocationID.z" in shader
    assert "gl_LocalInvocationID.z == 0" in shader
    assert "const int rllm_reduction_accumulate_slot" in shader
    assert shader.count("barrier();") == 3
    assert "for (int l_idx = 0; l_idx < 1024; ++l_idx)" not in shader
    assert "for (int l_idx = block_start_i; l_idx < block_end_i; ++l_idx)" in shader
