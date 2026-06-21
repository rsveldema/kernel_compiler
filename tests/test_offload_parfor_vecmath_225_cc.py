from unittest_offload_parfor_helper import parse_kernel

from codegen.kast.statement import Declaration, SharedDecl
from codegen.visitors.resolve_array_indices import resolve_array_indices
from codegen.visitors.tree_rewriter import TreeRewriter
from codegen.visitors.vulkan_kernel_visitor import VulkanKernelVisitor


def test_parse():
    parse_kernel("offload_parfor_vecmath_225.kernel")


def test_tree_rewriter_applies_optimize_vecmath225_tkernel():
    program = parse_kernel("offload_parfor_vecmath_225.kernel")

    program = program.accept(TreeRewriter({}))

    assert program.reduction_chunks == 16
    assert any(
        isinstance(stmt, Declaration) and stmt.name == "rllm_reduction_chunk_size"
        for stmt in program.body_stmts
    )
    assert any(
        isinstance(stmt, SharedDecl) and stmt.name == "rllm_reduction_resultA"
        for stmt in program.body_stmts
    )


def test_tree_rewriter_changes_vecmath225_glsl():
    program = parse_kernel("offload_parfor_vecmath_225.kernel")

    program = program.accept(TreeRewriter({}))
    program = resolve_array_indices(program)
    shader = program.accept(VulkanKernelVisitor())

    assert "const int rllm_reduction_chunk_size" in shader
    assert "layout(local_size_x = 8, local_size_y = 8, local_size_z = 16) in;" in shader
    assert "shared float rllm_reduction_resultA" in shader
    assert "gl_LocalInvocationID.z" in shader
    assert "gl_WorkGroupID.z" not in shader
    assert shader.count("barrier();") == 2
    assert "for (int l_idx = block_start_i; l_idx < block_end_i; ++l_idx)" in shader
    assert "for (int l_idx = 0; l_idx < 1024; ++l_idx)" not in shader
