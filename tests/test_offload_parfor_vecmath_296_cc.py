from codegen.kast.statement import SharedDecl
from testcommon import assert_parse_kernel, transformed_shader
from unittest_offload_parfor_helper import parse_kernel
from codegen.visitors.tree_rewriter import TreeRewriter


KERNEL_FILENAME = "offload_parfor_vecmath_296.kernel"


def test_parse():
    assert_parse_kernel(KERNEL_FILENAME)


def test_tree_rewriter_applies_optimize_vecmath296_tkernel():
    program = parse_kernel(KERNEL_FILENAME)

    program = program.accept(TreeRewriter({}))

    assert getattr(program, "tree_transformed", False)
    assert program.reduction_chunks == 8
    assert any(
        isinstance(stmt, SharedDecl) and stmt.name == "rllm_reduction_result"
        for stmt in program.body_stmts
    )


def test_tree_rewriter_changes_vecmath296_glsl():
    shader = transformed_shader(KERNEL_FILENAME)

    assert "layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;" in shader
    assert "shared float rllm_reduction_result[1024]" in shader
    assert "rllm_reduction_slot_base" in shader
    assert "rllm_reduction_chunk_size" in shader
    assert "gl_LocalInvocationID.z == 0" in shader
    assert "barrier();" in shader
    assert "for (int l_idx = 0; l_idx < 1024; ++l_idx)" not in shader

