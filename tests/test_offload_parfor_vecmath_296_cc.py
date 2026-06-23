from codegen.kast.statement import Declaration
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
    assert program.tile_size_x == 16
    assert program.tile_size_y == 16
    assert program.tile_chunk_size == 64
    assert any(
        isinstance(stmt, Declaration) and stmt.name == "rllm_tile_chunk_size"
        for stmt in program.body_stmts
    )


def test_tree_rewriter_changes_vecmath296_glsl():
    shader = transformed_shader(KERNEL_FILENAME)

    assert "layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;" in shader
    assert "shared float rllm_A_tile[1024]" in shader
    assert "shared float rllm_B_tile[1024]" in shader
    assert "const int rllm_tile_chunk_size = 64;" in shader
    assert "const int rllm_tile_size_x = 16;" in shader
    assert "const int rllm_tile_size_y = 16;" in shader
    assert "const int rllm_tile_local_linear" in shader
    assert "const int rllm_tile_local_count = (rllm_tile_size_x * rllm_tile_size_y);" in shader
    assert "gl_WorkGroupID.x" in shader
    assert "gl_WorkGroupID.y" in shader
    assert "for (int rllm_chunk_start = 0; rllm_chunk_start < 1024;" in shader
    assert "rllm_chunk_start += 64;" in shader
    assert shader.count("for (int rllm_load_slot = rllm_tile_local_linear; rllm_load_slot < 1024;") == 2
    assert shader.count("rllm_load_slot += rllm_tile_local_count;") == 2
    assert "rllm_A_tile[rllm_load_slot]" in shader
    assert "rllm_B_tile[rllm_load_slot]" in shader
    assert "rllm_A_tile[rllm_load_slot] = 0.0;" in shader
    assert "rllm_B_tile[rllm_load_slot] = 0.0;" in shader
    assert "(rllm_load_k < 1024) && (rllm_load_i < rllm_push.rllm_bound_x)" in shader
    assert "(rllm_load_k < 1024) && (rllm_load_j < rllm_push.rllm_bound_y)" in shader
    assert "if (i < rllm_push.rllm_bound_x)" in shader
    assert "if (j < rllm_push.rllm_bound_y)" in shader
    assert "rllm_A_tile[rllm_A_slot] * rllm_B_tile[rllm_B_slot]" in shader
    assert shader.count("barrier();") == 2
    assert "return;" not in shader
    assert "for (int l_idx = 0; l_idx < 1024; ++l_idx)" not in shader

