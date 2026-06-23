from testcommon import assert_parse_kernel, assert_tree_rewriter_applies, assert_single_reduction_shader, transformed_shader


KERNEL_FILENAME = "offload_parfor_vecmath_415.kernel"


def test_parse():
    assert_parse_kernel(KERNEL_FILENAME)


def test_tree_rewriter_applies_optimize_vecmath415_tkernel():
    assert_tree_rewriter_applies(KERNEL_FILENAME, reduction_chunks=8)


def test_tree_rewriter_changes_vecmath415_glsl():
    shader = transformed_shader(KERNEL_FILENAME)
    # 415 reduces three sums into one total per z-thread, so uses a single reduction array
    assert_single_reduction_shader(shader, reduction_chunks=8, orig_loop_bound=1024)
    assert "C[" in shader
