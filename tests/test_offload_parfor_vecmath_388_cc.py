from testcommon import assert_parse_kernel, assert_tree_rewriter_applies, assert_single_reduction_shader, transformed_shader


KERNEL_FILENAME = "offload_parfor_vecmath_388.kernel"


def test_parse():
    assert_parse_kernel(KERNEL_FILENAME)


def test_tree_rewriter_applies_optimize_vecmath388_tkernel():
    assert_tree_rewriter_applies(KERNEL_FILENAME, reduction_chunks=8)


def test_tree_rewriter_changes_vecmath388_glsl():
    shader = transformed_shader(KERNEL_FILENAME)
    assert_single_reduction_shader(shader, reduction_chunks=8, orig_loop_bound=4096)
    # C[i,j] += sum (accumulate, not assign)
    assert "C[" in shader
