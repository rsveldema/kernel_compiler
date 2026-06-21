from testcommon import (
    assert_common_reduction_shader,
    assert_parse_kernel,
    assert_tree_rewriter_applies,
    transformed_shader,
)


KERNEL_FILENAME = "offload_parfor_vecmath_225.kernel"
REDUCTION_CHUNKS = 8


def test_parse():
    assert_parse_kernel(KERNEL_FILENAME)


def test_tree_rewriter_applies_optimize_vecmath225_tkernel():
    assert_tree_rewriter_applies(KERNEL_FILENAME, REDUCTION_CHUNKS)


def test_tree_rewriter_changes_vecmath225_glsl():
    shader = transformed_shader(KERNEL_FILENAME)

    assert_common_reduction_shader(shader, REDUCTION_CHUNKS)
    assert "shared float rllm_reduction_resultC[1024]" in shader
