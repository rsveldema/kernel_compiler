from testcommon import assert_parse_kernel, assert_tree_rewriter_applies, transformed_shader


KERNEL_FILENAME = "offload_parfor_TransformerBlock_395.kernel"


def test_parse():
    assert_parse_kernel(KERNEL_FILENAME)


def test_tree_rewriter_applies_optimize_TransformerBlock395_tkernel():
    assert_tree_rewriter_applies(KERNEL_FILENAME, reduction_chunks=8)


def test_tree_rewriter_changes_TransformerBlock395_glsl():
    shader = transformed_shader(KERNEL_FILENAME)

    assert "layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;" in shader
    assert "shared float rllm_reduction_result[1024]" in shader
    assert "gl_LocalInvocationID.z == 0" in shader
    assert shader.count("barrier();") >= 1
    assert "rllm_reduction_result" in shader
    # Original single-threaded loop is gone
    assert "for (int d = 0; d < " not in shader
