from testcommon import assert_parse_kernel, assert_tree_rewriter_applies, transformed_shader


KERNEL_FILENAME = "offload_parfor_TransformerBlock_95.kernel"


def test_parse():
    assert_parse_kernel(KERNEL_FILENAME)


def test_tree_rewriter_applies_optimize_TransformerBlock95_tkernel():
    assert_tree_rewriter_applies(KERNEL_FILENAME, reduction_chunks=8)


def test_tree_rewriter_changes_TransformerBlock95_glsl():
    shader = transformed_shader(KERNEL_FILENAME)

    assert "layout(local_size_x = 16, local_size_y = 1, local_size_z = 8) in;" in shader
    assert "shared float rllm_sq_reduction[128]" in shader
    assert "shared float rllm_inv_arr[16]" in shader
    assert "shared float rllm_dot_reduction[128]" in shader
    assert "shared float rllm_dot_scaled_arr[16]" in shader
    assert shader.count("barrier();") == 4
    assert "lz == 0" in shader
    # Original single-threaded loops are gone
    assert "for (int i = 0; i < 1024; ++i)" not in shader
