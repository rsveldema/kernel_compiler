"""Regression tests for current kernel-codegen behavior."""

from codegen.parser import parse
from codegen.visitors.resolve_array_indices import resolve_array_indices
from codegen.visitors.hip_cpp_stub_visitor import HipCppStubVisitor
from codegen.visitors.hip_kernel_visitor import HipKernelVisitor
from codegen.blas import detect_safe_gemm
from offloadize_common import _defined_macros_for_backend, _eval_preprocessor_expr


def test_translator_enables_guarded_attention_diagnostics():
    macros = _defined_macros_for_backend("hip")

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

    shader = program.accept(HipKernelVisitor())

    assert "for (int d = 0; d < 64; ++d)" in shader
    assert "for (int d = 0; d < rllm_push.seq_len; ++d)" not in shader


def test_cpp_stub_launches_hip_module_with_buffer_arguments():
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

    stub = program.accept(HipCppStubVisitor())

    assert "void* buffer_0 = values.get();" in stub
    assert "kernel_.launch(queue" in stub
    assert "VkDescriptorSet" not in stub


def test_detects_only_exact_unclamped_at_b_accumulation():
    source = """
PROGRAM("vecmath.cc:1")
OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C, k_count))
PARAMETERS
    const flexible_rows_matrix<float, 16, 8>& A,
    const flexible_rows_matrix<float, 16, 32>& B,
    fixed_size_matrix<float, 8, 32>& C,
    int k_count
BEGIN
    const int i = int(gl_GlobalInvocationID.x);
    if (i >= rllm_bound_x) return;
    const int j = int(gl_GlobalInvocationID.y);
    if (j >= rllm_bound_y) return;
    float sum = 0.f;
    for (int l_idx = 0; l_idx < k_count; ++l_idx) {
        const int k = int(l_idx);
        const float term = (A[k, i] * B[k, j]);
        OVERFLOW_CHECK_ADD(sum, term);
        sum += term;
    }
    C[i, j] += sum;
END_PROGRAM
"""
    program = parse(source)
    gemm = detect_safe_gemm(program)
    assert gemm is not None
    assert (gemm.a, gemm.b, gemm.c, gemm.k) == ("A", "B", "C", "k_count")

    program.blas = gemm
    stub = resolve_array_indices(program).accept(HipCppStubVisitor())
    assert "rllm::hip_blas::gemm_at_b_acc" in stub
    assert "HipComputeKernel" not in stub

    # The same contraction with overwrite semantics uses the beta=0 path.
    overwrite_program = parse(source)
    overwrite_program.body_stmts[-1].assign_op = "="
    overwrite = detect_safe_gemm(overwrite_program)
    assert overwrite is not None
    assert not overwrite.accumulate
    assert overwrite.transpose_a
    overwrite_program.blas = overwrite
    overwrite_stub = resolve_array_indices(overwrite_program).accept(HipCppStubVisitor())
    assert "rllm::hip_blas::gemm_at_b_float" in overwrite_stub
    assert "beta=0;input=float32" in overwrite_stub


def test_detects_half_abt_projection_with_float_accumulation():
    program = parse(
        """
PROGRAM("vecmath.cc:2")
OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C))
PARAMETERS
    const flexible_rows_matrix<float16, 16, 8>& A,
    const fixed_size_matrix<float16, 32, 8>& B,
    flexible_rows_matrix<float, 16, 32>& C
BEGIN
    const int i = int(gl_GlobalInvocationID.x);
    if (i >= rllm_bound_x) return;
    const int j = int(gl_GlobalInvocationID.y);
    if (j >= rllm_bound_y) return;
    float sum = 0.f;
    for (int l_idx = 0; l_idx < 8; ++l_idx) {
        const int k = int(l_idx);
        const float term = (A[i, k] * B[j, k]);
        OVERFLOW_CHECK_ADD(sum, term);
        sum += term;
    }
    C[i, j] = sum;
END_PROGRAM
"""
    )
    gemm = detect_safe_gemm(program)
    assert gemm is not None
    assert gemm.input_type == "float16"
    assert gemm.k == 8

    program.blas = gemm
    stub = resolve_array_indices(program).accept(HipCppStubVisitor())
    assert "rllm::hip_blas::gemm_abt_half" in stub
    assert "beta=0;input=float16" in stub


def test_detects_float_abt_overwrite_for_muon_gram():
    program = parse(
        """
PROGRAM("backprop/Muon.cc:131")
OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (update, gram_matrix))
PARAMETERS
    const fixed_size_matrix<float, 8, 16>& update,
    fixed_size_matrix<float, 8, 8>& gram_matrix
BEGIN
    const int i = int(gl_GlobalInvocationID.x);
    if (i >= rllm_bound_x) return;
    const int j = int(gl_GlobalInvocationID.y);
    if (j >= rllm_bound_y) return;
    float sum = 0.f;
    for (int l_idx = 0; l_idx < 16; ++l_idx) {
        const int k = int(l_idx);
        const float term = (update[i, k] * update[j, k]);
        OVERFLOW_CHECK_ADD(sum, term);
        sum += term;
    }
    gram_matrix[i, j] = sum;
END_PROGRAM
"""
    )
    gemm = detect_safe_gemm(program)
    assert gemm is not None
    assert gemm.input_type == "float32"
    assert not gemm.accumulate
    assert gemm.transpose_b

    program.blas = gemm
    stub = resolve_array_indices(program).accept(HipCppStubVisitor())
    assert "rllm::hip_blas::gemm_abt_float" in stub
    assert "beta=0;input=float32" in stub


def test_rejects_half_abt_clip_from_blas_lowering():
    program = parse(
        """
PROGRAM("vecmath.cc:3")
OFFLOAD_PARFOR_2D_PARAM(queue, i, j, grid, (A, B, C))
PARAMETERS
    const flexible_rows_matrix<float16, 16, 8>& A,
    const fixed_size_matrix<float16, 32, 8>& B,
    flexible_rows_matrix<float, 16, 32>& C
BEGIN
    const int i = int(gl_GlobalInvocationID.x);
    if (i >= rllm_bound_x) return;
    const int j = int(gl_GlobalInvocationID.y);
    if (j >= rllm_bound_y) return;
    float sum = 0.f;
    for (int l_idx = 0; l_idx < 8; ++l_idx) {
        const int k = int(l_idx);
        const float term = (A[i, k] * B[j, k]);
        OVERFLOW_CHECK_ADD(sum, term);
        sum += term;
    }
    C[i, j] = clamp(sum, -10000.0f, 10000.0f);
END_PROGRAM
"""
    )
    assert detect_safe_gemm(program) is None
    forced = detect_safe_gemm(program, allow_clip_epilogue=True)
    assert forced is not None
    assert forced.epilogue == "clip"
