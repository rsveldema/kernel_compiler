"""Unit tests for the codegen package.

Tests the AST parser, visitor pattern, and printer against realistic parfor dumps.
"""

from codegen.parser import parse
from codegen.visitors.resolve_array_indices import resolve_array_indices
from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor
from codegen.visitors.vulkan_kernel_visitor import VulkanKernelVisitor


def test_vulkan_constant_folds_sqrt_and_casts():
    program = parse(
        """
PROGRAM("fold.cc:1")

OFFLOAD_PARFOR_1D_PARAM(i, limit<8>(length), (dst, length))

PARAMETERS
        fixed_size_vector<float, 8>& dst,
        int length

BEGIN
        const float scale = (1.0f / sqrt(float(size_t(128))));
        dst[i] = scale;

END_PROGRAM
"""
    )
    program = resolve_array_indices(program)

    shader = program.accept(VulkanKernelVisitor())

    assert "sqrt" not in shader
    assert "float(size_t" not in shader
    assert "0.0883883476" in shader


def test_vulkan_preserves_float16_buffers_and_casts_stores():
    program = parse(
        """
PROGRAM("half.cc:1")

OFFLOAD_PARFOR_1D_PARAM(i, limit<8>(), (src, weights))

PARAMETERS
        fixed_size_vector<float, 8>& src,
        fixed_size_matrix<float16, 8, 8>& weights

BEGIN
        weights[i, i] = clamp(src[i], -2.0f, 2.0f);
        src[i] = (src[i] + weights[i, i]);

END_PROGRAM
"""
    )
    program = resolve_array_indices(program)

    shader = program.accept(VulkanKernelVisitor())

    assert "float16_t weights[64];" in shader
    assert "weights[((8 * i) + (1 * i))] = float16_t(clamp(src[i], -2.0, 2.0));" in shader
    assert "src[i] = (src[i] + float(weights[((8 * i) + (1 * i))]));" in shader


def test_vulkan_preserves_float_buffers():
    program = parse(
        """
PROGRAM("small.cc:1")

OFFLOAD_PARFOR_1D_PARAM(i, limit<8>(), (src, weights))

PARAMETERS
        fixed_size_vector<float, 8>& src,
        fixed_size_matrix<float, 8, 8>& weights

BEGIN
        src[i] = weights[i, i];

END_PROGRAM
"""
    )
    program = resolve_array_indices(program)

    shader = program.accept(VulkanKernelVisitor())

    assert "float weights[64];" in shader
    assert "float16_t weights[64];" not in shader


def test_vulkan_parses_coopmat_declarations_and_call_statements():
    program = parse(
        """
PROGRAM("coop.cc:1")

OFFLOAD_PARFOR_1D_PARAM(i, limit<1>(), (x))

PARAMETERS
        fixed_size_vector<float, 1>& x

BEGIN
        coopmat<float, gl_ScopeWorkgroup, 8, 16, gl_MatrixUseA> matrixA;
        coopMatLoadTensorNV(matrixA, x, 0, sliceTensorLayoutNV(tensorLayoutA, 0, 8, 0, 16));

END_PROGRAM
"""
    )

    shader = program.accept(VulkanKernelVisitor())

    assert "coopmat<float, gl_ScopeWorkgroup, 8, 16, gl_MatrixUseA> matrixA;" in shader
    assert "coopMatLoadTensorNV(matrixA, x, 0, sliceTensorLayoutNV(tensorLayoutA, 0, 8, 0, 16));" in shader


def test_vulkan_emits_overflow_check_as_comment():
    program = parse(
        """
PROGRAM("overflow.cc:1")

OFFLOAD_PARFOR_1D_PARAM(i, limit<8>(), (dst, src))

PARAMETERS
        fixed_size_vector<float, 8>& dst,
        fixed_size_vector<float, 8>& src

BEGIN
        const float term = src[i];
        OVERFLOW_CHECK_ADD(dst[i], term);
        dst[i] += term;

END_PROGRAM
"""
    )
    program = resolve_array_indices(program)

    shader = program.accept(VulkanKernelVisitor())

    assert "// OVERFLOW_CHECK_ADD(dst[i], term);" in shader
    assert "\nOVERFLOW_CHECK_ADD(" not in shader


def test_vulkan_fixed_size_triangular_matrix_uses_compact_storage():
    program = parse(
        """
PROGRAM("tri.cc:1")

OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(i, j, limit<8>(), (scores, out))

PARAMETERS
        fixed_size_triangular_matrix<float, 8, 8>& scores,
        fixed_size_vector<float, 8>& out

BEGIN
        scores[i, j] = (out[i] + out[j]);
        out[i] = scores[i, j];

END_PROGRAM
"""
    )
    program = resolve_array_indices(program)

    shader = program.accept(VulkanKernelVisitor())

    assert "float scores[36];" in shader
    assert "scores[(((i * (i + 1)) / 2) + j)] = (out[i] + out[j]);" in shader
    assert "out[i] = scores[(((i * (i + 1)) / 2) + j)];" in shader


def test_vulkan_preserves_if_comparison_operator():
    program = parse(
        """
PROGRAM("if.cc:1")

OFFLOAD_PARFOR_1D_PARAM(i, limit<8>(), (values, expected_output_token))

PARAMETERS
        fixed_size_vector<float, 8>& values,
        int expected_output_token

BEGIN
        if (i == expected_output_token)
            values[i] += 0.9f;

END_PROGRAM
"""
    )
    program = resolve_array_indices(program)

    shader = program.accept(VulkanKernelVisitor())

    assert "if (i == rllm_push.expected_output_token)" in shader


def test_vulkan_preserves_binary_operator_precedence():
    program = parse(
        """
PROGRAM("precedence.cc:1")

OFFLOAD_PARFOR_1D_PARAM(i, limit<8>(), (dst, src, learning_rate))

PARAMETERS
        fixed_size_vector<float, 8>& dst,
        fixed_size_vector<float, 8>& src,
        float learning_rate

BEGIN
        dst[i] = ((0.9f * dst[i]) + (learning_rate * src[i]));

END_PROGRAM
"""
    )
    program = resolve_array_indices(program)

    shader = program.accept(VulkanKernelVisitor())

    assert "dst[i] = ((0.9 * dst[i]) + (rllm_push.learning_rate * src[i]));" in shader


def test_cpp_stub_fixed_size_triangular_matrix_uses_compact_storage():
    program = parse(
        """
PROGRAM("tri.cc:1")

OFFLOAD_PARFOR_2D_TRIANGULAR_PARAM(i, j, limit<8>(), (scores))

PARAMETERS
        fixed_size_triangular_matrix<float, 8, 8>& scores

BEGIN
        scores[i, j] = 1.0f;

END_PROGRAM
"""
    )
    program = resolve_array_indices(program)

    stub = program.accept(VulkanCppStubVisitor())

    assert "inline constexpr uint32_t scores_X = 8;" in stub
    assert "inline constexpr uint32_t scores_Y = 8;" in stub
    assert "float data[36];" in stub
    assert "KernelType::Triangular" in stub



def test_workgroup_declaration_uses_custom_dispatch_sizes():
    """Verify workgroup { x: 8, y: 1, z: 1 } is used in vkCmdDispatch."""
    program = parse(
        """
PROGRAM("wg.cc:1")
workgroup { x: 8, y: 1, z: 1 }

OFFLOAD_PARFOR_1D_PARAM(i, limit<1024>(n), (dst))

PARAMETERS
        fixed_size_vector<float, 1024>& dst

BEGIN
        dst[i] = (dst[i] * 2.0f);

END_PROGRAM
"""
    )

    stub = program.accept(VulkanCppStubVisitor())

    # The dispatch should divide by 8, not 1
    assert "(dispatch_rows + 8 - 1) / 8" in stub
    # And the old hardcoded /1 should not appear
    assert "(dispatch_rows + 1 - 1) / 1" not in stub


def test_workgroup_sizes_512x64x1_in_dispatch():
    """Verify large workgroup sizes from the kernel are reflected in the stub."""
    program = parse(
        """
PROGRAM("bigwg.cc:1")
workgroup { x: 512, y: 64, z: 1 }

OFFLOAD_PARFOR_2D_PARAM(i, j, limit<512>(n), limit<256>(m), (dst))

PARAMETERS
        fixed_size_matrix<float, 512, 256>& dst

BEGIN
        dst[i, j] = (dst[i, j] + 1.0f);

END_PROGRAM
"""
    )

    stub = program.accept(VulkanCppStubVisitor())

    assert "(dispatch_rows + 512 - 1) / 512" in stub
    assert "(dispatch_cols + 64 - 1) / 64" in stub


def test_default_workgroup_when_no_declaration():
    """When no workgroup is declared, the visitor uses 16x1x1 for 1D."""
    program = parse(
        """
PROGRAM("nowg.cc:1")

OFFLOAD_PARFOR_1D_PARAM(i, limit<1024>(n), (dst))

PARAMETERS
        fixed_size_vector<float, 1024>& dst

BEGIN
        dst[i] = (dst[i] * 2.0f);

END_PROGRAM
"""
    )

    stub = program.accept(VulkanCppStubVisitor())

    # Default 1D workgroup size is 16x1x1
    assert "(dispatch_rows + 16 - 1) / 16" in stub
