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

END_PROGRAM
"""
    )
    program = resolve_array_indices(program)

    shader = program.accept(VulkanKernelVisitor())

    assert "float16_t weights[64];" in shader
    assert "weights[((8 * i) + (1 * i))] = float16_t(clamp(src[i], -2.0, 2.0));" in shader


def test_vulkan_uses_float_for_normalized_small_float_layout():
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
