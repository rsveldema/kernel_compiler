"""Unit tests for the codegen package.

Tests the AST parser, visitor pattern, and printer against realistic parfor dumps.
"""

from codegen.parser import parse
from codegen.visitors.resolve_array_indices import resolve_array_indices
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
        const float scale = 1.0f / sqrt(float(size_t(128)));
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
