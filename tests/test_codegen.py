"""Unit tests for the codegen package.

Tests the AST parser, visitor pattern, and printer against realistic parfor dumps.
"""

from unittest_offload_parfor_helper import parse_kernel

from codegen.parser import parse
from codegen.kast.expression import BinaryExpr, Number, WildcardExpression
from codegen.kast.statement import Declaration, If
from codegen.visitors.resolve_array_indices import resolve_array_indices
from codegen.visitors.tree_rewriter import TreeRewriter
from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor
from codegen.visitors.vulkan_kernel_visitor import VulkanKernelVisitor
from codegen.visitors.rllm_vulkan_dispatch_stub_visitor import RllmVulkanDispatchStubVisitor
import pytest


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


def test_logical_binary_operators_create_binary_expr_ast():
    program = parse(
        """
PROGRAM("logical.cc:1")

OFFLOAD_PARFOR_1D_PARAM(i, limit<8>(), (dst))

PARAMETERS
        fixed_size_vector<float, 8>& dst

BEGIN
        const int mask_and = ((i < 7) && (i > 0));
        const int mask_or = ((i < 7) || (i > 0));
        if ((i < 7) && (mask_and != 0)) return;
        dst[i] = float(mask_and);

END_PROGRAM
"""
    )

    declarations = [stmt for stmt in program.body_stmts if isinstance(stmt, Declaration)]

    assert isinstance(declarations[0].init_expr, BinaryExpr)
    assert declarations[0].init_expr.op == "&&"
    assert isinstance(declarations[1].init_expr, BinaryExpr)
    assert declarations[1].init_expr.op == "||"
    if_stmt = next(stmt for stmt in program.body_stmts if isinstance(stmt, If))
    assert isinstance(if_stmt.condition, BinaryExpr)
    assert if_stmt.condition.op == "&&"


def test_tkernel_meta_sets_reduction_chunks():
    program = parse_kernel("offload_parfor_vecmath_225.kernel")

    program = program.accept(TreeRewriter({}))
    program = resolve_array_indices(program)

    assert program.reduction_chunks == 8


def test_tkernel_meta_rejects_unknown_program_fields():
    program = parse(
        """
PROGRAM("vecmath.cc:225")

OFFLOAD_PARFOR_1D_PARAM(i, limit<8>(), (dst))

PARAMETERS
        fixed_size_vector<float, 8>& dst

BEGIN
        dst[i] = (dst[i] + 1.0f);

END_PROGRAM
"""
    )
    pattern = type(
        "FakePattern",
        (),
        {
            "filename": "optimize_vecmath225.tkernel",
            "meta": {"does_not_exist": Number(1)},
        },
    )()

    with pytest.raises(AttributeError, match="does_not_exist"):
        TreeRewriter({})._apply_program_meta(program, pattern)


def test_tkernel_meta_constant_folds_program_field_values():
    program = parse(
        """
PROGRAM("vecmath.cc:225")

OFFLOAD_PARFOR_1D_PARAM(i, limit<8>(), (dst))

PARAMETERS
        fixed_size_vector<float, 8>& dst

BEGIN
        dst[i] = (dst[i] + 1.0f);

END_PROGRAM
"""
    )
    pattern = type(
        "FakePattern",
        (),
        {
            "filename": "optimize_vecmath225.tkernel",
            "meta": {"reduction_chunks": BinaryExpr(Number(8), "*", Number(2))},
        },
    )()

    TreeRewriter({})._apply_program_meta(program, pattern)

    assert program.reduction_chunks == 16


def test_tkernel_meta_constant_folds_wildcard_values():
    program = parse(
        """
PROGRAM("vecmath.cc:296")

OFFLOAD_PARFOR_1D_PARAM(i, limit<8>(), (dst))

PARAMETERS
        fixed_size_vector<float, 8>& dst

BEGIN
        dst[i] = (dst[i] + 1.0f);

END_PROGRAM
"""
    )
    pattern = type(
        "FakePattern",
        (),
        {
            "filename": "optimize_vecmath296.tkernel",
            "meta": {
                "tile_chunk_size": Number(64),
                "num_z_threads": BinaryExpr(WildcardExpression("N"), "/", WildcardExpression("tile_chunk_size")),
            },
            "wildcard_expression_map": {"N": Number(1024)},
        },
    )()

    TreeRewriter({})._apply_program_meta(program, pattern)

    assert program.num_z_threads == 16


def test_tile_size_xy_meta_sets_generated_workgroup_size():
    program = parse(
        """
PROGRAM("tile.cc:1")

OFFLOAD_PARFOR_2D_PARAM(i, j, limit<64>(), limit<64>(), (dst))

PARAMETERS
        fixed_size_matrix<float, 64, 64>& dst

BEGIN
        dst[i, j] = 1.0f;

END_PROGRAM
"""
    )
    program.tile_size_x = 8
    program.tile_size_y = 32
    program.tile_chunk_size = 64

    shader = program.accept(VulkanKernelVisitor())
    stub = program.accept(VulkanCppStubVisitor())
    dispatch_stub = program.accept(RllmVulkanDispatchStubVisitor("/tile.spv"))

    assert "layout(local_size_x = 8, local_size_y = 32, local_size_z = 1) in;" in shader
    assert "(dispatch_rows + 8 - 1) / 8" in stub
    assert "(dispatch_cols + 32 - 1) / 32" in stub
    assert "tile_size_x=8;" in stub
    assert "tile_size_y=32;" in stub
    assert "tile_chunk_size=64;" in stub
    assert "tile_L1_kernel_wg_x = 8;" in dispatch_stub
    assert "tile_L1_kernel_wg_y = 32;" in dispatch_stub


def test_num_z_threads_sets_generated_workgroup_z():
    program = parse(
        """
PROGRAM("tile.cc:2")

OFFLOAD_PARFOR_3D_PARAM(i, j, k, limit<64>(), (dst))

PARAMETERS
        fixed_size_vector<float, 64>& dst

BEGIN
        dst[i] = 1.0f;

END_PROGRAM
"""
    )
    program.tile_size_x = 8
    program.tile_size_y = 8
    program.num_z_threads = 4

    shader = program.accept(VulkanKernelVisitor())
    stub = program.accept(VulkanCppStubVisitor())
    dispatch_stub = program.accept(RllmVulkanDispatchStubVisitor("/tile.spv"))

    assert "layout(local_size_x = 8, local_size_y = 8, local_size_z = 4) in;" in shader
    assert "(dispatch_levels + 4 - 1) / 4" in stub
    assert "tile_L2_kernel_wg_z = 4;" in dispatch_stub


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

    assert "#extension GL_EXT_shader_16bit_storage : require" in shader
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


def test_workgroup_sizes_512x2x1_in_dispatch():
    """Verify a large valid workgroup size from the kernel is reflected in the stub."""
    program = parse(
        """
PROGRAM("bigwg.cc:1")
workgroup { x: 512, y: 2, z: 1 }

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
    assert "(dispatch_cols + 2 - 1) / 2" in stub


def test_stub_generation_rejects_too_many_local_invocations():
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

    with pytest.raises(ValueError, match="maxWorkGroupInvocations"):
        program.accept(VulkanCppStubVisitor())


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


def test_vulkan_constant_folds_max():
    program = parse(
        """
PROGRAM("max.cc:1")

OFFLOAD_PARFOR_1D_PARAM(i, limit<8>(), (dst))

PARAMETERS
        fixed_size_vector<float, 8>& dst

BEGIN
        const float threshold = std.max(10.5f, 3.2f);
        dst[i] = threshold;

END_PROGRAM
"""
    )
    program = resolve_array_indices(program)

    shader = program.accept(VulkanKernelVisitor())

    assert "std.max" not in shader
    assert "max(" not in shader
    assert "10.5" in shader


def test_vulkan_constant_folds_min():
    program = parse(
        """
PROGRAM("min.cc:1")

OFFLOAD_PARFOR_1D_PARAM(i, limit<8>(), (dst))

PARAMETERS
        fixed_size_vector<float, 8>& dst

BEGIN
        const float threshold = std.min(10.5f, 3.2f);
        dst[i] = threshold;

END_PROGRAM
"""
    )
    program = resolve_array_indices(program)

    shader = program.accept(VulkanKernelVisitor())

    assert "std.min" not in shader
    assert "min(" not in shader
    assert "3.2" in shader


def test_vulkan_constant_folds_max_int():
    program = parse(
        """
PROGRAM("max_int.cc:1")

OFFLOAD_PARFOR_1D_PARAM(i, limit<8>(), (dst))

PARAMETERS
        fixed_size_vector<int, 8>& dst

BEGIN
        const int val = std.max(42, 17);
        dst[i] = val;

END_PROGRAM
"""
    )
    program = resolve_array_indices(program)

    shader = program.accept(VulkanKernelVisitor())

    assert "std.max" not in shader
    assert "max(" not in shader
    assert "42" in shader


def test_vulkan_constant_folds_max_mixed_types():
    """max(int, float) should produce a float result."""
    program = parse(
        """
PROGRAM("max_mixed.cc:1")

OFFLOAD_PARFOR_1D_PARAM(i, limit<8>(), (dst))

PARAMETERS
        fixed_size_vector<float, 8>& dst

BEGIN
        const float val = std.max(3, 2.5f);
        dst[i] = val;

END_PROGRAM
"""
    )
    program = resolve_array_indices(program)

    shader = program.accept(VulkanKernelVisitor())

    assert "std.max" not in shader
    assert "max(" not in shader
    # Should fold to 3.0 (float)
    assert "3.0" in shader
