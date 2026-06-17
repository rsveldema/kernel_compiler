from codegen.parser import parse
from codegen.workgroup_partitioning import perform_tiling
from codegen.visitors.vulkan_kernel_visitor import VulkanKernelVisitor


def test_perform_tiling_tiles_a_loop_into_a_chunked_range():
    program = parse(
        """
PROGRAM("tile.cc:1")

OFFLOAD_PARFOR_1D_PARAM(i, limit<8>(n), (n))

PARAMETERS
        int n

BEGIN
        for (int i = 0; i < 8; i++) {
            n += i;
        }

END_PROGRAM
"""
    )

    tiled = perform_tiling(program, workgroups=4)
    shader = tiled.accept(VulkanKernelVisitor())

    assert tiled.tiled is True
    assert tiled.tile_block_size == 4
    assert "const int rllm_wg_count = 4;" in shader
    assert "const int local_id = int(gl_LocalInvocationID.x);" in shader
    assert (
        "const int chunk_size_i = ((8 + 3) / rllm_wg_count);" in shader
        or "const int chunk_size_i = (11 / rllm_wg_count);" in shader
    )
    assert "const int start_i = (local_id * chunk_size_i);" in shader
    assert "const int end_i = (start_i + chunk_size_i);" in shader
    assert (
        "for (int i = start_i; (i < end_i); ++i) {" in shader
        or "for (int i = start_i; i < end_i; ++i) {" in shader
    )
    assert "barrier();" in shader
