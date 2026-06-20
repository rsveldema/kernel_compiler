
// ── Kernel dispatch stub: vecmath.cc:40 ─────────────
#include <cstdint>
#include <string>
#include <vulkan/vulkan_core.h>
#include "vulkan_session.hpp"
#include "_type_aliases.hpp"

namespace rllm_offload_parfor_vecmath_40.cc {

inline constexpr uint32_t dst_X = 16384;

struct RllmBuffer_dst {
        int32_t data[16384];
};

struct offload_parfor_vecmath_40.cc_vecmath_PushConstants {
        int32_t rllm_bound_x;
        int32_t value;
};

class OffloadParforVecmath40.ccVecmathKernel : public AbstractKernel {
public:
    OffloadParforVecmath40.ccVecmathKernel(VulkanSession& session, const std::string& glsl_file)
        : AbstractKernel("rllm_offload_parfor_vecmath_40.cc::OffloadParforVecmath40.ccVecmathKernel|vecmath:40", KernelDimension::OneD, KernelType::Matrix, "tiling=off;parallelized=off;tile_block_size=1;workgroup_count=1;shared_memory_tiling=off")
        , kernel_(session, glsl_file, sizeof(offload_parfor_vecmath_40.cc_vecmath_PushConstants), 1)
    {
    }
    
    static constexpr const char* generated_descriptor() { return "tiling=off;parallelized=off;tile_block_size=1;workgroup_count=1;shared_memory_tiling=off"; }
    
    void dispatch(
            VulkanComputeContext& context,
            uint32_t dispatch_rows,
            VBaseDeviceBuffer& dst,
            const offload_parfor_vecmath_40.cc_vecmath_PushConstants& push_constants
    ) {
        ComputeKernelRegistry::ScopedActiveKernel active_kernel(*this);
        VkCommandBuffer command_buffer = context.begin_command_buffer();
        VkDescriptorSet desc_set = kernel_.desc_set();
        VkDescriptorBufferInfo buffer_infos[1]{};
        VkWriteDescriptorSet writes[1]{};
        buffer_infos[0].buffer = dst.get();
        buffer_infos[0].offset = 0;
        buffer_infos[0].range = VK_WHOLE_SIZE;
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = desc_set;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &buffer_infos[0];
        vkUpdateDescriptorSets(context.get_device(), 1, writes, 0, nullptr);
        vkCmdPushConstants(command_buffer, kernel_.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel_.pipeline());
        vkCmdDispatch(command_buffer, (dispatch_rows + 16 - 1) / 16, 1, 1);
        context.submit_and_wait();
        }
    
    private:
        VulkanComputeKernel kernel_;
    };
    }
