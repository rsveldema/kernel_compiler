#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_EXT_shader_atomic_float : require
#extension GL_EXT_shader_atomic_float2 : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


layout(push_constant) uniform RllmPushConstants {
    int rllm_wg_count;
    int rllm_bound_x;
    int value;
} rllm_push;
layout(std430, set = 0, binding = 0) buffer RllmBuffer_dst {
    int dst[16384];
};

void main() {
    const int global_i = int(gl_GlobalInvocationID.x);
    const int i = global_i;
    int rllm_wg_count = rllm_push.rllm_wg_count;
    int rllm_bound_x = rllm_push.rllm_bound_x;
    int value = rllm_push.value;
        dst[i] = rllm_push.value;
}
