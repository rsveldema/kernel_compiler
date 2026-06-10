#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_clustered : require

layout(std430, set = 0, binding = 0) buffer RllmBuffer_dst {
    float dst[ /* unknown */ ];
};

layout(push_constant) uniform RllmPushConstants {
    int value;
} rllm_push;

void main() {
    int i = int(gl_GlobalInvocationID.x);
    int value = rllm_push.value;
        dst[i] = rllm_push.value;
}
