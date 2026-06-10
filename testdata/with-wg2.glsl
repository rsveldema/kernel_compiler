#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_clustered : require


layout(push_constant) uniform RllmPushConstants {
    int A;
} rllm_push;

void main() {
    int i = int(gl_GlobalInvocationID.x);
    int A = rllm_push.A;
}
