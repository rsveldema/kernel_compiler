#version 450

layout(std430, set = 0, binding = 0) buffer RllmBuffer_dst {
    int dst[ /* unknown */ ];
};

layout(push_constant) uniform RllmPushConstants {
    int value;
} rllm_push;

void main() {
    int i = int(gl_GlobalInvocationID.x);
    int value = rllm_push.value;
        dst[i] = rllm_push.value;
}
