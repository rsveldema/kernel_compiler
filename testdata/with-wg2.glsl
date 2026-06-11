#version 450


layout(push_constant) uniform RllmPushConstants {
    int A;
} rllm_push;

void main() {
    int i = int(gl_GlobalInvocationID.x);
    int A = rllm_push.A;
}
