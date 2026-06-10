#version 450

layout(std430, set = 0, binding = 0) buffer RllmBuffer_A1 {
    float A1[1];
} A1;
layout(std430, set = 0, binding = 1) buffer RllmBuffer_B1 {
    float B1[1024*1024];
} B1;
layout(std430, set = 0, binding = 2) buffer RllmBuffer_A2 {
    float A2[1];
} A2;
layout(std430, set = 0, binding = 3) buffer RllmBuffer_B2 {
    float B2[1024*1024];
} B2;
layout(std430, set = 0, binding = 4) buffer RllmBuffer_A3 {
    float A3[1];
} A3;
layout(std430, set = 0, binding = 5) buffer RllmBuffer_B3 {
    float B3[1024*1024];
} B3;
layout(std430, set = 0, binding = 6) buffer RllmBuffer_C {
    float C[1];
} C;

void main() {
    int i = int(gl_GlobalInvocationID.x);
    int j = int(gl_GlobalInvocationID.y);
        float sum1 = 0;
        float sum2 = 0;
        float sum3 = 0;
                    for (int l_idx = l_idx; l_idx < 1024; ++l_idx) {
            }
        C[((1024 * i) + (1 * j))] += (sum1 + (sum2 + sum3));
}
