#version 450

layout(binding = 0, std430) buffer _ssbo_0 {
    float[16777216] data;
} A1;
layout(binding = 1, std430) buffer _ssbo_1 {
    float[1048576] data;
} B1;
layout(binding = 2, std430) buffer _ssbo_2 {
    float[16777216] data;
} A2;
layout(binding = 3, std430) buffer _ssbo_3 {
    float[1048576] data;
} B2;
layout(binding = 4, std430) buffer _ssbo_4 {
    float[16777216] data;
} A3;
layout(binding = 5, std430) buffer _ssbo_5 {
    float[1048576] data;
} B3;
layout(binding = 6, std430) buffer _ssbo_6 {
    float[16777216] data;
} C;

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint j = gl_GlobalInvocationID.y;
        float sum1 = 0;
        float sum2 = 0;
        float sum3 = 0;
        for (uint l_idx; l_idx < 1024; l_idx++) {
            const uint k = l_idx;
            const float term1 = A1.data[i * 16777216 + k] * B1.data[k * 1048576 + j];
            sum1 += term1;
            const float term2 = A2.data[i * 16777216 + k] * B2.data[k * 1048576 + j];
            sum2 += term2;
            const float term3 = A3.data[i * 16777216 + k] * B3.data[k * 1048576 + j];
            sum3 += term3;
        }
        C.data[i * 16777216 + j] += sum1 + sum2 + sum3;
}
