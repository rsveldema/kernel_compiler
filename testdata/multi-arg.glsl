#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_clustered : require

layout(std430, set = 0, binding = 0) buffer RllmBuffer_A1 {
    float A1[ /* unknown */ ];
};
layout(std430, set = 0, binding = 1) buffer RllmBuffer_B1 {
    float B1[1048576];
};
layout(std430, set = 0, binding = 2) buffer RllmBuffer_A2 {
    float A2[ /* unknown */ ];
};
layout(std430, set = 0, binding = 3) buffer RllmBuffer_B2 {
    float B2[1048576];
};
layout(std430, set = 0, binding = 4) buffer RllmBuffer_A3 {
    float A3[ /* unknown */ ];
};
layout(std430, set = 0, binding = 5) buffer RllmBuffer_B3 {
    float B3[1048576];
};
layout(std430, set = 0, binding = 6) buffer RllmBuffer_C {
    float C[ /* unknown */ ];
};

void main() {
    int i = int(gl_GlobalInvocationID.x);
    int j = int(gl_GlobalInvocationID.y);
        float sum1 = 0;
        float sum2 = 0;
        float sum3 = 0;
        sum1 = 0;
        sum2 = 0;
        sum3 = 0;
        const int block_start = 0;
        const int block_end = 1024;
                    for (int k = block_start; k < block_end; ++k) {
            const float term1 = (A1[((1024 * i) + (1 * k))] * B1[((1024 * k) + (1 * j))]);
            sum1 += term1;
            const float term2 = (A2[((1024 * i) + (1 * k))] * B2[((1024 * k) + (1 * j))]);
            sum2 += term2;
            const float term3 = (A3[((1024 * i) + (1 * k))] * B3[((1024 * k) + (1 * j))]);
            sum3 += term3;
            }
        C[((1024 * i) + (1 * j))] += (sum1 + (sum2 + sum3));
}
