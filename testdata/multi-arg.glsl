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
                    for (int l_idx = 0; l_idx < 1024; ++l_idx) {
            }
        C[((1024 * i) + (1 * j))] += (sum1 + (sum2 + sum3));
}
