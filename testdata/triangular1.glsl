#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_clustered : require

layout(std430, set = 0, binding = 0) buffer RllmBuffer_d_scores {
    float d_scores[ /* too large for int: 2147483648 */ ];
};
layout(std430, set = 0, binding = 1) buffer RllmBuffer_d_raw {
    float d_raw[ /* too large for int: 2147483648 */ ];
};
layout(std430, set = 0, binding = 2) buffer RllmBuffer_attn_w {
    float attn_w[ /* too large for int: 2147483648 */ ];
};

layout(push_constant) uniform RllmPushConstants {
    int seq_len;
    int d_scores_rows;
    int d_scores_cols;
    int d_raw_rows;
    int d_raw_cols;
} rllm_push;

void main() {
    int hi = int(gl_GlobalInvocationID.x);
    int i = int(gl_GlobalInvocationID.y);
    int j = int(gl_GlobalInvocationID.z);
    int seq_len = rllm_push.seq_len;
    int d_scores_rows = rllm_push.d_scores_rows;
    int d_scores_cols = rllm_push.d_scores_cols;
    int d_raw_rows = rllm_push.d_raw_rows;
    int d_raw_cols = rllm_push.d_raw_cols;
    if (hi >= 8 || i >= 8 || j >= 8 || i >= rllm_push.seq_len || j >= rllm_push.seq_len) return;
        float row_dot = 0;
                    for (int k = 0; k < 16384; ++k) {
            row_dot += (d_scores[(((268435456U * hi) + (16384U * i)) + (1U * k))] * attn_w[(((131072U * hi) + (8U * i)) + (1U * k))]);
            }
        d_raw[(((268435456U * hi) + (16384U * i)) + (1U * j))] = (attn_w[(((131072U * hi) + (8U * i)) + (1U * j))] * (d_scores[(((268435456U * hi) + (16384U * i)) + (1U * j))] - row_dot));
}
