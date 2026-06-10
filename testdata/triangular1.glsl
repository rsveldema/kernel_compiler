#version 450

layout(std430, set = 0, binding = 0) buffer RllmBuffer_d_scores {
    float d_scores[8*16384*16384];
} d_scores;
layout(std430, set = 0, binding = 1) buffer RllmBuffer_d_raw {
    float d_raw[8*16384*16384];
} d_raw;
layout(std430, set = 0, binding = 2) buffer RllmBuffer_attn_w {
    float attn_w[8*16384*16384];
} attn_w;

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
            row_dot += (d_scores[(((268435456 * hi) + (16384 * i)) + (1 * k))] * attn_w[(((131072 * hi) + (8 * i)) + (1 * k))]);
            }
        d_raw[(((268435456 * hi) + (16384 * i)) + (1 * j))] = (attn_w[(((131072 * hi) + (8 * i)) + (1 * j))] * (d_scores[(((268435456 * hi) + (16384 * i)) + (1 * j))] - row_dot));
}
