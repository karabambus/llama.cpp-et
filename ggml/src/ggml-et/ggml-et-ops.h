#pragma once

#include "ggml-et-common.h"
#include "ggml.h"

#include <inttypes.h>

// Performance logging macros for ET ops
// Logs in machine-parseable pipe-delimited format: ET_PERF|field=value|...
#ifdef ET_PERF_RECORD
#    define ET_PERF_START() int64_t _et_perf_start = ggml_time_us()

#    define ET_PERF_END(op_name, kernel_name, node)                                                                \
        do {                                                                                                       \
            int64_t _et_perf_end      = ggml_time_us();                                                            \
            int64_t _et_perf_duration = _et_perf_end - _et_perf_start;                                             \
            GGML_LOG_DEBUG("ET_PERF|op=%s|kernel=%s|duration_us=%" PRId64 "|tensor=%s|shape=[%" PRId64 ",%" PRId64 \
                           ",%" PRId64 ",%" PRId64 "]|start_us=%" PRId64 "|end_us=%" PRId64 "\n",                  \
                           op_name, kernel_name, _et_perf_duration, (node)->name, (node)->ne[0], (node)->ne[1],    \
                           (node)->ne[2], (node)->ne[3], _et_perf_start, _et_perf_end);                            \
        } while (0)

#    define ET_PERF_END_EXT(op_name, kernel_name, node, fmt, ...)                                                  \
        do {                                                                                                       \
            int64_t _et_perf_end      = ggml_time_us();                                                            \
            int64_t _et_perf_duration = _et_perf_end - _et_perf_start;                                             \
            GGML_LOG_DEBUG("ET_PERF|op=%s|kernel=%s|duration_us=%" PRId64 "|tensor=%s|shape=[%" PRId64 ",%" PRId64 \
                           ",%" PRId64 ",%" PRId64 "]|start_us=%" PRId64 "|end_us=%" PRId64 "|" fmt "\n",          \
                           op_name, kernel_name, _et_perf_duration, (node)->name, (node)->ne[0], (node)->ne[1],    \
                           (node)->ne[2], (node)->ne[3], _et_perf_start, _et_perf_end, ##__VA_ARGS__);             \
        } while (0)
#else

#    define ET_PERF_START() \
        do {                \
        } while (0)
#    define ET_PERF_END_EXT(op_name, kernel_name, node, fmt, ...) \
        do {                                                      \
            (void) (node);                                        \
        } while (0)
#    define ET_PERF_END(op_name, kernel_name, node) \
        do {                                        \
            (void) (node);                          \
        } while (0)

#endif  // ET_PERF_RECORD

struct ggml_et_binary_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
};

// Q8_0 mul_mat with optional residual bias.
// bias.data == NULL means "no bias" - kernel skips the add.
// When non-NULL, bias must have the same shape and strides as dst.
struct ggml_et_mm_q8_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
    ggml_tensor bias;
    uint64_t    tail_scratch;  // PQ2_0 GEMV: per-row records for the K-split tail (0 = off)
    uint32_t    tail_gen;      // this launch's generation, unique per launch and never 0
    uint32_t    tail_pad;
};

struct ggml_et_im2col_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
};

// Element map parameters for embarrassingly parallel binary operations (MUL, ADD, etc.)
// Operation type is determined by dst->op (GGML_OP_MUL, GGML_OP_ADD, etc.)
struct ggml_et_elmap_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
};

struct ggml_et_rope_settings {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t mode;    // ROPE mode, GGML_ROPE_TYPE_*
    int32_t n_ctx;
    int32_t n_ctx_orig;
    float   freq_base;    // Base frequency (usually 10000.0f)
    float   freq_scale;   // Frequency scaling factor
    float   ext_factor;   // Extension factor for YaRN
    float   attn_factor;  // Attention factor for YaRN
    float   beta_fast;    // Fast beta for YaRN
    float   beta_slow;    // Slow beta for YaRN
    int32_t sections[4];  // Sections for multi-modal ROPE
};

struct ggml_et_rope_params {
    ggml_tensor           src0;
    ggml_tensor           src1;
    ggml_tensor           src2;
    ggml_tensor           dst;
    ggml_et_rope_settings rope_params;
};

struct ggml_et_rms_norm_params {
    ggml_tensor src0;  // F32 input tensor
    ggml_tensor dst;   // F32 output tensor
    float       eps;   // Epsilon parameter for numerical stability
};

struct ggml_et_norm_params {
    ggml_tensor src0;  // F32 input tensor
    ggml_tensor dst;   // F32 output tensor
    float       eps;   // Epsilon parameter for numerical stability
};

struct ggml_et_l2_norm_params {
    ggml_tensor src0;  // F32 input tensor
    ggml_tensor dst;   // F32 output tensor
    float       eps;   // Epsilon parameter for numerical stability
};

struct ggml_et_group_norm_params {
    ggml_tensor src0;      // F32 input tensor
    ggml_tensor dst;       // F32 output tensor
    int32_t     n_groups;  // Number of channel groups
    float       eps;       // Epsilon parameter for numerical stability
};

struct ggml_et_glu_params {
    ggml_tensor src0;         // F32 input tensor A (or combined tensor if src1 is null)
    ggml_tensor src1;         // F32 input tensor B (null for single tensor mode)
    ggml_tensor dst;          // F32 output tensor (n/2 columns)
    int32_t     glu_op_type;  // GLU operation type (REGLU=0, GEGLU=1, SWIGLU=2, etc.)
    int32_t     swapped;      // Whether gate and value are swapped
    float       alpha;        // SWIGLU_OAI: sigmoid scaling factor (unused for other variants)
    float       limit;        // SWIGLU_OAI: clamp limit (unused for other variants)
};

struct ggml_et_softmax_params {
    ggml_tensor src0;      // F32 input tensor
    ggml_tensor src1;      // F32 mask tensor (optional, may be zeroed if not used)
    ggml_tensor src2;      // F32 sinks tensor (optional, may be zeroed if not used)
    ggml_tensor dst;       // F32 output tensor
    float       scale;     // Scale factor
    float       max_bias;  // Max bias for ALiBi (0.0f if not used)
};

struct ggml_et_flash_attn_ext_params {
    ggml_tensor src0;      // Q tensor (F32)
    ggml_tensor src1;      // K tensor (F32)
    ggml_tensor src2;      // V tensor (F32)
    ggml_tensor mask;      // mask tensor (F16 or F32), zeroed when absent
    ggml_tensor dst;       // Output tensor (F32)
    float       scale;     // Scale factor applied to QK
    int32_t     has_mask;  // nonzero if mask is present
    int32_t     gqa_tiles; // nonzero: batches of >= 64 queries may take the GQA tile path (GGML_ET_FA_GQA=0 disables)
};

struct ggml_et_get_rows_params {
    ggml_tensor src0;  // Data tensor (F32 or Q8_0)
    ggml_tensor src1;  // Row indices tensor (I32)
    ggml_tensor dst;   // Output tensor (F32)
};

struct ggml_et_cont_params {
    ggml_tensor src0;  // F32 input tensor (non-contiguous)
    ggml_tensor dst;   // F32 output tensor (contiguous)
};

struct ggml_et_q8k_params {
    ggml_tensor src0;  // F32, ne0 % 256 == 0
    ggml_tensor dst;   // Q8_K, contiguous (quantize_q8k_f32.c); engine mode: I8 [K, N] rows
    ggml_tensor sc;    // engine mode: F32 scale lines, line (t8 * K/256 + kb) = tokens 8*t8 .. +7 of 256-block kb
};

struct ggml_et_pq2_i8_params {
    ggml_tensor src0;  // PQ2_0 weights [K, M]
    ggml_tensor a8;    // I8 activations [K, N] (quantize_q8k_f32 engine mode)
    ggml_tensor sc;    // their scale lines
    ggml_tensor dst;   // F32 [M, N]
};

struct ggml_et_concat_params {
    ggml_tensor src0;  // F32 input tensor 0
    ggml_tensor src1;  // F32 input tensor 1
    ggml_tensor dst;   // F32 output tensor
    int32_t     dim;   // Concatenation dimension
};

struct ggml_et_repeat_params {
    ggml_tensor src0;  // F32 input tensor (tile)
    ggml_tensor dst;   // F32 output tensor (tiled result)
};

struct ggml_et_fill_params {
    ggml_tensor dst;  // F32 output tensor (contiguous)
    float       c;    // Constant value to fill
};

struct ggml_et_tri_params {
    ggml_tensor src0;      // F32 input tensor
    ggml_tensor dst;       // F32 output tensor
    int32_t     tri_type;  // ggml_tri_type enum value
};

struct ggml_et_solve_tri_params {
    ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

struct ggml_et_pad_params {
    ggml_tensor src0;   // F32 input (may be non-contiguous, nb[0] must == 4)
    ggml_tensor dst;    // F32 output (contiguous, ne[0] % 16 == 0)
    int32_t     lp[4];  // left padding per dimension
    int32_t     rp[4];  // right padding per dimension
};

struct ggml_et_diag_params {
    ggml_tensor src0;  // F32 input vector
    ggml_tensor dst;   // F32 output diagonal matrix
};

#define GGML_ET_SSMC_MAX_UPD 8
struct ggml_et_ssm_conv_params {
    ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]; state mode: the conv-state table [3 * d_inner, rows]
    ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
    // State mode (n_upd > 0; one sequence, d_conv = 4): conv_x is state row ids[0] of src0 followed by x, dst is
    // silu(conv), and state slot j (upd_dst[j], d_inner x 3 floats) receives conv_x columns upd_off[j] .. upd_off[j] + 2.
    ggml_tensor ids;   // I32 [1]
    ggml_tensor x;     // new inputs [d_inner, n_t], unit channel stride
    uint32_t    n_upd;
    uint32_t    reserved;
    uint64_t    upd_dst[GGML_ET_SSMC_MAX_UPD];
    int64_t     upd_off[GGML_ET_SSMC_MAX_UPD];
};

struct ggml_et_ssm_scan_params {
    ggml_tensor src0;  // s:   [d_state, head_dim, n_head, n_seqs]
    ggml_tensor src1;  // x:   [head_dim, n_head, n_seq_tokens, n_seqs]
    ggml_tensor src2;  // dt:  [n_head, n_seq_tokens, n_seqs]
    ggml_tensor src3;  // A:   [d_state, n_head] or [1, n_head]
    ggml_tensor src4;  // B:   [d_state, n_group, n_seq_tokens, n_seqs]
    ggml_tensor src5;  // C:   [d_state, n_group, n_seq_tokens, n_seqs]
    ggml_tensor src6;  // ids: [n_seqs] i32
    ggml_tensor dst;   // [y, states] packed output from ggml_ssm_scan()
    int32_t     K;
};

struct ggml_et_rwkv_wkv6_params {
    float * k;         // src[0]: [S, H, T]  key
    float * v;         // src[1]: [S, H, T]  value
    float * r;         // src[2]: [S, H, T]  receptance
    float * tf;        // src[3]: [S, H]     time_faaaa (per-head)
    float * td;        // src[4]: [S, H, T]  time_decay
    float * state_in;  // src[5]: [S*S*H, n_seqs]  initial state
    float * dst;       // [C, T + S*n_seqs]  output + state_out
    int32_t C;         // total channels (S * H)
    int32_t H;         // number of heads
    int32_t S;         // head size
    int32_t T;         // number of tokens
    int32_t n_seqs;    // number of sequences
};

struct ggml_et_rwkv_wkv7_params {
    float * r;         // [S, H, T]  receptance
    float * w;         // [S, H, T]  decay
    float * k;         // [S, H, T]  key
    float * v;         // [S, H, T]  value
    float * a;         // [S, H, T]  bonus gate
    float * b;         // [S, H, T]  bonus key
    float * state_in;  // [S*S*H, n_seqs]  initial state
    float * dst;       // [C, T + S*n_seqs]  output + state_out
    int32_t C;         // total channels (S * H)
    int32_t H;         // number of heads
    int32_t S;         // head size
    int32_t T;         // number of tokens
    int32_t n_seqs;    // number of sequences
};

struct ggml_et_gated_delta_net_params {
    ggml_tensor q;         // [S_v, H_q, n_tokens, n_seqs_q]
    ggml_tensor k;         // [S_v, H_k, n_tokens, n_seqs_k]
    ggml_tensor v;         // [S_v, H, n_tokens, n_seqs]
    ggml_tensor g;         // [1 or S_v, H, n_tokens, n_seqs]
    ggml_tensor beta;      // [1, H, n_tokens, n_seqs]
    ggml_tensor state_in;  // [S_v, S_v, H, n_seqs], or the 2-D cache view [S_v*S_v*H, n_rows] in rows mode
    ggml_tensor dst;       // [S_v*H, n_tokens*n_seqs + S_v*n_seqs*K]
    // Optional tensors: data == NULL when the variant is off. The uberkernel dispatcher mirrors the tensor
    // prefix q .. a (uber_gated_delta_net_params) to evict the writable inputs; keep the order.
    ggml_tensor rows;      // I32 [n_seqs]: rows mode (ggml_gated_delta_net_rows), state of seq s at row rows[s]
    ggml_tensor dt_bias;   // F32 [H]: raw gates (ggml_gated_delta_net_set_raw_gates)
    ggml_tensor a;         // F32 [H]: raw gates
    int32_t     S_v;       // head dimension (value size)
    int32_t     H;         // number of value heads
    int32_t     H_q;       // number of Q heads
    int32_t     H_k;       // number of K heads
    int32_t     n_tokens;  // total tokens
    int32_t     n_seqs;    // number of sequences (from V)
    int32_t     n_seqs_q;  // Q sequence count
    int32_t     n_seqs_k;  // K sequence count
    int32_t     kda;       // 1 if per-element gate (g_ne0 == S_v), 0 if scalar
    int32_t     K;         // snapshot slot count
    int32_t     raw_gates; // 1: beta = sigmoid(beta), g = a[h] * softplus(g + dt_bias[h]) (scalar gate only)
    float       scale;     // 1/sqrt(S_v)
    // Folded snapshot scatter (rows mode, one sequence; data == NULL when off): the graph's SET_ROWS that copies the
    // snapshot planes of dst into the state cache is skipped and the kernel writes slot s straight to row
    // snap_rows[s] of snap_dst.
    ggml_tensor snap_rows; // I64 [n_written]
    ggml_tensor snap_dst;  // state cache [S_v*S_v*H, n_cache_rows], contiguous rows
};

struct ggml_et_set_rows_params {
    ggml_tensor src0;  // F32 source data tensor
    ggml_tensor src1;  // I64 row indices tensor
    ggml_tensor dst;   // F32/F16 destination tensor
};

struct ggml_et_set_params {
    ggml_tensor src1;    // F32 source view to write into dst
    ggml_tensor dst;     // F32 destination/base tensor
    int32_t     nb1;     // destination view stride for dim 1
    int32_t     nb2;     // destination view stride for dim 2
    int32_t     nb3;     // destination view stride for dim 3
    int32_t     offset;  // byte offset into destination
};

struct ggml_et_rms_norm_mul_params {
    ggml_tensor src0;     // F32 input tensor (to be normalized)
    ggml_tensor src1;     // F32 weights tensor (element-wise multiply)
    ggml_tensor dst;      // F32 output tensor
    float       eps;      // Epsilon for numerical stability
    int32_t     has_res;  // 1: normalise sum = src0 + res and write sum too (a fused residual ADD)
    ggml_tensor res;      // F32, same shape as src0 (has_res only)
    ggml_tensor sum;      // F32, same shape as src0 (has_res only); may alias src0 or res
};

struct ggml_et_mul_mat_id_params {
    ggml_tensor src0;  // Expert weight matrices (Q8_0/F16/F32) [K, M, n_expert]
    ggml_tensor src1;  // Activations (F32) [K, n_expert_used, batch]
    ggml_tensor src2;  // Expert indices (I32) [n_expert_used, batch]
    ggml_tensor dst;   // Output (F32) [M, n_expert_used, batch, 1]
    uint64_t    shire_mask;  // narrowed launch mask (GGML_ET_MMID_SHIRE_MASK) or 0 = kernel uses kernel_env->shire_mask;
                             // must match et-kernels/src/ggml_tensor.h
};

struct ggml_et_sqr_params {
    ggml_tensor src0;  // F32 input tensor
    ggml_tensor dst;   // F32 output tensor
};

struct ggml_et_unary_params {
    ggml_tensor src0;      // F32 input tensor
    ggml_tensor dst;       // F32 output tensor
    int32_t     unary_op;  // ggml_unary_op enum value
};

struct ggml_et_argsort_params {
    ggml_tensor src0;  // F32 input [ne0, ne1, ne2, ne3]
    ggml_tensor dst;   // I32 output, same shape; order in dst.op_params[0]
};

struct ggml_et_sum_rows_params {
    ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    ggml_tensor dst;   // F32 output tensor [1, ne01, ne02, ne03]
};

struct ggml_et_mean_params {
    ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    ggml_tensor dst;   // F32 output tensor [1, ne01, ne02, ne03]
};

struct ggml_et_clamp_params {
    ggml_tensor src0;  // F32 input tensor (contiguous)
    ggml_tensor dst;   // F32 output tensor (contiguous; may alias src0)
    float       min_val;
    float       max_val;
};

struct ggml_et_cumsum_params {
    ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    ggml_tensor dst;   // F32 output tensor [ne00, ne01, ne02, ne03]
};

// hadamard_f32: dst[n, R] = blockwise FWHT of scale * signs * src; signs.data == NULL means no sign flip.
// Device-side twin in et-kernels/src/hadamard_f32.c.
struct ggml_et_hadamard_params {
    ggml_tensor src;
    ggml_tensor signs;
    ggml_tensor dst;
    int64_t     width;
    int64_t     blk;
    float       scale;
    int32_t     rows_mode;  // nonzero: launches of >= 256 rows give each hart whole rows (GGML_ET_HAD_ROWS=0 disables)
};

struct ggml_et_scale_params {
    ggml_tensor src0;   // F32 input tensor
    ggml_tensor dst;    // F32 output tensor
    float       scale;  // Scale factor
    float       bias;   // Bias (additive offset)
};

// ---- T2 launch-count fusions (experiments/T2, 2026-08-26) ----------------------------------------------------

// Fusion 1: the llama.cpp MoE routing chain SOFT_MAX -> ARGSORT(top_k view) -> GET_ROWS -> SUM_ROWS -> CLAMP -> DIV
// as one moe_router_f32 launch. Only the tensors the rest of the graph reads are written: the first n_top entries
// of every ARGSORT row (the top_k view feeds MUL_MAT_ID) and the normalised weights (the DIV output).
struct ggml_et_moe_router_params {
    ggml_tensor src0;       // logits [E, N] F32 (the SOFT_MAX input)
    ggml_tensor ids;        // ARGSORT output [E, N] I32
    ggml_tensor dst;        // DIV output [K, N] F32
    float       clamp_min;  // CLAMP bounds applied to the weight sum
    float       clamp_max;
    int32_t     n_top;      // K = n_expert_used
    int32_t     reserved;
};

// Fusion 4b: 038's moe_sum with the layer's residual ADD folded in (res.data == nullptr -> none).
struct ggml_et_moe_sum_params {
    ggml_tensor src0;  // experts [E, S, N]
    ggml_tensor src1;  // weights [1, S, N]
    ggml_tensor dst;   // [E, N]
    ggml_tensor res;   // residual [E, N] (optional)
};

// Fusion 2: attention glue — per job: y = rms_norm(x)*w (optional), z = rope(y) (optional), z -> dst (optional),
// z -> kv[idx[t]] (SET_ROWS into the KV cache, optional). Compact descriptors (no ggml_tensor copies) so three
// jobs fit comfortably in the launch argument block. Layout must match attn_qkv_glue_f32.c exactly.
#define ET_QKV_GLUE_MAX_JOBS 3
struct ggml_et_qkv_job {
    const void * src;
    const void * w;
    const void * pos;
    void *       dst;
    void *       kv;
    const void * idx;
    uint64_t     src_nb1, src_nb2;
    uint64_t     dst_nb1, dst_nb2;
    uint64_t     kv_nb1, idx_nb0;
    int64_t      kv_rows;
    int32_t      D, H, N;
    int32_t      kv_f16;
    float        eps;
    int32_t      reserved;
};

struct ggml_et_qkv_glue_params {
    ggml_et_qkv_job       job[ET_QKV_GLUE_MAX_JOBS];
    ggml_et_rope_settings rope;
    int32_t               n_jobs;
    int32_t               reserved;
};

// Host-side description of one matched glue job (filled by the graph matcher in ggml-et.cpp).
struct ggml_et_qkv_glue_job_desc {
    const ggml_tensor * x;         // [D, H, N] input rows (the RMS_NORM's src, or the SET_ROWS source for a copy job)
    const ggml_tensor * rms_norm;  // nullptr for a copy-only job
    const ggml_tensor * mul;       // the MUL that applies the norm weight
    const ggml_tensor * rope;      // nullptr for a copy-only job
    const ggml_tensor * set_rows;  // nullptr when the rope output is only materialised
    int64_t             D, H, N;
};

struct ggml_et_qkv_glue_plan {
    ggml_et_qkv_glue_job_desc job[ET_QKV_GLUE_MAX_JOBS];
    int                       n_jobs;
    int                       last_idx;  // last graph node consumed
};

bool ggml_et_op_cumsum(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_sqr(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_unary(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_sum_rows(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_argsort(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
// 038: fused MoE expert aggregation dst = sum_s experts[:,s,:] * weights[0,s,:] (MUL + view-ADD chain);
// T2: optional residual `res` (the following ADD) folded into the same launch.
bool ggml_et_op_moe_sum(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * mul_node, const ggml_tensor * dst_node,
                        const ggml_tensor * res = nullptr);
// T2 fusion 1: routing chain. logits = the SOFT_MAX input, argsort = the ARGSORT node, div_node = the DIV output,
// clamp_node supplies the CLAMP bounds.
bool ggml_et_op_moe_router(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * logits, const ggml_tensor * argsort,
                           const ggml_tensor * div_node, const ggml_tensor * clamp_node, int n_top);
// T2 fusion 2: attention glue (norm + rope + set_rows jobs), see ggml_et_qkv_glue_plan.
bool ggml_et_op_qkv_glue(ggml_backend_et_device_context * dev_ctx, const ggml_et_qkv_glue_plan & plan);
// Packs a ROPE node's op_params into the kernel settings struct (shared by rope and the glue kernel).
void ggml_et_fill_rope_settings(const ggml_tensor * rope_node, ggml_et_rope_settings * out);
bool ggml_et_op_mean(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_clamp(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
// Fused Hadamard activation transform: `mm` is the hinted MUL_MAT (dst), `x` the activation before the sign flip
// (mm->src[1] itself when there is none), `signs` the 1-D sign vector or nullptr.
bool ggml_et_op_hadamard(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * x, const ggml_tensor * signs,
                         const ggml_tensor * mm);
bool ggml_et_op_scale(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_mul(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_add(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_div(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_sub(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
// add_node is optional: when non-NULL and the pair (node, add_node) was
// validated by ggml_et_can_fuse({MUL_MAT, ADD}), the Q8_0 path writes
// dst = mm(...) + add_node's "other" operand (the bias) in one launch.
bool ggml_et_op_mul_mat(ggml_backend_et_device_context * dev_ctx,
                        const ggml_tensor *              node,
                        const ggml_tensor *              add_node = nullptr);
bool ggml_et_op_mul_mat_id(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_rope(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_rms_norm(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_norm(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_l2_norm(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_group_norm(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_glu(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_softmax(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_im2col(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_conv_2d(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_flash_attn_ext(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_get_rows(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_set_rows(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_cont(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_concat(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_repeat(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_rwkv_wkv6(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_rwkv_wkv7(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_cpy(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_gated_delta_net(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node,
                                const ggml_tensor * fused_set_rows = nullptr);
bool ggml_et_op_elmap(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_fill(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_diag(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_tri(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_solve_tri(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_pad(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_set(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
bool ggml_et_op_ssm_conv(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
// State mode of ssm_conv_f32: reads state row ids[0] of `table` and the new inputs x ([d_inner, n_t], unit channel
// stride), writes silu(conv) to `act` and conv_x columns upd_off[j] .. +2 to the state slots upd_dst[j].
bool ggml_et_op_ssm_conv_state(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * table,
                               const ggml_tensor * ids, const ggml_tensor * x_t, const ggml_tensor * weight,
                               const ggml_tensor * act, const ggml_tensor * const * upd, const int64_t * upd_off,
                               int n_upd);
bool ggml_et_op_ssm_scan(ggml_backend_et_device_context * dev_ctx, const ggml_tensor * node);
// add_node (optional): rms_norm_node->src[0] == add_node; the kernel also computes and writes the ADD.
bool ggml_et_op_rms_norm_mul(ggml_backend_et_device_context * dev_ctx,
                             const ggml_tensor *              rms_norm_node,
                             const ggml_tensor *              mul_node,
                             const ggml_tensor *              add_node = nullptr);

// Independent Q8_0 mul_mats that share one activation, issued as one dispatch.
// Members write disjoint outputs and none reads another's, so no barrier is needed
// between them -- which is the whole saving under the uberkernel, where every
// instruction is preceded by a chip-wide barrier.
#define ET_MUL_MAT_GROUP_MAX 3

struct ggml_et_mul_mat_group_params {
    int32_t     n_mats;
    int32_t     reserved;  // keeps the tensor array 8B-aligned
    ggml_tensor src0[ET_MUL_MAT_GROUP_MAX];
    ggml_tensor src1;
    ggml_tensor dst[ET_MUL_MAT_GROUP_MAX];
    ggml_tensor bias[ET_MUL_MAT_GROUP_MAX];  // .data == nullptr means no residual bias
};

bool ggml_et_op_mul_mat_group(ggml_backend_et_device_context * dev_ctx,
                              const ggml_tensor * const *      nodes,
                              int                              n_mats);
