#include "ggml-et-uberkernel-common.h"
#include "ggml-et-uberkernel-kernel-map.h"
#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"

#include <stdint.h>

struct ggml_et_glu_params;
struct ggml_et_unary_params;
struct ggml_et_rope_params;
struct ggml_et_rms_norm_params;
struct ggml_et_rms_norm_mul_params;
struct ggml_et_softmax_params;
struct ggml_et_set_rows_params;
struct ggml_et_get_rows_params;
struct ggml_et_cont_params;
struct ggml_et_concat_params;
struct ggml_et_cumsum_params;
struct ggml_et_diag_params;
struct ggml_et_fill_params;
struct ggml_et_flash_attn_ext_params;
struct ggml_et_gated_delta_net_params;
struct ggml_et_group_norm_params;
struct ggml_et_im2col_params;
struct ggml_et_l2_norm_params;
struct ggml_et_mul_mat_id_params;
struct ggml_et_norm_params;
struct ggml_et_pad_params;
struct ggml_et_repeat_params;
struct ggml_et_rwkv_wkv6_params;
struct ggml_et_rwkv_wkv7_params;
struct ggml_et_scale_params;
struct ggml_et_set_params;
struct ggml_et_solve_tri_params;
struct ggml_et_sqr_params;
struct ggml_et_ssm_conv_params;
struct ggml_et_ssm_scan_params;
struct ggml_et_sum_rows_params;
struct ggml_et_tri_params;

extern int el_map_f32_entry(struct ggml_et_binary_params *, void *);
extern int glu_f32_entry(struct ggml_et_glu_params *, void *);
extern int unary_f32_entry(struct ggml_et_unary_params *, void *);
extern int rope_f32_entry(struct ggml_et_rope_params *, void *);
extern int rms_norm_f32_entry(struct ggml_et_rms_norm_params *, void *);
extern int rms_norm_mul_f32_entry(struct ggml_et_rms_norm_mul_params *, void *);
extern int softmax_f32_entry(struct ggml_et_softmax_params *, void *);
extern int set_rows_f32_entry(struct ggml_et_set_rows_params *, void *);
extern int get_rows_f32_entry(struct ggml_et_get_rows_params *, void *);
extern int cont_f32_entry(struct ggml_et_cont_params *, void *);
extern int cont_f16_entry(struct ggml_et_cont_params *, void *);
extern int cpy_f32_f16_entry(struct ggml_et_cont_params *, void *);
extern int concat_f32_entry(struct ggml_et_concat_params *, void *);
extern int cumsum_f32_entry(struct ggml_et_cumsum_params *, void *);
extern int diag_f32_entry(struct ggml_et_diag_params *, void *);
extern int fill_f32_entry(struct ggml_et_fill_params *, void *);
extern int flash_attn_ext_f32_entry(struct ggml_et_flash_attn_ext_params *, void *);
extern int flash_attn_ext_f16_me_entry(struct ggml_et_flash_attn_ext_params *, void *);
extern int gated_delta_net_f32_entry(struct ggml_et_gated_delta_net_params *, void *);
extern int group_norm_f32_entry(struct ggml_et_group_norm_params *, void *);
extern int im2col_entry(struct ggml_et_im2col_params *, void *);
extern int l2_norm_f32_entry(struct ggml_et_l2_norm_params *, void *);
extern int mul_mat_id_f32_entry(struct ggml_et_mul_mat_id_params *, void *);
extern int norm_f32_entry(struct ggml_et_norm_params *, void *);
extern int pad_f32_entry(struct ggml_et_pad_params *, void *);
extern int repeat_f32_entry(struct ggml_et_repeat_params *, void *);
extern int rwkv_wkv6_f32_entry(struct ggml_et_rwkv_wkv6_params *, void *);
extern int rwkv_wkv7_f32_entry(struct ggml_et_rwkv_wkv7_params *, void *);
extern int scale_f32_entry(struct ggml_et_scale_params *, void *);
extern int set_f32_entry(struct ggml_et_set_params *, void *);
extern int solve_tri_f32_entry(struct ggml_et_solve_tri_params *, void *);
extern int sqr_f32_entry(struct ggml_et_sqr_params *, void *);
extern int ssm_conv_f32_entry(struct ggml_et_ssm_conv_params *, void *);
extern int ssm_scan_f32_entry(struct ggml_et_ssm_scan_params *, void *);
extern int sum_rows_f32_entry(struct ggml_et_sum_rows_params *, void *);
extern int tri_f32_entry(struct ggml_et_tri_params *, void *);
extern int mul_mat_f16_entry(struct ggml_et_binary_params *, void *);
extern int mul_mat_f16_matrix_engine_entry(struct ggml_et_binary_params *, void *);
extern int mul_mat_f32_entry(struct ggml_et_binary_params *, void *);
extern int mul_mat_f32_matrix_engine_entry(struct ggml_et_binary_params *, void *);
extern int mul_mat_Q8_0_entry(struct ggml_et_mm_q8_params *, void *);
extern int mul_mat_PQ2_0_entry(struct ggml_et_mm_q8_params *, void *);  // same bias-capable struct as Q8_0
struct ggml_et_hadamard_params;
extern int hadamard_f32_entry(struct ggml_et_hadamard_params *, void *);
struct ggml_et_mul_mat_group_params;
extern int mul_mat_Q8_0_group_entry(struct ggml_et_mul_mat_group_params *, void *);
extern int mul_mat_Q4_0_entry(struct ggml_et_mm_q8_params *, void *);  // T2: bias-capable params (prefix-compatible)
extern int mul_mat_Q4_K_entry(struct ggml_et_binary_params *, void *);
extern int mul_mat_Q2_K_entry(struct ggml_et_binary_params *, void *);
extern int mul_mat_Q3_K_entry(struct ggml_et_binary_params *, void *);
extern int mul_mat_Q5_K_entry(struct ggml_et_binary_params *, void *);
extern int mul_mat_Q6_K_entry(struct ggml_et_binary_params *, void *);
// 052: Qwen3-MoE op set
struct ggml_et_moe_router_params;
struct ggml_et_moe_sum_params;
struct ggml_et_qkv_glue_params;
extern int mul_mat_id_Q4_0_entry(struct ggml_et_mul_mat_id_params *, void *);
extern int mul_mat_id_Q4_0_v0_entry(struct ggml_et_mul_mat_id_params *, void *);
extern int moe_router_f32_entry(struct ggml_et_moe_router_params *, void *);
extern int attn_qkv_glue_f32_entry(struct ggml_et_qkv_glue_params *, void *);
extern int moe_sum_f32_entry(struct ggml_et_moe_sum_params *, void *);
extern int mul_mat_Q4_0_group_entry(struct ggml_et_mul_mat_group_params *, void *);

static inline size_t tensor_bytes(const struct ggml_tensor * t) {
    return (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

// 052: byte span of a possibly strided/permuted view (what a consumer may touch), not the packed size.
static inline size_t tensor_span_bytes(const struct ggml_tensor * t) {
    size_t span = t->nb[0];
    for (int i = 0; i < 4; ++i) {
        if (t->ne[i] > 1) {
            span += (size_t) (t->ne[i] - 1) * t->nb[i];
        }
    }
    return span;
}

// 052 eviction modes (ggml_et_uberkernel_params.flags bit 0, host env GGML_ET_UK_EVICT_MODE):
//   0: every hart evicts every input region from its own L1+L2 (Saqib recipe; redundant 64x per shire - measured
//      +195 ms/forward at N=32 where the spans reach 2 MB).
//   1: every hart first evicts its WHOLE L1 to L2 with the M-mode syscall (cache_ops_priv_evict_l1: dirty outputs
//      and private SCP scratch land in L2/SCP), then the L2->L3 walk of each region is split across the 64 harts
//      of the shire (line h, h+64, ... via the 16-line/stride form), so each line is evicted once per shire.
//      First board run 2026-08-27: N=1 83.8 ms, N=32 760.9 ms, but PPL 10.4845 vs 10.4855 (a race between one hart's
//      L1 evict and another hart's L2 walk of the same line) -> shire barrier between the two steps added.
static uint32_t uk_mode;  // set by every hart at entry from params->flags (same value everywhere)
// Timing-attribution knobs (host env GGML_ET_UK_TIMING_FLAGS, never set by default): bit 1 skips every per-instruction
// evict (results are then wrong across shires - bench timing only), bit 2 takes the chip-wide barrier twice per
// instruction (safe: every hart does it) so the delta against a plain run is the barrier's cost.
static uint32_t uk_flags_all;

static inline void uk_evict_region(const void * addr, size_t bytes) {
    if (!addr || bytes == 0 || (uk_flags_all & 2u)) {
        return;
    }
    if (uk_mode == 0) {
        evict_region_past_l2(addr, bytes);
        return;
    }
    const uint64_t CL     = 64;
    const uint64_t base   = (uint64_t) addr & ~(CL - 1);
    const uint64_t end    = ((uint64_t) addr + bytes + CL - 1) & ~(CL - 1);
    const uint64_t nlines = (end - base) / CL;
    const uint64_t h      = get_hart_id() & 63;  // this hart's share: lines h, h+64, h+128, ...
    for (uint64_t off = h; off < nlines; off += 16 * 64) {
        uint64_t batch = (nlines - off + 63) / 64;  // lines left in this hart's share
        if (batch > 16) {
            batch = 16;
        }
        evict_past_l2((const void *) (base + off * CL), batch, 64 * CL);  // 16 lines, stride 64 lines
    }
}

// Evict a tensor's span past L2 when it has data (writable inputs only - weights are never evicted, they are
// read-only within a segment and enormous).
static inline void uk_evict(const struct ggml_tensor * t) {
    if (t->data) {
        uk_evict_region(t->data, tensor_span_bytes(t));
    }
}

// 052: evict the last (256 + n_batch) positions of a KV-cache view [D, n_kv, n_head_kv] (positions = ne[1]).
// Positions are the outer stride in llama.cpp's cache view (nb[1] >= ne[2]*nb[2]); otherwise fall back to the span.
static inline void uk_evict_kv_tail(const struct ggml_tensor * t, int64_t n_batch) {
    if (!t->data) {
        return;
    }
    const int64_t n_kv = t->ne[1];
    if ((size_t) t->nb[1] >= (size_t) t->ne[2] * t->nb[2] && n_kv > 0) {
        int64_t r0 = n_kv - 256 - n_batch;
        if (r0 < 0) {
            r0 = 0;
        }
        uk_evict_region((const char *) t->data + (size_t) r0 * t->nb[1], (size_t) (n_kv - r0) * t->nb[1]);
    } else {
        uk_evict_region(t->data, tensor_span_bytes(t));
    }
}

// 052 mirrors of the MoE kernels' parameter structs (layouts must match the kernel sources exactly)
struct uber_moe_router_params {
    struct ggml_tensor src0;  // logits (written by the router mul_mat)
    struct ggml_tensor ids;   // written here
    struct ggml_tensor dst;   // written here
    float              clamp_min, clamp_max;
    int32_t            n_top, reserved;
};
struct uber_moe_sum_params {
    struct ggml_tensor src0;  // experts (mul_mat_id output)
    struct ggml_tensor src1;  // router weights
    struct ggml_tensor dst;
    struct ggml_tensor res;   // optional residual (.data == 0: none)
};
#define UBER_MUL_MAT_GROUP_MAX 3
struct uber_mul_mat_group_params {
    int32_t            n_mats, reserved;
    struct ggml_tensor src0[UBER_MUL_MAT_GROUP_MAX];  // weights
    struct ggml_tensor src1;                          // shared activation
    struct ggml_tensor dst[UBER_MUL_MAT_GROUP_MAX];
    struct ggml_tensor bias[UBER_MUL_MAT_GROUP_MAX];  // .data == 0: none
};
struct uber_qkv_job {
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
#define UBER_GLUE_MAX_JOBS 3
struct uber_qkv_glue_params {
    struct uber_qkv_job job[UBER_GLUE_MAX_JOBS];  // rope settings + n_jobs follow; not needed for eviction
};

struct uber_glu_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
    // trailing scalars omitted — not needed for eviction
};

struct uber_unary_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};

struct uber_rope_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor dst;
};

struct uber_rms_norm_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};

struct uber_rms_norm_mul_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct uber_softmax_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor dst;
};

struct uber_set_rows_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct uber_get_rows_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

// hadamard_f32: src (activation, writable), signs (weight-like), dst; trailing scalars omitted
struct uber_hadamard_params {
    struct ggml_tensor src;
    struct ggml_tensor signs;
    struct ggml_tensor dst;
};
// Generic prefixes for kernels whose params start with their tensors (device-side layouts checked per kernel):
// src0+dst: l2_norm, scale, cumsum, tri, sum_rows, sqr, norm, pad, repeat, group_norm, diag; src0+src1+dst: solve_tri, set_rows,
// get_rows, concat; src0+src1+src2+dst: softmax, rope.
struct uber_src_dst { struct ggml_tensor src0; struct ggml_tensor dst; };
struct uber_src2_dst { struct ggml_tensor src0; struct ggml_tensor src1; struct ggml_tensor dst; };
struct uber_src3_dst { struct ggml_tensor src0; struct ggml_tensor src1; struct ggml_tensor src2; struct ggml_tensor dst; };
struct uber_set_params { struct ggml_tensor src1; struct ggml_tensor dst; };  // SET writes into dst in place
struct uber_cont_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};

// src0 + src1 + dst (no trailing scalars needed for eviction)
struct uber_concat_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct uber_ssm_conv_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct uber_solve_tri_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct uber_mul_mat_id_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor dst;
    uint64_t           shire_mask;  // mirrors ggml_et_mul_mat_id_params (0 = kernel_env->shire_mask)
};

// flash_attn_ext: Q=src0, K=src1, V=src2, mask=src3, dst (mask optional)
struct uber_flash_attn_ext_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor mask;
    struct ggml_tensor dst;
};

// ssm_scan: 7 source tensors + dst
struct uber_ssm_scan_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor src3;
    struct ggml_tensor src4;
    struct ggml_tensor src5;
    struct ggml_tensor src6;
    struct ggml_tensor dst;
};

// gated_delta_net: q,k,v,g,beta,state_in,dst,rows,dt_bias,a (rows / dt_bias / a have data == NULL when
// the rows-mode / raw-gates form is off; dt_bias and a are weights)
struct uber_gated_delta_net_params {
    struct ggml_tensor q;
    struct ggml_tensor k;
    struct ggml_tensor v;
    struct ggml_tensor g;
    struct ggml_tensor beta;
    struct ggml_tensor state_in;
    struct ggml_tensor dst;
    struct ggml_tensor rows;
    struct ggml_tensor dt_bias;
    struct ggml_tensor a;
};

static void copy_f32_to_f16_row(uint16_t * dst, const float * src, int64_t num_elements) {
    for (int64_t i = 0; i < num_elements; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

static void copy_f32_row(float * dst, const float * src, int64_t num_elements) {
    for (int64_t i = 0; i < num_elements; i++) {
        dst[i] = src[i];
    }
}

static void evict_region_past_l2_local(const void * addr, size_t bytes) {
    if (!addr || bytes == 0) {
        return;
    }

    const uint64_t CL     = 64;
    uint64_t       base   = (uint64_t) addr & ~(CL - 1);
    uint64_t       end    = ((uint64_t) addr + bytes + CL - 1) & ~(CL - 1);
    uint64_t       nlines = (end - base) / CL;
    cache_ops_priv_evict_sw(0, /*to_L2*/ 3, 0, 0, CL);
}

int entry_point(struct ggml_et_uberkernel_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env || !params) {
        return -1;
    }

    struct ggml_et_uberkernel_inst * insts       = (struct ggml_et_uberkernel_inst *) (uintptr_t) params->insts;
    uint8_t *                        params_blob = (uint8_t *) (uintptr_t) params->params_blob;

    if (!insts || !params_blob || params->inst_stride < sizeof(struct ggml_et_uberkernel_inst)) {
        return -1;
    }

    // 052 per-instruction publication contract (SaqibAkram-10xE n_way recipe, backlog 052): L1 and L2 are
    // incoherent (L2 is per shire), so before every instruction EVERY hart (1) drains its store buffer, (2) evicts
    // the instruction's *writable* inputs from its own L1+L2 to L3/DRAM - on the producing shires this writes back
    // what they computed, on the consuming shires it drops stale copies of a reused buffer - (3) waits for its
    // cache ops, and (4) takes the chip-wide barrier, so no shire reads before every shire has published.
    // Weights are never evicted (read-only in a segment). UK_SYNC() is steps 3+4, called by every case after its
    // evictions; the barrier count is 32 = every launch is chip-wide (the host refuses to batch a narrowed one).
    // A hart whose instruction failed keeps taking the barriers (it must never leave the loop while the others
    // still wait for it) and reports the first error at the end.
#define UK_SYNC()                       \
    do {                                \
        WAIT_CACHEOPS;                  \
        et_barrier_global(32ULL);       \
        if (uk_flags_all & 4u) {        \
            et_barrier_global(32ULL);   \
        }                               \
    } while (0)
    int failed   = 0;
    uk_mode      = params->flags & 1u;
    uk_flags_all = params->flags;
    for (uint32_t i = 0; i < params->num_insts; ++i) {
        struct ggml_et_uberkernel_inst * inst =
            (struct ggml_et_uberkernel_inst *) ((uint8_t *) insts + (i * params->inst_stride));
        void * inst_params = params_blob + inst->params_offset;
        int    rc          = -1;

        FENCE;
        if (uk_mode == 1 && !(uk_flags_all & 2u)) {
            cache_ops_priv_evict_l1(0, to_L2);  // whole L1 -> L2 (dirty outputs + SCP scratch), M-mode syscall
            // The L2->L3 walk below is split across the shire's harts, so a hart may push a line another hart of
            // this shire still holds dirty in L1: wait for every hart's L1 evict to land in L2 first. Without this
            // shire barrier the 32-chunk PPL moved 10.4855 -> 10.4845 (2026-08-27, first mode-1 board run).
            WAIT_CACHEOPS;
            et_barrier(ET_BARRIER_SHIRE);
        }
        if (failed) {
            UK_SYNC();
            continue;
        }

        switch (inst->kernel_id) {
            case GGML_ET_UBERKERNEL_KERNEL_EL_MAP_F32:
                {
                    struct ggml_et_binary_params * p = (struct ggml_et_binary_params *) inst_params;
                    uk_evict(&p->src0);
                    uk_evict(&p->src1);
                    UK_SYNC();
                    rc                               = el_map_f32_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_UNARY_F32:
                {
                    struct uber_unary_params * p = (struct uber_unary_params *) inst_params;
                    uk_evict(&p->src0);
                    UK_SYNC();
                    rc = unary_f32_entry((struct ggml_et_unary_params *) inst_params, env);
                    break;
                }
            // case GGML_ET_UBERKERNEL_KERNEL_CPY_F32_F16: {
            //     struct uber_unary_params *p = (struct uber_unary_params *) inst_params;
            //     // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
            //     rc = cpy_f32_f16_entry((struct ggml_et_cont_params *) inst_params, env);
            //     break;
            // }
            // case GGML_ET_UBERKERNEL_KERNEL_GET_ROWS_F32: {
            //     struct uber_get_rows_params *p = (struct uber_get_rows_params *) inst_params;
            //     rc = get_rows_f32_entry((struct ggml_et_get_rows_params *) inst_params, env);
            //     break;
            // }
            case GGML_ET_UBERKERNEL_KERNEL_CONT_F32:
                {
                    struct uber_cont_params * p = (struct uber_cont_params *) inst_params;
                    uk_evict(&p->src0);  // strided view: uk_evict walks the span, not the packed size
                    UK_SYNC();
                    rc = cont_f32_entry((struct ggml_et_cont_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_GLU_F32:
                {
                    { struct uber_glu_params * g = (struct uber_glu_params *) inst_params; uk_evict(&g->src0); uk_evict(&g->src1); }
                    UK_SYNC();
                    rc = glu_f32_entry((struct ggml_et_glu_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_ROPE_F32:
                {
                    struct uber_rope_params * p = (struct uber_rope_params *) inst_params;
                    uk_evict(&p->src0);
                    uk_evict(&p->src1);
                    uk_evict(&p->src2);
                    UK_SYNC();
                    rc = rope_f32_entry((struct ggml_et_rope_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32:
                {
                    struct uber_rms_norm_params * p = (struct uber_rms_norm_params *) inst_params;
                    uk_evict(&p->src0);
                    UK_SYNC();
                    rc = rms_norm_f32_entry((struct ggml_et_rms_norm_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32:
                {
                    struct uber_rms_norm_mul_params * p = (struct uber_rms_norm_mul_params *) inst_params;
                    uk_evict(&p->src0);
                    uk_evict(&p->src1);
                    UK_SYNC();
                    rc = rms_norm_mul_f32_entry((struct ggml_et_rms_norm_mul_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_SOFTMAX_F32:
                {
                    struct uber_softmax_params * p = (struct uber_softmax_params *) inst_params;
                    uk_evict(&p->src0);
                    uk_evict(&p->src1);
                    uk_evict(&p->src2);
                    UK_SYNC();
                    rc = softmax_f32_entry((struct ggml_et_softmax_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_SET_ROWS_F32:
                {
                    struct uber_set_rows_params * p = (struct uber_set_rows_params *) inst_params;
                    uk_evict(&p->src0);
                    uk_evict(&p->src1);
                    UK_SYNC();
                    rc = set_rows_f32_entry((struct ggml_et_set_rows_params *) inst_params, env);
                    break;
                }

            // Single-source ops (src0 → dst)
            case GGML_ET_UBERKERNEL_KERNEL_SQR_F32:
                {
                    struct uber_src_dst * p = (struct uber_src_dst *) inst_params;
                    uk_evict(&p->src0);
                    UK_SYNC();
                    rc = sqr_f32_entry((struct ggml_et_sqr_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_SCALE_F32:
                {
                    struct uber_src_dst * p = (struct uber_src_dst *) inst_params;
                    uk_evict(&p->src0);
                    UK_SYNC();
                    rc = scale_f32_entry((struct ggml_et_scale_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_SUM_ROWS_F32:
                {
                    struct uber_src_dst * p = (struct uber_src_dst *) inst_params;
                    uk_evict(&p->src0);
                    UK_SYNC();
                    rc = sum_rows_f32_entry((struct ggml_et_sum_rows_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_CUMSUM_F32:
                {
                    struct uber_src_dst * p = (struct uber_src_dst *) inst_params;
                    uk_evict(&p->src0);
                    UK_SYNC();
                    rc = cumsum_f32_entry((struct ggml_et_cumsum_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_NORM_F32:
                {
                    struct uber_src_dst * p = (struct uber_src_dst *) inst_params;
                    uk_evict(&p->src0);
                    UK_SYNC();
                    rc = norm_f32_entry((struct ggml_et_norm_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_L2_NORM_F32:
                {
                    struct uber_src_dst * p = (struct uber_src_dst *) inst_params;
                    uk_evict(&p->src0);
                    UK_SYNC();
                    rc = l2_norm_f32_entry((struct ggml_et_l2_norm_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_GROUP_NORM_F32:
                {
                    struct uber_src_dst * p = (struct uber_src_dst *) inst_params;
                    uk_evict(&p->src0);
                    UK_SYNC();
                    rc = group_norm_f32_entry((struct ggml_et_group_norm_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_REPEAT_F32:
                {
                    struct uber_src_dst * p = (struct uber_src_dst *) inst_params;
                    uk_evict(&p->src0);
                    UK_SYNC();
                    rc = repeat_f32_entry((struct ggml_et_repeat_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_DIAG_F32:
                {
                    struct uber_src_dst * p = (struct uber_src_dst *) inst_params;
                    uk_evict(&p->src0);
                    UK_SYNC();
                    rc = diag_f32_entry((struct ggml_et_diag_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_TRI_F32:
                {
                    struct uber_src_dst * p = (struct uber_src_dst *) inst_params;
                    uk_evict(&p->src0);
                    UK_SYNC();
                    rc = tri_f32_entry((struct ggml_et_tri_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_PAD_F32:
                {
                    struct uber_src_dst * p = (struct uber_src_dst *) inst_params;
                    uk_evict(&p->src0);
                    UK_SYNC();
                    rc = pad_f32_entry((struct ggml_et_pad_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_CONT_F16:
                {
                    struct uber_cont_params * p = (struct uber_cont_params *) inst_params;
                    uk_evict(&p->src0);
                    UK_SYNC();
                    rc = cont_f16_entry((struct ggml_et_cont_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_FILL_F32:
                {
                    UK_SYNC();
                    rc = fill_f32_entry((struct ggml_et_fill_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_SET_F32:
                {
                    struct uber_set_params * p = (struct uber_set_params *) inst_params;
                    uk_evict(&p->src1);
                    uk_evict(&p->dst);
                    UK_SYNC();
                    rc = set_f32_entry((struct ggml_et_set_params *) inst_params, env);
                    break;
                }

            // Two-source ops
            case GGML_ET_UBERKERNEL_KERNEL_CONCAT_F32:
                {
                    struct uber_concat_params * p = (struct uber_concat_params *) inst_params;
                    uk_evict(&p->src0);
                    uk_evict(&p->src1);
                    UK_SYNC();
                    rc = concat_f32_entry((struct ggml_et_concat_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_SSM_CONV_F32:
                {
                    struct uber_ssm_conv_params * p = (struct uber_ssm_conv_params *) inst_params;
                    uk_evict(&p->src0);  // conv window (concat of state and new inputs); src1 is the conv weight
                    UK_SYNC();
                    rc = ssm_conv_f32_entry((struct ggml_et_ssm_conv_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_SOLVE_TRI_F32:
                {
                    struct uber_solve_tri_params * p = (struct uber_solve_tri_params *) inst_params;
                    uk_evict(&p->src0);
                    uk_evict(&p->src1);
                    UK_SYNC();
                    rc = solve_tri_f32_entry((struct ggml_et_solve_tri_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_IM2COL:
                {
                    UK_SYNC();
                    rc = im2col_entry((struct ggml_et_im2col_params *) inst_params, env);
                    break;
                }

            // Three-source ops
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_F32:
                {
                    struct uber_mul_mat_id_params * p = (struct uber_mul_mat_id_params *) inst_params;
                    uk_evict(&p->src1);
                    uk_evict(&p->src2);
                    UK_SYNC();
                    rc = mul_mat_id_f32_entry((struct ggml_et_mul_mat_id_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F32:
                {
                    { struct uber_flash_attn_ext_params * f = (struct uber_flash_attn_ext_params *) inst_params; uk_evict(&f->src0); uk_evict(&f->src1); uk_evict(&f->src2); uk_evict(&f->mask); }
                    UK_SYNC();
                    rc = flash_attn_ext_f32_entry((struct ggml_et_flash_attn_ext_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F16_ME:
                {
                    {
                        struct uber_flash_attn_ext_params * f = (struct uber_flash_attn_ext_params *) inst_params;
                        uk_evict(&f->src0);
                        uk_evict(&f->mask);
                        // K/V cache: attn_qkv_glue wrote rows [kv_used-N, kv_used) earlier in this segment and this
                        // shire may hold stale copies of exactly those rows (it read them as padding, n_kv is padded to
                        // 256, in an earlier forward / prefetched them). Evict the tail that can contain them:
                        // rows [n_kv - 256 - N, n_kv) - bounded, independent of the context length.
                        uk_evict_kv_tail(&f->src1, f->src0.ne[1]);
                        uk_evict_kv_tail(&f->src2, f->src0.ne[1]);
                    }
                    UK_SYNC();
                    rc = flash_attn_ext_f16_me_entry((struct ggml_et_flash_attn_ext_params *) inst_params, env);
                    break;
                }

            case GGML_ET_UBERKERNEL_KERNEL_GATED_DELTA_NET_F32:
                {
                    struct uber_gated_delta_net_params * p = (struct uber_gated_delta_net_params *) inst_params;
                    uk_evict(&p->q);
                    uk_evict(&p->k);
                    uk_evict(&p->v);
                    uk_evict(&p->g);
                    uk_evict(&p->beta);
                    uk_evict(&p->state_in);
                    uk_evict(&p->rows);  // host-written row indices (no-op when data == NULL)
                    UK_SYNC();
                    rc = gated_delta_net_f32_entry((struct ggml_et_gated_delta_net_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_SSM_SCAN_F32:
                {
                    struct uber_ssm_scan_params * p = (struct uber_ssm_scan_params *) inst_params;
                    uk_evict(&p->src0);
                    uk_evict(&p->src1);
                    uk_evict(&p->src2);
                    uk_evict(&p->src3);
                    uk_evict(&p->src4);
                    uk_evict(&p->src5);
                    uk_evict(&p->src6);
                    UK_SYNC();
                    rc = ssm_scan_f32_entry((struct ggml_et_ssm_scan_params *) inst_params, env);
                    break;
                }
            // rwkv: raw float* params, no ggml_tensor fields to evict via
            case GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV6_F32:
                {
                    UK_SYNC();
                    rc = rwkv_wkv6_f32_entry((struct ggml_et_rwkv_wkv6_params *) inst_params, env);
                    break;
                }

            case GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV7_F32:
                {
                    UK_SYNC();
                    rc = rwkv_wkv7_f32_entry((struct ggml_et_rwkv_wkv7_params *) inst_params, env);
                    break;
                }

            // MUL_MAT: evict src1 (activations); src0=weights is
            //  read-only so never stale from a prior uberkernel op
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16:
                {
                    struct ggml_et_binary_params * p = (struct ggml_et_binary_params *) inst_params;
                    uk_evict(&p->src1);
                    UK_SYNC();
                    rc                               = mul_mat_f16_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16_MATRIX_ENGINE:
                {
                    struct ggml_et_binary_params * p = (struct ggml_et_binary_params *) inst_params;
                    uk_evict(&p->src1);
                    UK_SYNC();
                    rc                               = mul_mat_f16_matrix_engine_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32:
                {
                    struct ggml_et_binary_params * p = (struct ggml_et_binary_params *) inst_params;
                    uk_evict(&p->src1);
                    UK_SYNC();
                    rc                               = mul_mat_f32_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32_MATRIX_ENGINE:
                {
                    struct ggml_et_binary_params * p = (struct ggml_et_binary_params *) inst_params;
                    uk_evict(&p->src1);
                    UK_SYNC();
                    rc                               = mul_mat_f32_matrix_engine_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q8_0:
                {
                    struct ggml_et_mm_q8_params * p = (struct ggml_et_mm_q8_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    uk_evict(&p->src1);
                    uk_evict(&p->bias);
                    UK_SYNC();
                    rc                              = mul_mat_Q8_0_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_PQ2_0:
                {
                    struct ggml_et_mm_q8_params * p = (struct ggml_et_mm_q8_params *) inst_params;
                    uk_evict(&p->src1);  // activations; the PQ2_0 weights are read-only in a segment
                    uk_evict(&p->bias);
                    UK_SYNC();
                    rc = mul_mat_PQ2_0_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_HADAMARD_F32:
                {
                    struct uber_hadamard_params * p = (struct uber_hadamard_params *) inst_params;
                    uk_evict(&p->src);  // the activation; signs are weights
                    UK_SYNC();
                    rc = hadamard_f32_entry((struct ggml_et_hadamard_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q8_0_GROUP:
                {
                    // Independent Q8_0 mul_mats sharing one activation. Running them inside a
                    // single instruction is the point: the loop above executes a chip-wide
                    // barrier per instruction, and these members need none between them.
                    struct uber_mul_mat_group_params * p = (struct uber_mul_mat_group_params *) inst_params;
                    uk_evict(&p->src1);
                    for (int32_t g = 0; g < p->n_mats && g < UBER_MUL_MAT_GROUP_MAX; g++) {
                        uk_evict(&p->bias[g]);
                    }
                    UK_SYNC();
                    rc = mul_mat_Q8_0_group_entry((struct ggml_et_mul_mat_group_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q4_0:
                {
                    // T2: the host packs ggml_et_mm_q8_params (bias-capable) for the scalar Q4_0 kernel
                    struct ggml_et_mm_q8_params * p = (struct ggml_et_mm_q8_params *) inst_params;
                    uk_evict(&p->src1);
                    uk_evict(&p->bias);
                    UK_SYNC();
                    rc                              = mul_mat_Q4_0_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q4_K:
                {
                    struct ggml_et_binary_params * p = (struct ggml_et_binary_params *) inst_params;
                    uk_evict(&p->src1);
                    UK_SYNC();
                    rc                               = mul_mat_Q4_K_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q2_K:
                {
                    struct ggml_et_binary_params * p = (struct ggml_et_binary_params *) inst_params;
                    uk_evict(&p->src1);
                    UK_SYNC();
                    rc                               = mul_mat_Q2_K_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q3_K:
                {
                    struct ggml_et_binary_params * p = (struct ggml_et_binary_params *) inst_params;
                    uk_evict(&p->src1);
                    UK_SYNC();
                    rc                               = mul_mat_Q3_K_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q5_K:
                {
                    struct ggml_et_binary_params * p = (struct ggml_et_binary_params *) inst_params;
                    uk_evict(&p->src1);
                    UK_SYNC();
                    rc                               = mul_mat_Q5_K_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q6_K:
                {
                    struct ggml_et_binary_params * p = (struct ggml_et_binary_params *) inst_params;
                    uk_evict(&p->src1);
                    UK_SYNC();
                    rc                               = mul_mat_Q6_K_entry(p, env);
                    break;
                }


            // ---- 052: Qwen3-MoE op set ----
            case GGML_ET_UBERKERNEL_KERNEL_GET_ROWS_F32:
                {
                    // src0 is not always a weight: the linear-attention models gather the device-written recurrent
                    // state with GET_ROWS (build_rs), so the table span and the ids are evicted like any input
                    struct uber_get_rows_params * p = (struct uber_get_rows_params *) inst_params;
                    // Only a float table can be device-written; a block-quantised table is a weight, and its span
                    // computed from ne0*nb0 overshoots the allocation (nb0 is the block size), so it is not evicted.
                    if (p->src0.type == GGML_TYPE_F32 || p->src0.type == GGML_TYPE_F16) {
                        uk_evict(&p->src0);
                    }
                    uk_evict(&p->src1);
                    UK_SYNC();
                    rc = get_rows_f32_entry((struct ggml_et_get_rows_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_Q4_0:
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_Q4_0_V0:
                {
                    struct ggml_et_mul_mat_id_params * p = (struct ggml_et_mul_mat_id_params *) inst_params;
                    uk_evict(&p->src1);  // activations (ffn_norm / swiglu output)
                    uk_evict(&p->src2);  // expert ids (moe_router output)
                    UK_SYNC();
                    rc = (inst->kernel_id == GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_Q4_0) ? mul_mat_id_Q4_0_entry(p, env)
                                                                                      : mul_mat_id_Q4_0_v0_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MOE_ROUTER_F32:
                {
                    struct uber_moe_router_params * p = (struct uber_moe_router_params *) inst_params;
                    uk_evict(&p->src0);  // router logits (mul_mat output); ids/dst are written here
                    UK_SYNC();
                    rc = moe_router_f32_entry((struct ggml_et_moe_router_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_ATTN_QKV_GLUE_F32:
                {
                    struct uber_qkv_glue_params * p = (struct uber_qkv_glue_params *) inst_params;
                    for (int j = 0; j < UBER_GLUE_MAX_JOBS; ++j) {
                        const struct uber_qkv_job * job = &p->job[j];
                        if (job->src && job->N > 0 && job->H > 0) {  // the q/k/v projections
                            const size_t span = (size_t) (job->N - 1) * job->src_nb2 + (size_t) (job->H - 1) * job->src_nb1 +
                                                (size_t) job->D * sizeof(float);
                            uk_evict_region(job->src, span);
                        }
                    }
                    UK_SYNC();
                    rc = attn_qkv_glue_f32_entry((struct ggml_et_qkv_glue_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MOE_SUM_F32:
                {
                    struct uber_moe_sum_params * p = (struct uber_moe_sum_params *) inst_params;
                    uk_evict(&p->src0);  // expert outputs
                    uk_evict(&p->src1);  // router weights
                    uk_evict(&p->res);   // residual (attention output + residual)
                    UK_SYNC();
                    rc = moe_sum_f32_entry((struct ggml_et_moe_sum_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q4_0_GROUP:
                {
                    struct uber_mul_mat_group_params * p = (struct uber_mul_mat_group_params *) inst_params;
                    uk_evict(&p->src1);
                    for (int j = 0; j < UBER_MUL_MAT_GROUP_MAX && j < p->n_mats; ++j) {
                        uk_evict(&p->bias[j]);
                    }
                    UK_SYNC();
                    rc = mul_mat_Q4_0_group_entry((struct ggml_et_mul_mat_group_params *) inst_params, env);
                    break;
                }

            default:
                UK_SYNC();
                rc = -1;
                break;
        }

        if (rc != 0) {
            failed = rc;  // keep looping for the barriers; the launch reports the first failure
        }
    }

    return failed;
}
