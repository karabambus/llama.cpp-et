#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"

#include <stdint.h>

//******************************************************************************
// MOE_ROUTER F32 kernel (T2 fusion 1 - the llama.cpp MoE routing chain)
//   probs   = softmax(logits[:, t])                       (SOFT_MAX, scale 1, no mask)
//   ids     = top-K indices of probs, descending, ties by index   (ARGSORT DESC + top_k VIEW)
//   w[j]    = probs[ids[j]]                                (GET_ROWS)
//   s       = clamp(sum_j w[j], min, max)                  (SUM_ROWS + CLAMP)
//   dst[j]  = w[j] / s                                     (DIV)
// One token per work unit; six launches per layer become one. Only the tensors
// the rest of the graph reads are written: the first K entries of every row of
// the ARGSORT output (the top_k view feeds MUL_MAT_ID / GET_ROWS through nb1)
// and the normalised weights [K, N]. Both rows are shorter than a cache line
// (K*4 bytes), so neighbouring tokens share lines -> every element is published
// with a global atomic swap, never a plain store.
// Numerics follow the per-op kernels: softmax = fexp.ps(log2e*(x-max)) * frcp(sum)
// (softmax_f32), the final division = frcp + two Newton steps (el_map_f32 DIV).
//******************************************************************************

#define ROUTER_MAX_E 512
#define ROUTER_MAX_K 32
#define ROUTER_LOG2E 1.4426950408889634f

struct ggml_et_moe_router_params {
    struct ggml_tensor src0;       // logits [E, N] F32
    struct ggml_tensor ids;        // ARGSORT output [E, N] I32 - top-K of each row written
    struct ggml_tensor dst;        // normalised weights [K, N] F32
    float              clamp_min;  // CLAMP bounds for the weight sum
    float              clamp_max;
    int32_t            n_top;      // K
    int32_t            reserved;
};

// 1/x with the el_map_f32 DIV recipe (frcp.ps seed + two Newton steps).
static inline float router_recip(float b) {
    float         r;
    unsigned long temp;
    __asm__ volatile(
        "mova.x.m  %[temp]              \n\t"
        "mov.m.x   m0, x0, 1            \n\t"
        "frcp.ps   %[r], %[b]           \n\t"
        "mova.m.x  %[temp]              \n\t"
        : [temp] "=&r"(temp), [r] "=&f"(r)
        : [b] "f"(b));
    r = r * (2.0f - b * r);
    r = r * (2.0f - b * r);
    return r;
}

int entry_point(struct ggml_et_moe_router_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    if (!kernel_env) {
        return -1;
    }
    const int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    const int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) {
        return 0;
    }
    if (params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }
    const struct ggml_tensor * src0 = &params->src0;
    const struct ggml_tensor * ids  = &params->ids;
    const struct ggml_tensor * dst  = &params->dst;
    if (src0->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        return -1;
    }
    const char * x_base = (const char *) src0->data;
    char *       i_base = (char *) ids->data;
    char *       d_base = (char *) dst->data;
    if (!x_base || !i_base || !d_base) {
        return -1;
    }
    const int64_t E = src0->ne[0];
    const int64_t N = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const int32_t K = params->n_top;
    if (E < 1 || E > ROUTER_MAX_E || K < 1 || K > ROUTER_MAX_K || K > E || src0->nb[0] != sizeof(float) ||
        ids->nb[0] != sizeof(int32_t) || dst->nb[0] != sizeof(float) || dst->ne[0] != K || ids->ne[0] != E ||
        src0->ne[2] != 1 || src0->ne[3] != 1) {
        return -1;
    }
    const size_t nb01 = src0->nb[1], nbi1 = ids->nb[1], nbd1 = dst->nb[1];
    const float  clamp_min = params->clamp_min, clamp_max = params->clamp_max;

    // hart 0 of every minion first (both harts of a minion share one pipeline)
    const int64_t half = num_threads / 2;
    const int64_t slot = (thread_id & 1) ? (half + (thread_id >> 1)) : (thread_id >> 1);

    // per-hart probability scratch in a private L2 SCP slice (64 harts x 2 KiB), off the U-mode stack
    const uint64_t hart_in_shire = get_hart_id() & 63;
    float *        p             = (float *) et_shire_l2scp_local(hart_in_shire * (ROUTER_MAX_E * sizeof(float)));
    const float   neg_inf = -__builtin_inff();
    const float   log2e   = ROUTER_LOG2E;
    const int32_t vec_end = (int32_t) ((E / 8) * 8);

    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int64_t t = slot; t < N; t += num_threads) {
        const float * x = (const float *) (x_base + t * nb01);

        // 1. row max (vector lanes, then horizontal)
        float mx;
        __asm__ volatile("fbc.ps f10, %[z]\n" : : [z] "m"(neg_inf) : "f10");
        for (int32_t i = 0; i < vec_end; i += 8) {
            __asm__ volatile(
                "flw.ps  f11, %[xv]\n"
                "fmax.ps f10, f10, f11\n"
                :
                : [xv] "m"(*(const float (*)[8]) & x[i])
                : "f10", "f11");
        }
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fmax.ps   f2, f10, f1 \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fmax.ps   f4, f2, f3 \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fmax.ps   %[vout], f4, f5 \n\t"
            : [vout] "=f"(mx)::"t0", "f1", "f2", "f3", "f4", "f5");
        for (int32_t i = vec_end; i < (int32_t) E; ++i) {
            if (x[i] > mx) {
                mx = x[i];
            }
        }

        // 2. e_i = 2^((x_i - max) * log2e), sum
        float sum;
        {
            const float nml2 = -mx * log2e;
            __asm__ volatile(
                "fbc.ps f12, %[l2e]\n"
                "fbc.ps f13, %[nml]\n"
                "fbci.pi f10, 0\n"
                :
                : [l2e] "m"(log2e), [nml] "m"(nml2)
                : "f10", "f12", "f13");
            for (int32_t i = 0; i < vec_end; i += 8) {
                __asm__ volatile(
                    "flw.ps   f11, %[xv]\n"
                    "fmadd.ps f11, f11, f12, f13\n"
                    "fexp.ps  f11, f11\n"
                    "fsw.ps   f11, %[pv]\n"
                    "fadd.ps  f10, f10, f11\n"
                    : [pv] "=m"(*(float (*)[8]) & p[i])
                    : [xv] "m"(*(const float (*)[8]) & x[i])
                    : "f10", "f11");
            }
            __asm__ __volatile__(
                "fswizz.ps f1, f10, 0xB1 \n\t"
                "fadd.ps   f2, f10, f1, rne \n\t"
                "fswizz.ps f3, f2, 0x4E \n\t"
                "fadd.ps   f4, f2, f3, rne \n\t"
                "fmvz.x.ps t0, f4, 4 \n\t"
                "fbcx.ps   f5, t0 \n\t"
                "fadd.ps   %[vout], f4, f5, rne \n\t"
                : [vout] "=f"(sum)::"t0", "f1", "f2", "f3", "f4", "f5");
            for (int32_t i = vec_end; i < (int32_t) E; ++i) {
                p[i] = et_exp2f((x[i] - mx) * log2e);
                sum += p[i];
            }
        }

        // 3. probs = e * frcp(sum)   (softmax_f32 pass 2)
        {
            const float inv_sum = et_fdiv(1.0f, sum);
            __asm__ volatile("fbc.ps f13, %[iv]\n" : : [iv] "m"(inv_sum) : "f13");
            for (int32_t i = 0; i < vec_end; i += 8) {
                __asm__ volatile(
                    "flw.ps  f11, %[pv]\n"
                    "fmul.ps f11, f11, f13\n"
                    "fsw.ps  f11, %[pv]\n"
                    : [pv] "+m"(*(float (*)[8]) & p[i])
                    :
                    : "f11");
            }
            for (int32_t i = vec_end; i < (int32_t) E; ++i) {
                p[i] *= inv_sum;
            }
        }

        // 4. top-K (descending, ties -> smaller index), then weight normalisation
        int32_t sel[ROUTER_MAX_K];
        float   w[ROUTER_MAX_K];
        float   wsum = 0.0f;
        for (int32_t j = 0; j < K; ++j) {
            int32_t best   = -1;
            float   best_v = neg_inf;
            for (int32_t i = 0; i < (int32_t) E; ++i) {
                const float v = p[i];
                if (v > best_v || best < 0) {
                    best_v = v;
                    best   = i;
                }
            }
            sel[j]  = best;
            w[j]    = best_v;
            p[best] = neg_inf;
            wsum += best_v;
        }
        if (wsum < clamp_min) {
            wsum = clamp_min;
        }
        if (wsum > clamp_max) {
            wsum = clamp_max;
        }
        const float inv = router_recip(wsum);

        // 5. publish (rows shorter than a line -> atomics)
        volatile int32_t * id_row = (volatile int32_t *) (i_base + t * nbi1);
        volatile float *   w_row  = (volatile float *) (d_base + t * nbd1);
        for (int32_t j = 0; j < K; ++j) {
            et_global_swap_w(id_row + j, (uint32_t) sel[j]);
            atomic_store_f32(w_row + j, w[j] * inv);
        }
    }

    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
#ifdef ET_UBERKERNEL
    // 052: see attn_qkv_glue_f32.c - evict this hart's private L2-SCP scratch from L1 before the next instruction
    FENCE;
    for (int64_t off = 0; off < (int64_t) (ROUTER_MAX_E * sizeof(float)); off += 16 * 64) {
        evict_to_l2((const char *) p + off, 16, 64);
    }
    WAIT_CACHEOPS;
#endif
    FENCE;
    return 0;
}
