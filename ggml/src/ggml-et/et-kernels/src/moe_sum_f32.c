#include "ggml_tensor.h"
#include "platform.h"
#include <stdint.h>

//******************************************************************************
// MOE_SUM F32 kernel (backlog 038 — fused expert aggregation)
//   dst[e, t] = sum_{s < S} src0[e, s, t] * src1[0, s, t]
// replaces MUL(experts [E,S,N], weights [1,S,N]) + (S-1) view-ADDs per MoE layer
// (8 el_map_f32 launches, the slow strided path on 32 harts) with one launch.
// Work unit = one 64-byte line of dst (16 consecutive e for one token t), striped
// over all harts; every hart owns whole lines, so plain vector stores are safe
// (same publication contract as el_map_f32's cache-aligned fast path).
// The host only fuses when E % 16 == 0, dst rows are line-aligned and contiguous
// in e (nb0 == 4, nb1 % 64 == 0). dst may alias slot 0 of src0 (in-place ADD chain):
// each (e,t) is read for all s before it is written, by the same hart.
//******************************************************************************

int entry_point(struct ggml_et_binary_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    if (!kernel_env) {
        return -1;
    }
    int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) {
        return 0;
    }
    if (params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }
    struct ggml_tensor * src0 = &params->src0;  // experts [E, S, N]
    struct ggml_tensor * src1 = &params->src1;  // weights [1, S, N]
    struct ggml_tensor * dst  = &params->dst;   // [E, N]
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1;
    }
    const char * x_base = (const char *) src0->data;
    const char * w_base = (const char *) src1->data;
    char *       d_base = (char *) dst->data;
    if (!x_base || !w_base || !d_base) {
        return -1;
    }
    const int64_t E = src0->ne[0], S = src0->ne[1], N = src0->ne[2];
    if (E % 16 != 0 || src0->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || src1->ne[1] != S ||
        src1->ne[2] != N || dst->ne[0] != E || dst->ne[1] != N) {
        return -1;
    }
    const size_t nb01 = src0->nb[1], nb02 = src0->nb[2];   // slot, token strides of experts
    const size_t nb11 = src1->nb[1], nb12 = src1->nb[2];   // slot, token strides of weights
    const size_t nbd1 = dst->nb[1];

    const int64_t lines_per_row = E / 16;
    const int64_t total_lines   = N * lines_per_row;

    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int64_t line = thread_id; line < total_lines; line += num_threads) {
        const int64_t t  = line / lines_per_row;
        const int64_t e0 = (line % lines_per_row) * 16;
        const char *  x  = x_base + t * nb02 + e0 * sizeof(float);
        const char *  w  = w_base + t * nb12;
        float *       d  = (float *) (d_base + t * nbd1 + e0 * sizeof(float));
        float zero = 0.0f;
        __asm__ volatile("fbc.ps f10, %[z]\n fbc.ps f11, %[z]\n" : : [z] "m"(zero) : "f10", "f11");
        for (int64_t s = 0; s < S; ++s) {
            const float * xs = (const float *) (x + s * nb01);
            const float * ws = (const float *) (w + s * nb11);
            __asm__ volatile(
                "fbc.ps  f13, %[wv]\n"
                "flw.ps  f12, %[x0]\n"
                "fmul.ps f12, f12, f13\n"
                "fadd.ps f10, f10, f12\n"
                "flw.ps  f14, %[x1]\n"
                "fmul.ps f14, f14, f13\n"
                "fadd.ps f11, f11, f14\n"
                :
                : [wv] "m"(*ws), [x0] "m"(*(const float (*)[8]) xs), [x1] "m"(*(const float (*)[8]) (xs + 8))
                : "f10", "f11", "f12", "f13", "f14");
        }
        __asm__ volatile(
            "fsw.ps f10, %[d0]\n"
            "fsw.ps f11, %[d1]\n"
            : [d0] "=m"(*(float (*)[8]) d), [d1] "=m"(*(float (*)[8]) (d + 8))
            :
            : "memory");
    }
    __asm__ volatile("mova.m.x %0" ::"r"(temp_mask));
    FENCE;
    return 0;
}
