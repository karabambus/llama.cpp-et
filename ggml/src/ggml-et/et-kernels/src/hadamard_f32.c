//******************************************************************************
// Blockwise normalised Walsh-Hadamard transform of F32 activations, with the optional sign flip folded in.
//
// Replaces the chain  MUL(x, signs) -> RESHAPE -> MUL_MAT(rotation [n x n], hint GGML_HINT_SRC0_IS_HADAMARD)  that the
// Prism runtime builds for every folded weight: dst[j][i] = FWHT_n(scale * signs[.] * x[.])[i], scale = 1/sqrt(n), one
// transform per n-element block of the activation. The rotation matrix is never read: the hint says it is the
// Sylvester-Hadamard matrix, which the butterflies compute exactly (the CPU backend does the same, ggml-cpu.c).
//
// Mapping: dst is [n, R] (R rows, each one block). Row j takes block b = j % (W/n) of activation row r = j / (W/n),
// where W = x->ne[0] is the activation width (W % n == 0); signs, when present, are indexed by position in the width.
// Each hart transforms whole rows, in place on dst: pass 0 writes scale * signs * x into the dst row, then the
// log2(n) butterfly stages run on that row. The row lives in the hart's L1D, so no scratchpad and no cross-hart
// traffic; the final values leave through plain fsw.ps / fscw.ps stores like every elementwise kernel here.
//
// Stages with distance >= 8 are contiguous 8-lane butterflies. Stages with distance 1, 2 and 4 pair elements inside
// one 8-lane vector, so they run on 16-element groups through word gathers/scatters whose lane patterns pick the
// "u" and "w" halves of each pair: e.g. distance 1 gathers elements {0,2,4,...,14} and {1,3,5,...,15}. n must be a
// power of two with 16 <= n <= 4096 (the host checks).
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"

#include <stdint.h>

struct ggml_et_hadamard_params {
    struct ggml_tensor src;    // x: F32, dim 0 contiguous, ne0 = W
    struct ggml_tensor signs;  // F32 [W], or data == NULL for no sign flip
    struct ggml_tensor dst;    // F32 [n, R], dim 0 contiguous
    int64_t            width;  // W
    int64_t            blk;    // n
    float              scale;  // 1/sqrt(n), computed on the host (no sqrt on the device)
};

// Byte-offset lane patterns for the three in-vector stages, over a 16-element group.
static const int32_t __attribute__((aligned(64))) pat_u1[8] = { 0, 8, 16, 24, 32, 40, 48, 56 };
static const int32_t __attribute__((aligned(64))) pat_w1[8] = { 4, 12, 20, 28, 36, 44, 52, 60 };
static const int32_t __attribute__((aligned(64))) pat_u2[8] = { 0, 4, 16, 20, 32, 36, 48, 52 };
static const int32_t __attribute__((aligned(64))) pat_w2[8] = { 8, 12, 24, 28, 40, 44, 56, 60 };
static const int32_t __attribute__((aligned(64))) pat_u4[8] = { 0, 4, 8, 12, 32, 36, 40, 44 };
static const int32_t __attribute__((aligned(64))) pat_w4[8] = { 16, 20, 24, 28, 48, 52, 56, 60 };

// One in-vector stage over the whole row: patterns in f28 (u) and f29 (w).
static inline void __attribute__((always_inline)) stage_gather(float * row, int64_t n, const int32_t * pu,
                                                                 const int32_t * pw) {
    __asm__ volatile(
        "flw.ps f28, %[u]\n\t"
        "flw.ps f29, %[w]\n\t"
        :
        : [u] "m"(*(const int32_t (*)[8]) pu), [w] "m"(*(const int32_t (*)[8]) pw)
        : "f28", "f29");
    for (int64_t g = 0; g < n; g += 16) {
        float * base = row + g;
        __asm__ volatile(
            "fgw.ps  f10, f28(%[b])\n\t"
            "fgw.ps  f11, f29(%[b])\n\t"
            "fadd.ps f12, f10, f11, rne\n\t"
            "fsub.ps f13, f10, f11, rne\n\t"
            "fscw.ps f12, f28(%[b])\n\t"
            "fscw.ps f13, f29(%[b])\n\t"
            :
            : [b] "r"(base)
            : "f10", "f11", "f12", "f13", "memory");
    }
}

// One contiguous stage (distance len >= 8) over the whole row.
static inline void __attribute__((always_inline)) stage_vector(float * row, int64_t n, int64_t len) {
    for (int64_t i = 0; i < n; i += 2 * len) {
        for (int64_t k = 0; k < len; k += 8) {
            float * pu = row + i + k;
            float * pw = pu + len;
            __asm__ volatile(
                "flw.ps  f10, 0(%[u])\n\t"
                "flw.ps  f11, 0(%[w])\n\t"
                "fadd.ps f12, f10, f11, rne\n\t"
                "fsub.ps f13, f10, f11, rne\n\t"
                "fsw.ps  f12, 0(%[u])\n\t"
                "fsw.ps  f13, 0(%[w])\n\t"
                :
                : [u] "r"(pu), [w] "r"(pw)
                : "f10", "f11", "f12", "f13", "memory");
        }
    }
}

int entry_point(struct ggml_et_hadamard_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    if (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }
    const int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    const int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) {
        return 0;
    }

    const struct ggml_tensor * src   = &params->src;
    const struct ggml_tensor * signs = &params->signs;
    const struct ggml_tensor * dst   = &params->dst;

    const int64_t n = params->blk;
    const int64_t W = params->width;
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src->nb[0] != sizeof(float) ||
        dst->nb[0] != sizeof(float) || n < 16 || n > 4096 || (n & (n - 1)) != 0 || W % n != 0 || src->ne[0] != W ||
        dst->ne[0] != n) {
        return -1;
    }
    const int64_t blocks_per_row = W / n;
    const int64_t R              = dst->ne[1] * dst->ne[2] * dst->ne[3];
    if (R * n != src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3]) {
        return -1;
    }

    const char *  x_base = (const char *) src->data;
    const char *  s_base = (const char *) signs->data;  // may be NULL
    char *        d_base = (char *) dst->data;
    const float   scale  = params->scale;

    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");
    __asm__ volatile("fbc.ps f30, %[s]\n" : : [s] "m"(scale) : "f30");

    for (int64_t j = thread_id; j < R; j += num_threads) {
        // destination row
        const int64_t d1  = j % dst->ne[1];
        const int64_t d23 = j / dst->ne[1];
        const int64_t d2  = d23 % dst->ne[2];
        const int64_t d3  = d23 / dst->ne[2];
        float * row = (float *) (d_base + d1 * dst->nb[1] + d2 * dst->nb[2] + d3 * dst->nb[3]);

        // source block
        const int64_t r   = j / blocks_per_row;
        const int64_t b   = j % blocks_per_row;
        const int64_t i1  = r % src->ne[1];
        const int64_t r23 = r / src->ne[1];
        const int64_t i2  = r23 % src->ne[2];
        const int64_t i3  = r23 / src->ne[2];
        const float * xp  = (const float *) (x_base + i1 * src->nb[1] + i2 * src->nb[2] + i3 * src->nb[3]) + b * n;

        // pass 0: row = scale * signs * x
        if (s_base) {
            const float * sp = (const float *) s_base + b * n;
            for (int64_t c = 0; c < n; c += 8) {
                __asm__ volatile(
                    "flw.ps  f10, %[x]\n\t"
                    "flw.ps  f11, %[sg]\n\t"
                    "fmul.ps f10, f10, f11, rne\n\t"
                    "fmul.ps f10, f10, f30, rne\n\t"
                    "fsw.ps  f10, %[d]\n\t"
                    : [d] "=m"(*(float (*)[8]) (row + c))
                    : [x] "m"(*(const float (*)[8]) (xp + c)), [sg] "m"(*(const float (*)[8]) (sp + c))
                    : "f10", "f11");
            }
        } else {
            for (int64_t c = 0; c < n; c += 8) {
                __asm__ volatile(
                    "flw.ps  f10, %[x]\n\t"
                    "fmul.ps f10, f10, f30, rne\n\t"
                    "fsw.ps  f10, %[d]\n\t"
                    : [d] "=m"(*(float (*)[8]) (row + c))
                    : [x] "m"(*(const float (*)[8]) (xp + c))
                    : "f10");
            }
        }

        // butterflies, distance 1, 2, 4 inside the vector, then 8 .. n/2 across vectors
        stage_gather(row, n, pat_u1, pat_w1);
        stage_gather(row, n, pat_u2, pat_w2);
        stage_gather(row, n, pat_u4, pat_w4);
        for (int64_t len = 8; len < n; len <<= 1) {
            stage_vector(row, n, len);
        }
    }

    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
    return 0;
}
