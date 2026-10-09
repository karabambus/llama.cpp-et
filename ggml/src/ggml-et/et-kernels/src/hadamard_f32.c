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
// Each hart transforms whole rows, in place on dst, in four memory passes instead of one per stage (the row is as
// large as the hart's L1D, so every pass streams through L2): pass A loads each 32-float group into four vector
// registers, applies scale and signs, runs the distance-1 and distance-2 butterflies with lane swizzles (0xB1,
// 0x4E: the board-proven pair/adjacent swaps, replicated per 4-lane half) and the distance-8 and -16 ones across
// the four registers, and stores; the distance-4 stage pairs lanes across the two halves of a vector, which no
// register swizzle reaches, so it stays a word gather/scatter pass; the stages of distance 512 and above are plain
// vector passes; the stages 32..256 run on 16-row chunks (rows of 32 floats, one column vector at a time) held in
// registers. Stages commute, so the order is free. n must be a power of two with 16 <= n <= 4096 (the host checks).
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
static const int32_t __attribute__((aligned(64))) pat_u4[8] = { 0, 4, 8, 12, 32, 36, 40, 44 };
static const int32_t __attribute__((aligned(64))) pat_w4[8] = { 16, 20, 24, 28, 48, 52, 56, 60 };

// One in-vector stage over the whole row: patterns in f18 (u) and f19 (w) — callee-saved registers, so the compiler
// preserves them around the stage and never allocates them between the asm blocks (KB 04 G2).
static inline void __attribute__((always_inline)) stage_gather(float * row, int64_t n, const int32_t * pu,
                                                                 const int32_t * pw) {
    __asm__ volatile(
        "flw.ps f18, %[u]\n\t"
        "flw.ps f19, %[w]\n\t"
        :
        : [u] "m"(*(const int32_t (*)[8]) pu), [w] "m"(*(const int32_t (*)[8]) pw)
        : "f18", "f19");
    for (int64_t g = 0; g < n; g += 16) {
        float * base = row + g;
        __asm__ volatile(
            "fgw.ps  f10, f18(%[b])\n\t"
            "fgw.ps  f11, f19(%[b])\n\t"
            "fadd.ps f12, f10, f11, rne\n\t"
            "fsub.ps f13, f10, f11, rne\n\t"
            "fscw.ps f12, f18(%[b])\n\t"
            "fscw.ps f13, f19(%[b])\n\t"
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

// Lane sign vectors of the two swizzle stages: new = a * sign + swizzle(a) gives a_u + a_w on the "u" lane and
// a_u - a_w on the "w" lane of each pair.
static const float __attribute__((aligned(64))) sgn1[8] = { 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f };
static const float __attribute__((aligned(64))) sgn2[8] = { 1.f, 1.f, -1.f, -1.f, 1.f, 1.f, -1.f, -1.f };

#define HAD_SWZ_STAGE(v, imm, sg)                      \
    "fswizz.ps f11, " v ", " imm "\n\t"                 \
    "fmadd.ps  " v ", " v ", " sg ", f11, rne\n\t"
// (a, b) <- (a + b, a - b); the copy back into b is an identity swizzle (0xE4), exact.
#define HAD_BF(a, b, t)                                \
    "fsub.ps   " t ", " a ", " b ", rne\n\t"            \
    "fadd.ps   " a ", " a ", " b ", rne\n\t"            \
    "fswizz.ps " b ", " t ", 0xE4\n\t"

// Pass A on one 32-float group: dst[0..31] = butterflies of distance 1, 2, 8, 16 over scale * signs * x[0..31].
static inline void __attribute__((always_inline)) pass_a32(float * d, const float * x, const float * sg,
                                                             const float * scale) {
    if (sg) {
        __asm__ volatile(
            "flw.ps  f0, 0(%[x])\n\t"  "flw.ps  f1, 32(%[x])\n\t"  "flw.ps  f2, 64(%[x])\n\t"  "flw.ps  f3, 96(%[x])\n\t"
            "flw.ps  f4, 0(%[s])\n\t"  "flw.ps  f5, 32(%[s])\n\t"  "flw.ps  f6, 64(%[s])\n\t"  "flw.ps  f7, 96(%[s])\n\t"
            "fmul.ps f0, f0, f4, rne\n\t" "fmul.ps f1, f1, f5, rne\n\t" "fmul.ps f2, f2, f6, rne\n\t" "fmul.ps f3, f3, f7, rne\n\t"
            "fbc.ps  f8, %[sc]\n\t"
            "fmul.ps f0, f0, f8, rne\n\t" "fmul.ps f1, f1, f8, rne\n\t" "fmul.ps f2, f2, f8, rne\n\t" "fmul.ps f3, f3, f8, rne\n\t"
            "flw.ps  f9, %[g1]\n\t"  "flw.ps  f10, %[g2]\n\t"
            HAD_SWZ_STAGE("f0", "0xB1", "f9") HAD_SWZ_STAGE("f1", "0xB1", "f9") HAD_SWZ_STAGE("f2", "0xB1", "f9") HAD_SWZ_STAGE("f3", "0xB1", "f9")
            HAD_SWZ_STAGE("f0", "0x4E", "f10") HAD_SWZ_STAGE("f1", "0x4E", "f10") HAD_SWZ_STAGE("f2", "0x4E", "f10") HAD_SWZ_STAGE("f3", "0x4E", "f10")
            HAD_BF("f0", "f1", "f11") HAD_BF("f2", "f3", "f11")
            HAD_BF("f0", "f2", "f11") HAD_BF("f1", "f3", "f11")
            "fsw.ps  f0, 0(%[d])\n\t"  "fsw.ps  f1, 32(%[d])\n\t"  "fsw.ps  f2, 64(%[d])\n\t"  "fsw.ps  f3, 96(%[d])\n\t"
            :
            : [x] "r"(x), [s] "r"(sg), [d] "r"(d), [sc] "m"(*scale), [g1] "m"(*(const float (*)[8]) sgn1),
              [g2] "m"(*(const float (*)[8]) sgn2)
            : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "memory");
    } else {
        __asm__ volatile(
            "flw.ps  f0, 0(%[x])\n\t"  "flw.ps  f1, 32(%[x])\n\t"  "flw.ps  f2, 64(%[x])\n\t"  "flw.ps  f3, 96(%[x])\n\t"
            "fbc.ps  f8, %[sc]\n\t"
            "fmul.ps f0, f0, f8, rne\n\t" "fmul.ps f1, f1, f8, rne\n\t" "fmul.ps f2, f2, f8, rne\n\t" "fmul.ps f3, f3, f8, rne\n\t"
            "flw.ps  f9, %[g1]\n\t"  "flw.ps  f10, %[g2]\n\t"
            HAD_SWZ_STAGE("f0", "0xB1", "f9") HAD_SWZ_STAGE("f1", "0xB1", "f9") HAD_SWZ_STAGE("f2", "0xB1", "f9") HAD_SWZ_STAGE("f3", "0xB1", "f9")
            HAD_SWZ_STAGE("f0", "0x4E", "f10") HAD_SWZ_STAGE("f1", "0x4E", "f10") HAD_SWZ_STAGE("f2", "0x4E", "f10") HAD_SWZ_STAGE("f3", "0x4E", "f10")
            HAD_BF("f0", "f1", "f11") HAD_BF("f2", "f3", "f11")
            HAD_BF("f0", "f2", "f11") HAD_BF("f1", "f3", "f11")
            "fsw.ps  f0, 0(%[d])\n\t"  "fsw.ps  f1, 32(%[d])\n\t"  "fsw.ps  f2, 64(%[d])\n\t"  "fsw.ps  f3, 96(%[d])\n\t"
            :
            : [x] "r"(x), [d] "r"(d), [sc] "m"(*scale), [g1] "m"(*(const float (*)[8]) sgn1),
              [g2] "m"(*(const float (*)[8]) sgn2)
            : "f0", "f1", "f2", "f3", "f8", "f9", "f10", "f11", "memory");
    }
}

// Pass A on one 16-float group (n == 16 only): distances 1, 2 by swizzle, 8 across the two registers.
static inline void __attribute__((always_inline)) pass_a16(float * d, const float * x, const float * sg,
                                                             const float * scale) {
    __asm__ volatile(
        "flw.ps  f0, 0(%[x])\n\t"  "flw.ps  f1, 32(%[x])\n\t"
        "fbc.ps  f8, %[sc]\n\t"
        "fmul.ps f0, f0, f8, rne\n\t" "fmul.ps f1, f1, f8, rne\n\t"
        "flw.ps  f9, %[g1]\n\t"  "flw.ps  f10, %[g2]\n\t"
        HAD_SWZ_STAGE("f0", "0xB1", "f9") HAD_SWZ_STAGE("f1", "0xB1", "f9")
        HAD_SWZ_STAGE("f0", "0x4E", "f10") HAD_SWZ_STAGE("f1", "0x4E", "f10")
        HAD_BF("f0", "f1", "f11")
        "fsw.ps  f0, 0(%[d])\n\t"  "fsw.ps  f1, 32(%[d])\n\t"
        :
        : [x] "r"(x), [d] "r"(d), [sc] "m"(*scale), [g1] "m"(*(const float (*)[8]) sgn1),
          [g2] "m"(*(const float (*)[8]) sgn2)
        : "f0", "f1", "f8", "f9", "f10", "f11", "memory");
    if (sg) {
        // signs applied before the transform: redo with the sign multiply (n == 16 is a test shape only)
        __asm__ volatile(
            "flw.ps  f0, 0(%[x])\n\t"  "flw.ps  f1, 32(%[x])\n\t"
            "flw.ps  f4, 0(%[s])\n\t"  "flw.ps  f5, 32(%[s])\n\t"
            "fmul.ps f0, f0, f4, rne\n\t" "fmul.ps f1, f1, f5, rne\n\t"
            "fbc.ps  f8, %[sc]\n\t"
            "fmul.ps f0, f0, f8, rne\n\t" "fmul.ps f1, f1, f8, rne\n\t"
            "flw.ps  f9, %[g1]\n\t"  "flw.ps  f10, %[g2]\n\t"
            HAD_SWZ_STAGE("f0", "0xB1", "f9") HAD_SWZ_STAGE("f1", "0xB1", "f9")
            HAD_SWZ_STAGE("f0", "0x4E", "f10") HAD_SWZ_STAGE("f1", "0x4E", "f10")
            HAD_BF("f0", "f1", "f11")
            "fsw.ps  f0, 0(%[d])\n\t"  "fsw.ps  f1, 32(%[d])\n\t"
            :
            : [x] "r"(x), [s] "r"(sg), [d] "r"(d), [sc] "m"(*scale), [g1] "m"(*(const float (*)[8]) sgn1),
              [g2] "m"(*(const float (*)[8]) sgn2)
            : "f0", "f1", "f4", "f5", "f8", "f9", "f10", "f11", "memory");
    }
}

// Pass B on 16 rows of 32 floats, one column vector (8 floats) each: butterflies of row distance 1, 2, 4, 8
// (= element distance 32, 64, 128, 256) in registers. p points at row 0's column vector; rows are 128 bytes apart.
static inline void __attribute__((always_inline)) pass_b16(float * p) {
    __asm__ volatile(
        "flw.ps f0, 0(%[p])\n\t"     "flw.ps f1, 128(%[p])\n\t"   "flw.ps f2, 256(%[p])\n\t"   "flw.ps f3, 384(%[p])\n\t"
        "flw.ps f4, 512(%[p])\n\t"   "flw.ps f5, 640(%[p])\n\t"   "flw.ps f6, 768(%[p])\n\t"   "flw.ps f7, 896(%[p])\n\t"
        "flw.ps f8, 1024(%[p])\n\t"  "flw.ps f9, 1152(%[p])\n\t"  "flw.ps f10, 1280(%[p])\n\t" "flw.ps f11, 1408(%[p])\n\t"
        "flw.ps f12, 1536(%[p])\n\t" "flw.ps f13, 1664(%[p])\n\t" "flw.ps f14, 1792(%[p])\n\t" "flw.ps f15, 1920(%[p])\n\t"
        HAD_BF("f0", "f1", "f16")  HAD_BF("f2", "f3", "f16")  HAD_BF("f4", "f5", "f16")  HAD_BF("f6", "f7", "f16")
        HAD_BF("f8", "f9", "f16")  HAD_BF("f10", "f11", "f16") HAD_BF("f12", "f13", "f16") HAD_BF("f14", "f15", "f16")
        HAD_BF("f0", "f2", "f16")  HAD_BF("f1", "f3", "f16")  HAD_BF("f4", "f6", "f16")  HAD_BF("f5", "f7", "f16")
        HAD_BF("f8", "f10", "f16") HAD_BF("f9", "f11", "f16") HAD_BF("f12", "f14", "f16") HAD_BF("f13", "f15", "f16")
        HAD_BF("f0", "f4", "f16")  HAD_BF("f1", "f5", "f16")  HAD_BF("f2", "f6", "f16")  HAD_BF("f3", "f7", "f16")
        HAD_BF("f8", "f12", "f16") HAD_BF("f9", "f13", "f16") HAD_BF("f10", "f14", "f16") HAD_BF("f11", "f15", "f16")
        HAD_BF("f0", "f8", "f16")  HAD_BF("f1", "f9", "f16")  HAD_BF("f2", "f10", "f16") HAD_BF("f3", "f11", "f16")
        HAD_BF("f4", "f12", "f16") HAD_BF("f5", "f13", "f16") HAD_BF("f6", "f14", "f16") HAD_BF("f7", "f15", "f16")
        "fsw.ps f0, 0(%[p])\n\t"     "fsw.ps f1, 128(%[p])\n\t"   "fsw.ps f2, 256(%[p])\n\t"   "fsw.ps f3, 384(%[p])\n\t"
        "fsw.ps f4, 512(%[p])\n\t"   "fsw.ps f5, 640(%[p])\n\t"   "fsw.ps f6, 768(%[p])\n\t"   "fsw.ps f7, 896(%[p])\n\t"
        "fsw.ps f8, 1024(%[p])\n\t"  "fsw.ps f9, 1152(%[p])\n\t"  "fsw.ps f10, 1280(%[p])\n\t" "fsw.ps f11, 1408(%[p])\n\t"
        "fsw.ps f12, 1536(%[p])\n\t" "fsw.ps f13, 1664(%[p])\n\t" "fsw.ps f14, 1792(%[p])\n\t" "fsw.ps f15, 1920(%[p])\n\t"
        :
        : [p] "r"(p)
        : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16",
          "memory");
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

        const float * sp = s_base ? (const float *) s_base + b * n : (const float *) 0;

        // pass A: scale * signs * x with the distance 1, 2, 8, 16 butterflies, per 32-float group
        if (n >= 32) {
            for (int64_t c = 0; c < n; c += 32) {
                pass_a32(row + c, xp + c, sp ? sp + c : (const float *) 0, &scale);
            }
        } else {
            pass_a16(row, xp, sp, &scale);
        }
        // distance 4: lanes of the two halves of a vector - a gather/scatter pass
        stage_gather(row, n, pat_u4, pat_w4);
        // distances 32 .. n/2 over rows of 32 floats
        const int64_t rows32 = n / 32;
        if (rows32 >= 16) {
            for (int64_t len = 512; len < n; len <<= 1) {
                stage_vector(row, n, len);
            }
            for (int64_t r0 = 0; r0 < rows32; r0 += 16) {
                for (int64_t c = 0; c < 4; c++) {
                    pass_b16(row + r0 * 32 + c * 8);
                }
            }
        } else {
            for (int64_t len = 32; len < n; len <<= 1) {
                stage_vector(row, n, len);
            }
        }
    }

    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));

#ifdef ET_UBERKERNEL
    // inside a segment the next instruction may run on other shires: publish dst past L2 (the standalone launch
    // relies on the firmware's post-kernel cleanup instead)
    FENCE;
    evict_region_past_l2(dst->data, (size_t) dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] * dst->nb[0]);
    WAIT_CACHEOPS;
    FENCE;
#endif
    return 0;
}
