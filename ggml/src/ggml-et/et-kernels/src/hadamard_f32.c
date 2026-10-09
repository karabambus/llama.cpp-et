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
// The 64 harts of one shire share a row (row j belongs to shire j % 32), in place on dst, in four memory passes
// with one shire barrier between the group-local and the column phases (a single hart streaming its row from DRAM
// one line at a time was the floor): pass A loads each 32-float group into four vector
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

// Butterfly of two 16-float column lines (rows ra, rb of the 32-float-row view, column line cl): in place.
static inline void __attribute__((always_inline)) bf_line(float * pa, float * pb) {
    __asm__ volatile(
        "flw.ps  f0, 0(%[a])\n\t"  "flw.ps  f1, 32(%[a])\n\t"
        "flw.ps  f2, 0(%[b])\n\t"  "flw.ps  f3, 32(%[b])\n\t"
        "fadd.ps f4, f0, f2, rne\n\t" "fsub.ps f5, f0, f2, rne\n\t"
        "fadd.ps f6, f1, f3, rne\n\t" "fsub.ps f7, f1, f3, rne\n\t"
        "fsw.ps  f4, 0(%[a])\n\t"  "fsw.ps  f6, 32(%[a])\n\t"
        "fsw.ps  f5, 0(%[b])\n\t"  "fsw.ps  f7, 32(%[b])\n\t"
        :
        : [a] "r"(pa), [b] "r"(pb)
        : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "memory");
}

// Stages of element distance len >= 32 on one column line (16 floats of every 32-float row) of a row of n floats.
static inline void __attribute__((always_inline)) col_stage(float * row, int64_t n, int64_t len, int64_t cl) {
    const int64_t rows32 = n / 32;
    const int64_t d      = len / 32;  // row distance
    for (int64_t i = 0; i < rows32; i += 2 * d) {
        for (int64_t k = 0; k < d; k++) {
            bf_line(row + (i + k) * 32 + cl * 16, row + (i + k + d) * 32 + cl * 16);
        }
    }
}

// Publish lines this hart wrote to the shire's L2 (a FENCE alone leaves them in L1): nlines at stride bytes apart,
// 16 per CSR write. Caller issues WAIT_CACHEOPS before the barrier.
static inline void __attribute__((always_inline)) publish_l2(const void * addr, int64_t nlines, int64_t stride) {
    FENCE;
    for (int64_t i = 0; i < nlines; i += 16) {
        const int64_t cnt = nlines - i < 16 ? nlines - i : 16;
        evict_to_l2((const char *) addr + i * stride, (uint64_t) cnt, (uint64_t) stride);
    }
}

int entry_point(struct ggml_et_hadamard_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    if (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }
    (void) kernel_env;
    const int     h     = (int) (get_hart_id() & 63);  // hart within its shire
    const int64_t shire = (int64_t) get_shire_id();     // 0..31 (the host launches every shire)

    const struct ggml_tensor * src   = &params->src;
    const struct ggml_tensor * signs = &params->signs;
    const struct ggml_tensor * dst   = &params->dst;

    const int64_t n = params->blk;
    const int64_t W = params->width;
    // Every check precedes the first barrier, so all harts of a shire return together or not at all.
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

    // Row j belongs to shire j % 32; the 64 harts of the shire share it. Every shire runs the same number of row
    // slots and every hart takes the two shire barriers of each slot, work or no work (a slot past R is empty).
    const int64_t slots  = (R + 31) / 32;
    const int64_t groups = n >= 32 ? n / 32 : 1;  // 32-float groups per row (one 16-float group when n == 16)
    const int64_t rows32 = n / 32;

    for (int64_t t = 0; t < slots; t++) {
        const int64_t j    = shire + 32 * t;
        const int     have = j < R;
        float *       row  = (float *) 0;
        const float * xp   = (const float *) 0;
        const float * sp   = (const float *) 0;
        if (have) {
            const int64_t d1  = j % dst->ne[1];
            const int64_t d23 = j / dst->ne[1];
            const int64_t d2  = d23 % dst->ne[2];
            const int64_t d3  = d23 / dst->ne[2];
            row               = (float *) (d_base + d1 * dst->nb[1] + d2 * dst->nb[2] + d3 * dst->nb[3]);
            const int64_t r   = j / blocks_per_row;
            const int64_t b   = j % blocks_per_row;
            const int64_t i1  = r % src->ne[1];
            const int64_t r23 = r / src->ne[1];
            const int64_t i2  = r23 % src->ne[2];
            const int64_t i3  = r23 / src->ne[2];
            xp = (const float *) (x_base + i1 * src->nb[1] + i2 * src->nb[2] + i3 * src->nb[3]) + b * n;
            sp = s_base ? (const float *) s_base + b * n : (const float *) 0;

            // phase 1: per 32-float group (two whole cache lines per group, one hart per group): scale, signs,
            // distances 1, 2, 8, 16 in registers, then the distance-4 gather/scatter on the group
            for (int64_t g = h; g < groups; g += 64) {
                if (n >= 32) {
                    pass_a32(row + g * 32, xp + g * 32, sp ? sp + g * 32 : (const float *) 0, &scale);
                    stage_gather(row + g * 32, 32, pat_u4, pat_w4);
                    publish_l2(row + g * 32, 2, 64);
                } else {
                    pass_a16(row, xp, sp, &scale);
                    stage_gather(row, 16, pat_u4, pat_w4);
                    publish_l2(row, 1, 64);
                }
            }
        }
        WAIT_CACHEOPS;
        FENCE;
        et_barrier(ET_BARRIER_SHIRE);

        // phase 2: distances 32 .. n/2, one hart per 64-byte column line (16 floats of every 32-float row), so no
        // two harts write the same line; the lines it reads were written by other harts in phase 1 and are not in
        // this hart's L1 (clean at entry, untouched since).
        if (have && rows32 >= 2 && h < 2) {
            const int64_t cl = h;
            if (rows32 >= 16) {
                for (int64_t len = 512; len < n; len <<= 1) {
                    col_stage(row, n, len, cl);
                }
                for (int64_t r0 = 0; r0 < rows32; r0 += 16) {
                    pass_b16(row + r0 * 32 + cl * 16);
                    pass_b16(row + r0 * 32 + cl * 16 + 8);
                }
            } else {
                for (int64_t len = 32; len < n; len <<= 1) {
                    col_stage(row, n, len, cl);
                }
            }
            publish_l2(row + cl * 16, rows32, 128);
        }
        WAIT_CACHEOPS;
        FENCE;
        et_barrier(ET_BARRIER_SHIRE);
    }

    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));

    // Each hart published its lines to L2 before the closing barrier; the next instruction's consumer-side evict
    // moves them past L2 (and the firmware's post-kernel cleanup does so for a standalone launch).
    return 0;
}
