// PQ2_0 (Prism ternary, group 128) row-dot helpers for the vector lanes.
//
// Block layout (ggml-common.h): { ggml_half d; uint8_t qs[32]; } = 34 bytes. Element j of the block sits in
// byte j/4 at bit offset 2*(j%4), LSB first; the 2-bit code q maps to the weight (q - 1), i.e. 00 = -1, 01 = 0,
// 10 = +1 (11 = +2 is never produced by the ternary packer but decodes the same way as on the CPU).
//
// Unpacking 8 consecutive elements into 8 lanes: a byte gather with the pattern {0,0,0,0,1,1,1,1} places byte 2c
// in lanes 0..3 and byte 2c+1 in lanes 4..7; a per-lane logical shift by {0,2,4,6,0,2,4,6} and a mask of 3 then
// leave the lane's own 2-bit code. The gather sign-extends bytes >= 0x80, which the logical shift and mask make
// irrelevant. Two gather forms: fgb.ps (index vector in f31, any address) and the windowed fg32b.ps (index pattern
// in an integer register, lane e reads (a & ~31) + ((a + idx_e) & 31)), which costs a fraction of fgb.ps's issue
// time. With idx <= 1 the window wraps only when (a & 31) == 31, so the two are bit-identical at every even chunk
// address; the kernel checks once per launch that the weight base and strides are even (blocks are 34 bytes, so
// every chunk address is then even) and otherwise keeps the fgb.ps tiles.
//
// The fp16 block scale is converted in integer code (fp16_to_fp32_bits) and broadcast with fbcx.ps; fcvt.ps.f16
// flushes subnormal halves and must not touch block scales (KB 02 section 5.1).
//
// Register contract:
//   f20       - row accumulator (reset per row by pq2_dot_reset, read by pq2_dot_reduce)
//   f31       - byte gather pattern, f30 - per-lane shift amounts (both loaded by pq2_dot_tile)
//   f10-f15   - scratch inside the tile
//   f1-f5, t0 - scratch inside the reduce
//   two-row tile: row 1 uses f16 (block acc), f17/f18 (scratch), f21 (row acc); f13 and f15 are shared

#ifndef PQ2_OPS_H
#define PQ2_OPS_H

#include "math_fp.h"
#include "quants.h"

#include <stdint.h>

static inline void __attribute__((always_inline)) pq2_dot_reset(void) {
    __asm__ volatile("fbci.pi f20, 0" ::: "f20");
}

// One 8-element chunk: a = &qs[2c], b = &b_col[8c]. Accumulates into f10.
#define PQ2V_CHUNK(a, b)                                                                    \
    __asm__ volatile(                                                                       \
        "fgb.ps     f11, f31(%[q])\n\t"                                                     \
        "fsrl.pi    f12, f11, f30\n\t"                                                      \
        "fandi.pi   f12, f12, 3\n\t"                                                        \
        "faddi.pi   f12, f12, -1\n\t"                                                       \
        "fcvt.ps.pw f12, f12, rne\n\t"                                                      \
        "flw.ps     f13, %[bv]\n\t"                                                         \
        "fmadd.ps   f10, f12, f13, f10, rne\n\t"                                            \
        :                                                                                   \
        : [q] "r"(a), [bv] "m"(*(const float (*)[8]) (b))                                   \
        : "f10", "f11", "f12", "f13")

// fg32b.ps index pattern {0,0,0,0,1,1,1,1}: 5 bits per lane, lane e at bits 5e..5e+4.
#define PQ2_GATHER_WINDOW_IDX 0x842100000ULL

// PQ2V_CHUNK with the windowed gather; gi holds PQ2_GATHER_WINDOW_IDX, a must be even.
#define PQ2V_CHUNK_W(a, b, gi)                                                              \
    __asm__ volatile(                                                                       \
        "fg32b.ps   f11, %[gi](%[q])\n\t"                                                   \
        "fsrl.pi    f12, f11, f30\n\t"                                                      \
        "fandi.pi   f12, f12, 3\n\t"                                                        \
        "faddi.pi   f12, f12, -1\n\t"                                                       \
        "fcvt.ps.pw f12, f12, rne\n\t"                                                      \
        "flw.ps     f13, %[bv]\n\t"                                                         \
        "fmadd.ps   f10, f12, f13, f10, rne\n\t"                                            \
        :                                                                                   \
        : [q] "r"(a), [gi] "r"(gi), [bv] "m"(*(const float (*)[8]) (b))                     \
        : "f10", "f11", "f12", "f13")

// Accumulate n_blocks PQ2_0 blocks of q_row against b_col into f20.
static inline void __attribute__((always_inline)) pq2_dot_tile(const block_pq2_0 * q_row,
                                                               const float *       b_col,
                                                               int64_t             n_blocks) {
    static const int32_t __attribute__((aligned(64))) gather_pattern[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };
    static const int32_t __attribute__((aligned(64))) shift_pattern[8]  = { 0, 2, 4, 6, 0, 2, 4, 6 };

    __asm__ volatile(
        "flw.ps f31, %[g]\n\t"
        "flw.ps f30, %[s]\n\t"
        :
        : [g] "m"(*(const int32_t (*)[8]) gather_pattern), [s] "m"(*(const int32_t (*)[8]) shift_pattern)
        : "f30", "f31");

    for (int64_t kb = 0; kb < n_blocks; kb++) {
        const block_pq2_0 * blk = q_row + kb;
        const uint8_t *     qs  = blk->qs;
        const float *       b   = b_col + kb * QK_PQ2_0;

        __asm__ volatile("fbci.pi f10, 0" ::: "f10");

        // 128 elements = 16 chunks of 8; four chunks per unrolled step (one 32-element group, 8 bytes).
        for (int c = 0; c < 16; c += 4) {
            PQ2V_CHUNK(qs + 2 * c + 0, b + 8 * c + 0);
            PQ2V_CHUNK(qs + 2 * c + 2, b + 8 * c + 8);
            PQ2V_CHUNK(qs + 2 * c + 4, b + 8 * c + 16);
            PQ2V_CHUNK(qs + 2 * c + 6, b + 8 * c + 24);
        }

        const uint32_t scale_bits = fp16_to_fp32_bits(blk->d);
        __asm__ volatile(
            "fbcx.ps  f15, %[sb]\n\t"
            "fmadd.ps f20, f10, f15, f20, rne\n\t"
            :
            : [sb] "r"(scale_bits)
            : "f15", "f20");
    }
}

// pq2_dot_tile with the windowed gather (even chunk addresses only).
static inline void __attribute__((always_inline)) pq2_dot_tile_w(const block_pq2_0 * q_row,
                                                                 const float *       b_col,
                                                                 int64_t             n_blocks) {
    static const int32_t __attribute__((aligned(64))) shift_pattern[8] = { 0, 2, 4, 6, 0, 2, 4, 6 };
    const uint64_t                                    gi               = PQ2_GATHER_WINDOW_IDX;

    __asm__ volatile("flw.ps f30, %[s]\n\t" : : [s] "m"(*(const int32_t (*)[8]) shift_pattern) : "f30");

    for (int64_t kb = 0; kb < n_blocks; kb++) {
        const block_pq2_0 * blk = q_row + kb;
        const uint8_t *     qs  = blk->qs;
        const float *       b   = b_col + kb * QK_PQ2_0;

        __asm__ volatile("fbci.pi f10, 0" ::: "f10");

        for (int c = 0; c < 16; c += 4) {
            PQ2V_CHUNK_W(qs + 2 * c + 0, b + 8 * c + 0, gi);
            PQ2V_CHUNK_W(qs + 2 * c + 2, b + 8 * c + 8, gi);
            PQ2V_CHUNK_W(qs + 2 * c + 4, b + 8 * c + 16, gi);
            PQ2V_CHUNK_W(qs + 2 * c + 6, b + 8 * c + 24, gi);
        }

        const uint32_t scale_bits = fp16_to_fp32_bits(blk->d);
        __asm__ volatile(
            "fbcx.ps  f15, %[sb]\n\t"
            "fmadd.ps f20, f10, f15, f20, rne\n\t"
            :
            : [sb] "r"(scale_bits)
            : "f15", "f20");
    }
}

// Horizontal sum of the 8-lane accumulator f20.
static inline float __attribute__((always_inline)) pq2_dot_reduce(void) {
    float result;
    __asm__ __volatile__(
        "fswizz.ps f1, f20, 0xB1 \n\t"
        "fadd.ps   f2, f20, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(result)::"t0", "f1", "f2", "f3", "f4", "f5", "f20");
    return result;
}

// Full-row dot product; the caller owns the lane mask (set to 0xFF around the row loop).
static inline float pq2_dot_compute(const block_pq2_0 * q_row, const float * b_col, int64_t K_blocks) {
    pq2_dot_reset();
    pq2_dot_tile(q_row, b_col, K_blocks);
    return pq2_dot_reduce();
}

static inline float pq2_dot_compute_w(const block_pq2_0 * q_row, const float * b_col, int64_t K_blocks) {
    pq2_dot_reset();
    pq2_dot_tile_w(q_row, b_col, K_blocks);
    return pq2_dot_reduce();
}

// Two rows interleaved: the chip-wide stream of this kernel is latency-bound at one 64-byte line in flight per hart
// (~25 GB/s against a 62-70 GB/s band), so a hart works two rows at once and keeps two lines in flight. Register contract: row 0 as above (f10 block acc, f11/f12 scratch, f20 row
// acc), row 1 in f16 (block acc), f17/f18 (scratch), f21 (row acc); the activation vector f13 and the scale f15 are
// shared per chunk / per block; f30/f31 are the patterns.
#define PQ2V_CHUNK2(a0, a1, b)                                                              \
    __asm__ volatile(                                                                       \
        "fgb.ps     f11, f31(%[q0])\n\t"                                                    \
        "fgb.ps     f17, f31(%[q1])\n\t"                                                    \
        "flw.ps     f13, %[bv]\n\t"                                                         \
        "fsrl.pi    f12, f11, f30\n\t"                                                      \
        "fsrl.pi    f18, f17, f30\n\t"                                                      \
        "fandi.pi   f12, f12, 3\n\t"                                                        \
        "fandi.pi   f18, f18, 3\n\t"                                                        \
        "faddi.pi   f12, f12, -1\n\t"                                                       \
        "faddi.pi   f18, f18, -1\n\t"                                                       \
        "fcvt.ps.pw f12, f12, rne\n\t"                                                      \
        "fcvt.ps.pw f18, f18, rne\n\t"                                                      \
        "fmadd.ps   f10, f12, f13, f10, rne\n\t"                                            \
        "fmadd.ps   f16, f18, f13, f16, rne\n\t"                                            \
        :                                                                                   \
        : [q0] "r"(a0), [q1] "r"(a1), [bv] "m"(*(const float (*)[8]) (b))                  \
        : "f10", "f11", "f12", "f13", "f16", "f17", "f18")

// PQ2V_CHUNK2 with the windowed gather; gi holds PQ2_GATHER_WINDOW_IDX, a0 and a1 must be even.
#define PQ2V_CHUNK2_W(a0, a1, b, gi)                                                        \
    __asm__ volatile(                                                                       \
        "fg32b.ps   f11, %[gi](%[q0])\n\t"                                                  \
        "fg32b.ps   f17, %[gi](%[q1])\n\t"                                                  \
        "flw.ps     f13, %[bv]\n\t"                                                         \
        "fsrl.pi    f12, f11, f30\n\t"                                                      \
        "fsrl.pi    f18, f17, f30\n\t"                                                      \
        "fandi.pi   f12, f12, 3\n\t"                                                        \
        "fandi.pi   f18, f18, 3\n\t"                                                        \
        "faddi.pi   f12, f12, -1\n\t"                                                       \
        "faddi.pi   f18, f18, -1\n\t"                                                       \
        "fcvt.ps.pw f12, f12, rne\n\t"                                                      \
        "fcvt.ps.pw f18, f18, rne\n\t"                                                      \
        "fmadd.ps   f10, f12, f13, f10, rne\n\t"                                            \
        "fmadd.ps   f16, f18, f13, f16, rne\n\t"                                            \
        :                                                                                   \
        : [q0] "r"(a0), [q1] "r"(a1), [gi] "r"(gi), [bv] "m"(*(const float (*)[8]) (b))     \
        : "f10", "f11", "f12", "f13", "f16", "f17", "f18")

static inline void __attribute__((always_inline)) pq2_dot_reset2(void) {
    __asm__ volatile("fbci.pi f20, 0\n\tfbci.pi f21, 0" ::: "f20", "f21");
}

static inline void __attribute__((always_inline)) pq2_dot_tile2(const block_pq2_0 * q_row0,
                                                                const block_pq2_0 * q_row1,
                                                                const float *       b_col,
                                                                int64_t             n_blocks) {
    static const int32_t __attribute__((aligned(64))) gather_pattern[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };
    static const int32_t __attribute__((aligned(64))) shift_pattern[8]  = { 0, 2, 4, 6, 0, 2, 4, 6 };
    __asm__ volatile(
        "flw.ps f31, %[g]\n\t"
        "flw.ps f30, %[s]\n\t"
        :
        : [g] "m"(*(const int32_t (*)[8]) gather_pattern), [s] "m"(*(const int32_t (*)[8]) shift_pattern)
        : "f30", "f31");
    for (int64_t kb = 0; kb < n_blocks; kb++) {
        const uint8_t * qs0 = q_row0[kb].qs;
        const uint8_t * qs1 = q_row1[kb].qs;
        const float *   b   = b_col + kb * QK_PQ2_0;
        __asm__ volatile("fbci.pi f10, 0\n\tfbci.pi f16, 0" ::: "f10", "f16");
        for (int c = 0; c < 16; c += 4) {
            PQ2V_CHUNK2(qs0 + 2 * c + 0, qs1 + 2 * c + 0, b + 8 * c + 0);
            PQ2V_CHUNK2(qs0 + 2 * c + 2, qs1 + 2 * c + 2, b + 8 * c + 8);
            PQ2V_CHUNK2(qs0 + 2 * c + 4, qs1 + 2 * c + 4, b + 8 * c + 16);
            PQ2V_CHUNK2(qs0 + 2 * c + 6, qs1 + 2 * c + 6, b + 8 * c + 24);
        }
        const uint32_t sb0 = fp16_to_fp32_bits(q_row0[kb].d);
        const uint32_t sb1 = fp16_to_fp32_bits(q_row1[kb].d);
        __asm__ volatile(
            "fbcx.ps  f15, %[s0]\n\t"
            "fmadd.ps f20, f10, f15, f20, rne\n\t"
            "fbcx.ps  f15, %[s1]\n\t"
            "fmadd.ps f21, f16, f15, f21, rne\n\t"
            :
            : [s0] "r"(sb0), [s1] "r"(sb1)
            : "f15", "f20", "f21");
    }
}

// pq2_dot_tile2 with the windowed gather (even chunk addresses only).
static inline void __attribute__((always_inline)) pq2_dot_tile2_w(const block_pq2_0 * q_row0,
                                                                  const block_pq2_0 * q_row1,
                                                                  const float *       b_col,
                                                                  int64_t             n_blocks) {
    static const int32_t __attribute__((aligned(64))) shift_pattern[8] = { 0, 2, 4, 6, 0, 2, 4, 6 };
    const uint64_t                                    gi               = PQ2_GATHER_WINDOW_IDX;
    __asm__ volatile("flw.ps f30, %[s]\n\t" : : [s] "m"(*(const int32_t (*)[8]) shift_pattern) : "f30");
    for (int64_t kb = 0; kb < n_blocks; kb++) {
        const uint8_t * qs0 = q_row0[kb].qs;
        const uint8_t * qs1 = q_row1[kb].qs;
        const float *   b   = b_col + kb * QK_PQ2_0;
        __asm__ volatile("fbci.pi f10, 0\n\tfbci.pi f16, 0" ::: "f10", "f16");
        for (int c = 0; c < 16; c += 4) {
            PQ2V_CHUNK2_W(qs0 + 2 * c + 0, qs1 + 2 * c + 0, b + 8 * c + 0, gi);
            PQ2V_CHUNK2_W(qs0 + 2 * c + 2, qs1 + 2 * c + 2, b + 8 * c + 8, gi);
            PQ2V_CHUNK2_W(qs0 + 2 * c + 4, qs1 + 2 * c + 4, b + 8 * c + 16, gi);
            PQ2V_CHUNK2_W(qs0 + 2 * c + 6, qs1 + 2 * c + 6, b + 8 * c + 24, gi);
        }
        const uint32_t sb0 = fp16_to_fp32_bits(q_row0[kb].d);
        const uint32_t sb1 = fp16_to_fp32_bits(q_row1[kb].d);
        __asm__ volatile(
            "fbcx.ps  f15, %[s0]\n\t"
            "fmadd.ps f20, f10, f15, f20, rne\n\t"
            "fbcx.ps  f15, %[s1]\n\t"
            "fmadd.ps f21, f16, f15, f21, rne\n\t"
            :
            : [s0] "r"(sb0), [s1] "r"(sb1)
            : "f15", "f20", "f21");
    }
}

// Horizontal sums of the two row accumulators (same tree as pq2_dot_reduce, on f20 then f21). The accumulators
// are read implicitly from the tile, so they are declared clobbered: otherwise an output may be allocated onto
// f21 and overwrite row 1 before it is read.
static inline void __attribute__((always_inline)) pq2_dot_reduce2(float * r0, float * r1) {
    float a, b;
    __asm__ __volatile__(
        "fswizz.ps f1, f20, 0xB1 \n\t"
        "fadd.ps   f2, f20, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[v0], f4, f5, rne \n\t"
        "fswizz.ps f1, f21, 0xB1 \n\t"
        "fadd.ps   f2, f21, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[v1], f4, f5, rne \n\t"
        : [v0] "=f"(a), [v1] "=f"(b)::"t0", "f1", "f2", "f3", "f4", "f5", "f20", "f21");
    *r0 = a;
    *r1 = b;
}

#endif  // PQ2_OPS_H
