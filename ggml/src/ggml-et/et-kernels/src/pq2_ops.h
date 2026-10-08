// PQ2_0 (Prism ternary, group 128) row-dot helpers for the vector lanes.
//
// Block layout (ggml-common.h): { ggml_half d; uint8_t qs[32]; } = 34 bytes. Element j of the block sits in
// byte j/4 at bit offset 2*(j%4), LSB first; the 2-bit code q maps to the weight (q - 1), i.e. 00 = -1, 01 = 0,
// 10 = +1 (11 = +2 is never produced by the ternary packer but decodes the same way as on the CPU).
//
// Unpacking 8 consecutive elements into 8 lanes: a byte gather with the pattern {0,0,0,0,1,1,1,1} places byte 2c
// in lanes 0..3 and byte 2c+1 in lanes 4..7; a per-lane logical shift by {0,2,4,6,0,2,4,6} and a mask of 3 then
// leave the lane's own 2-bit code. The gather sign-extends bytes >= 0x80, which the logical shift and mask make
// irrelevant. Only fgb.ps is used for the bytes: blocks are 34 bytes long, so qs is never 32-byte aligned and the
// windowed fg32b.ps precondition (address & 31 <= 24) would not hold for every block.
//
// The fp16 block scale is converted in integer code (fp16_to_fp32_bits) and broadcast with fbcx.ps; fcvt.ps.f16
// flushes subnormal halves and must not touch block scales (KB 02 section 5.1).
//
// Register contract:
//   f20       - row accumulator (reset per row by pq2_dot_reset, read by pq2_dot_reduce)
//   f31       - byte gather pattern, f30 - per-lane shift amounts (both loaded by pq2_dot_tile)
//   f10-f15   - scratch inside the tile
//   f1-f5, t0 - scratch inside the reduce

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
        : [vout] "=f"(result)::"t0", "f1", "f2", "f3", "f4", "f5");
    return result;
}

// Full-row dot product; the caller owns the lane mask (set to 0xFF around the row loop).
static inline float pq2_dot_compute(const block_pq2_0 * q_row, const float * b_col, int64_t K_blocks) {
    pq2_dot_reset();
    pq2_dot_tile(q_row, b_col, K_blocks);
    return pq2_dot_reduce();
}

#endif  // PQ2_OPS_H
