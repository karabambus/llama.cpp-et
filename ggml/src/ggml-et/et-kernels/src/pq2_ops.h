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
// The windowed tiles also load each chunk's activation vector two chunks (one 64-byte line) ahead into a rotating set
// of four registers (f13, f19, f22, f23; chunk c of a block reads register c % 4): every second activation load misses
// the hart's small L1, and issuing it a line early lets the miss overlap the current chunk's gathers and FMAs instead
// of stalling the FMA that consumes it. The FMA order is unchanged, so results are bit-identical to the fgb.ps tiles.
// The look-ahead of a row's last block points back at the row's first activation chunk (a valid address; the value is
// never used).
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
//   windowed tiles: activation rotation f13, f19, f22, f23 (live across the asm blocks of a row; no compiler float
//   code runs inside the tile loops)

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

// PQ2V_CHUNK with the windowed gather and the activation pipelined: reads the activation already in register RC,
// loads the chunk two ahead (bn) into RN. gi holds PQ2_GATHER_WINDOW_IDX, a must be even.
#define PQ2V_CHUNK_WP(a, RC, RN, bn, gi)                                                    \
    __asm__ volatile(                                                                       \
        "fg32b.ps   f11, %[gi](%[q])\n\t"                                                   \
        "flw.ps     " RN ", %[bv]\n\t"                                                      \
        "fsrl.pi    f12, f11, f30\n\t"                                                      \
        "fandi.pi   f12, f12, 3\n\t"                                                        \
        "faddi.pi   f12, f12, -1\n\t"                                                       \
        "fcvt.ps.pw f12, f12, rne\n\t"                                                      \
        "fmadd.ps   f10, f12, " RC ", f10, rne\n\t"                                         \
        :                                                                                   \
        : [q] "r"(a), [gi] "r"(gi), [bv] "m"(*(const float (*)[8]) (bn))                    \
        : "f10", "f11", "f12", RN)

// Load the first two activation chunks of a row into the rotation (f13 = chunk 0, f19 = chunk 1).
#define PQ2V_ACT_PROLOGUE(b0)                                                               \
    __asm__ volatile(                                                                       \
        "flw.ps f13, %[v0]\n\t"                                                             \
        "flw.ps f19, %[v1]\n\t"                                                             \
        :                                                                                   \
        : [v0] "m"(*(const float (*)[8]) (b0)), [v1] "m"(*(const float (*)[8]) ((b0) + 8))   \
        : "f13", "f19")

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
    if (n_blocks > 0) {
        PQ2V_ACT_PROLOGUE(b_col);
    }

    for (int64_t kb = 0; kb < n_blocks; kb++) {
        const block_pq2_0 * blk = q_row + kb;
        const uint8_t *     qs  = blk->qs;
        const float *       b   = b_col + kb * QK_PQ2_0;

        const float * bn = (kb + 1 < n_blocks) ? b + QK_PQ2_0 : b_col;  // next block's activation (row end: any valid)

        __asm__ volatile("fbci.pi f10, 0" ::: "f10");

        for (int c = 0; c < 12; c += 4) {
            PQ2V_CHUNK_WP(qs + 2 * c + 0, "f13", "f22", b + 8 * c + 16, gi);
            PQ2V_CHUNK_WP(qs + 2 * c + 2, "f19", "f23", b + 8 * c + 24, gi);
            PQ2V_CHUNK_WP(qs + 2 * c + 4, "f22", "f13", b + 8 * c + 32, gi);
            PQ2V_CHUNK_WP(qs + 2 * c + 6, "f23", "f19", b + 8 * c + 40, gi);
        }
        PQ2V_CHUNK_WP(qs + 24, "f13", "f22", b + 112, gi);
        PQ2V_CHUNK_WP(qs + 26, "f19", "f23", b + 120, gi);
        PQ2V_CHUNK_WP(qs + 28, "f22", "f13", bn, gi);
        PQ2V_CHUNK_WP(qs + 30, "f23", "f19", bn + 8, gi);

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

// PQ2V_CHUNK2 with the windowed gather and the activation pipelined (see PQ2V_CHUNK_WP). a0 and a1 must be even.
#define PQ2V_CHUNK2_WP(a0, a1, RC, RN, bn, gi)                                              \
    __asm__ volatile(                                                                       \
        "fg32b.ps   f11, %[gi](%[q0])\n\t"                                                  \
        "fg32b.ps   f17, %[gi](%[q1])\n\t"                                                  \
        "flw.ps     " RN ", %[bv]\n\t"                                                      \
        "fsrl.pi    f12, f11, f30\n\t"                                                      \
        "fsrl.pi    f18, f17, f30\n\t"                                                      \
        "fandi.pi   f12, f12, 3\n\t"                                                        \
        "fandi.pi   f18, f18, 3\n\t"                                                        \
        "faddi.pi   f12, f12, -1\n\t"                                                       \
        "faddi.pi   f18, f18, -1\n\t"                                                       \
        "fcvt.ps.pw f12, f12, rne\n\t"                                                      \
        "fcvt.ps.pw f18, f18, rne\n\t"                                                      \
        "fmadd.ps   f10, f12, " RC ", f10, rne\n\t"                                         \
        "fmadd.ps   f16, f18, " RC ", f16, rne\n\t"                                         \
        :                                                                                   \
        : [q0] "r"(a0), [q1] "r"(a1), [gi] "r"(gi), [bv] "m"(*(const float (*)[8]) (bn))     \
        : "f10", "f11", "f12", "f16", "f17", "f18", RN)

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
    if (n_blocks > 0) {
        PQ2V_ACT_PROLOGUE(b_col);
    }
    for (int64_t kb = 0; kb < n_blocks; kb++) {
        const uint8_t * qs0 = q_row0[kb].qs;
        const uint8_t * qs1 = q_row1[kb].qs;
        const float *   b   = b_col + kb * QK_PQ2_0;
        const float *   bn  = (kb + 1 < n_blocks) ? b + QK_PQ2_0 : b_col;  // next block's activation (row end: any valid)
        __asm__ volatile("fbci.pi f10, 0\n\tfbci.pi f16, 0" ::: "f10", "f16");
        for (int c = 0; c < 12; c += 4) {
            PQ2V_CHUNK2_WP(qs0 + 2 * c + 0, qs1 + 2 * c + 0, "f13", "f22", b + 8 * c + 16, gi);
            PQ2V_CHUNK2_WP(qs0 + 2 * c + 2, qs1 + 2 * c + 2, "f19", "f23", b + 8 * c + 24, gi);
            PQ2V_CHUNK2_WP(qs0 + 2 * c + 4, qs1 + 2 * c + 4, "f22", "f13", b + 8 * c + 32, gi);
            PQ2V_CHUNK2_WP(qs0 + 2 * c + 6, qs1 + 2 * c + 6, "f23", "f19", b + 8 * c + 40, gi);
        }
        PQ2V_CHUNK2_WP(qs0 + 24, qs1 + 24, "f13", "f22", b + 112, gi);
        PQ2V_CHUNK2_WP(qs0 + 26, qs1 + 26, "f19", "f23", b + 120, gi);
        PQ2V_CHUNK2_WP(qs0 + 28, qs1 + 28, "f22", "f13", bn, gi);
        PQ2V_CHUNK2_WP(qs0 + 30, qs1 + 30, "f23", "f19", bn + 8, gi);
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

// Two token columns per weight pass (windowed path, even chunk addresses only). A batch of N > 1 tokens (speculative
// verification, prefill) used to stream every weight row once per column; these tiles gather and unpack each chunk of
// codes once and apply it to two columns. Each (row, column) keeps its own block accumulator and sees chunks 0..15
// and the block scale in the same order as the one-column tiles, so results are bit-identical per column. Both
// columns' activations are loaded two chunks ahead as in PQ2V_CHUNK_WP: column 0 rotates f13, f19, f22, f23 and
// column 1 rotates f24, f25, f26, f27. At most two weight lines and two activation lines are live per L1 set (a
// hart's rows, and the two columns, are each a multiple of 128 bytes apart, so they share a set).
// Register contract (live across the asm blocks of a tile; no compiler float code inside the loops, KB 04 G2a):
//   block accumulators: row0/col0 f10, row1/col0 f16, row0/col1 f6, row1/col1 f7
//   row accumulators:   row0/col0 f20, row1/col0 f21, row0/col1 f28, row1/col1 f29
//   gather scratch f11/f12 (row 0), f17/f18 (row 1); shift pattern f30; scale f15
#define PQ2V_CH22(a0, a1, RC0, RN0, bn0, RC1, RN1, bn1, gi)                                 \
    __asm__ volatile(                                                                       \
        "fg32b.ps   f11, %[gi](%[q0])\n\t"                                                  \
        "fg32b.ps   f17, %[gi](%[q1])\n\t"                                                  \
        "flw.ps     " RN0 ", %[v0]\n\t"                                                     \
        "flw.ps     " RN1 ", %[v1]\n\t"                                                     \
        "fsrl.pi    f12, f11, f30\n\t"                                                      \
        "fsrl.pi    f18, f17, f30\n\t"                                                      \
        "fandi.pi   f12, f12, 3\n\t"                                                        \
        "fandi.pi   f18, f18, 3\n\t"                                                        \
        "faddi.pi   f12, f12, -1\n\t"                                                       \
        "faddi.pi   f18, f18, -1\n\t"                                                       \
        "fcvt.ps.pw f12, f12, rne\n\t"                                                      \
        "fcvt.ps.pw f18, f18, rne\n\t"                                                      \
        "fmadd.ps   f10, f12, " RC0 ", f10, rne\n\t"                                        \
        "fmadd.ps   f16, f18, " RC0 ", f16, rne\n\t"                                        \
        "fmadd.ps   f6, f12, " RC1 ", f6, rne\n\t"                                          \
        "fmadd.ps   f7, f18, " RC1 ", f7, rne\n\t"                                          \
        :                                                                                   \
        : [q0] "r"(a0), [q1] "r"(a1), [gi] "r"(gi), [v0] "m"(*(const float (*)[8]) (bn0)),  \
          [v1] "m"(*(const float (*)[8]) (bn1))                                             \
        : "f10", "f11", "f12", "f16", "f17", "f18", "f6", "f7", RN0, RN1)

#define PQ2V_CH12(a0, RC0, RN0, bn0, RC1, RN1, bn1, gi)                                     \
    __asm__ volatile(                                                                       \
        "fg32b.ps   f11, %[gi](%[q0])\n\t"                                                  \
        "flw.ps     " RN0 ", %[v0]\n\t"                                                     \
        "flw.ps     " RN1 ", %[v1]\n\t"                                                     \
        "fsrl.pi    f12, f11, f30\n\t"                                                      \
        "fandi.pi   f12, f12, 3\n\t"                                                        \
        "faddi.pi   f12, f12, -1\n\t"                                                       \
        "fcvt.ps.pw f12, f12, rne\n\t"                                                      \
        "fmadd.ps   f10, f12, " RC0 ", f10, rne\n\t"                                        \
        "fmadd.ps   f6, f12, " RC1 ", f6, rne\n\t"                                          \
        :                                                                                   \
        : [q0] "r"(a0), [gi] "r"(gi), [v0] "m"(*(const float (*)[8]) (bn0)),                \
          [v1] "m"(*(const float (*)[8]) (bn1))                                             \
        : "f10", "f11", "f12", "f6", RN0, RN1)

// First two activation chunks of both columns (column 0 -> f13, f19; column 1 -> f24, f25).
#define PQ2V_ACT_PROLOGUE2(c0, c1)                                                          \
    __asm__ volatile(                                                                       \
        "flw.ps f13, %[a0]\n\t"                                                             \
        "flw.ps f19, %[a1]\n\t"                                                             \
        "flw.ps f24, %[b0]\n\t"                                                             \
        "flw.ps f25, %[b1]\n\t"                                                             \
        :                                                                                   \
        : [a0] "m"(*(const float (*)[8]) (c0)), [a1] "m"(*(const float (*)[8]) ((c0) + 8)),  \
          [b0] "m"(*(const float (*)[8]) (c1)), [b1] "m"(*(const float (*)[8]) ((c1) + 8))   \
        : "f13", "f19", "f24", "f25")

// One block for a 2x2 or 1x2 tile: CH is PQ2V_CH22 or PQ2V_CH12 with its row arguments bound by the caller macro.
#define PQ2V_BLOCK22(qs0, qs1, b0, b1, bn0, bn1, gi)                                                         \
    do {                                                                                                     \
        for (int c = 0; c < 12; c += 4) {                                                                    \
            PQ2V_CH22((qs0) + 2 * c + 0, (qs1) + 2 * c + 0, "f13", "f22", (b0) + 8 * c + 16, "f24", "f26", (b1) + 8 * c + 16, gi); \
            PQ2V_CH22((qs0) + 2 * c + 2, (qs1) + 2 * c + 2, "f19", "f23", (b0) + 8 * c + 24, "f25", "f27", (b1) + 8 * c + 24, gi); \
            PQ2V_CH22((qs0) + 2 * c + 4, (qs1) + 2 * c + 4, "f22", "f13", (b0) + 8 * c + 32, "f26", "f24", (b1) + 8 * c + 32, gi); \
            PQ2V_CH22((qs0) + 2 * c + 6, (qs1) + 2 * c + 6, "f23", "f19", (b0) + 8 * c + 40, "f27", "f25", (b1) + 8 * c + 40, gi); \
        }                                                                                                    \
        PQ2V_CH22((qs0) + 24, (qs1) + 24, "f13", "f22", (b0) + 112, "f24", "f26", (b1) + 112, gi);           \
        PQ2V_CH22((qs0) + 26, (qs1) + 26, "f19", "f23", (b0) + 120, "f25", "f27", (b1) + 120, gi);           \
        PQ2V_CH22((qs0) + 28, (qs1) + 28, "f22", "f13", (bn0), "f26", "f24", (bn1), gi);                     \
        PQ2V_CH22((qs0) + 30, (qs1) + 30, "f23", "f19", (bn0) + 8, "f27", "f25", (bn1) + 8, gi);             \
    } while (0)

#define PQ2V_BLOCK12(qs0, b0, b1, bn0, bn1, gi)                                                              \
    do {                                                                                                     \
        for (int c = 0; c < 12; c += 4) {                                                                    \
            PQ2V_CH12((qs0) + 2 * c + 0, "f13", "f22", (b0) + 8 * c + 16, "f24", "f26", (b1) + 8 * c + 16, gi); \
            PQ2V_CH12((qs0) + 2 * c + 2, "f19", "f23", (b0) + 8 * c + 24, "f25", "f27", (b1) + 8 * c + 24, gi); \
            PQ2V_CH12((qs0) + 2 * c + 4, "f22", "f13", (b0) + 8 * c + 32, "f26", "f24", (b1) + 8 * c + 32, gi); \
            PQ2V_CH12((qs0) + 2 * c + 6, "f23", "f19", (b0) + 8 * c + 40, "f27", "f25", (b1) + 8 * c + 40, gi); \
        }                                                                                                    \
        PQ2V_CH12((qs0) + 24, "f13", "f22", (b0) + 112, "f24", "f26", (b1) + 112, gi);                       \
        PQ2V_CH12((qs0) + 26, "f19", "f23", (b0) + 120, "f25", "f27", (b1) + 120, gi);                       \
        PQ2V_CH12((qs0) + 28, "f22", "f13", (bn0), "f26", "f24", (bn1), gi);                                 \
        PQ2V_CH12((qs0) + 30, "f23", "f19", (bn0) + 8, "f27", "f25", (bn1) + 8, gi);                         \
    } while (0)

static inline void __attribute__((always_inline)) pq2_dot_reset22(void) {
    __asm__ volatile("fbci.pi f20, 0\n\tfbci.pi f21, 0\n\tfbci.pi f28, 0\n\tfbci.pi f29, 0" ::: "f20", "f21", "f28", "f29");
}

// Rows q_row0/q_row1 against columns c0/c1 into f20 (r0c0), f21 (r1c0), f28 (r0c1), f29 (r1c1).
static inline void __attribute__((always_inline)) pq2_dot_tile22_w(const block_pq2_0 * q_row0,
                                                                   const block_pq2_0 * q_row1,
                                                                   const float *       c0,
                                                                   const float *       c1,
                                                                   int64_t             n_blocks) {
    static const int32_t __attribute__((aligned(64))) shift_pattern[8] = { 0, 2, 4, 6, 0, 2, 4, 6 };
    const uint64_t                                    gi               = PQ2_GATHER_WINDOW_IDX;
    __asm__ volatile("flw.ps f30, %[s]\n\t" : : [s] "m"(*(const int32_t (*)[8]) shift_pattern) : "f30");
    if (n_blocks > 0) {
        PQ2V_ACT_PROLOGUE2(c0, c1);
    }
    for (int64_t kb = 0; kb < n_blocks; kb++) {
        const uint8_t * qs0 = q_row0[kb].qs;
        const uint8_t * qs1 = q_row1[kb].qs;
        const float *   b0  = c0 + kb * QK_PQ2_0;
        const float *   b1  = c1 + kb * QK_PQ2_0;
        const float *   bn0 = (kb + 1 < n_blocks) ? b0 + QK_PQ2_0 : c0;  // next block's activation (row end: any valid)
        const float *   bn1 = (kb + 1 < n_blocks) ? b1 + QK_PQ2_0 : c1;
        __asm__ volatile("fbci.pi f10, 0\n\tfbci.pi f16, 0\n\tfbci.pi f6, 0\n\tfbci.pi f7, 0" ::: "f10", "f16", "f6", "f7");
        PQ2V_BLOCK22(qs0, qs1, b0, b1, bn0, bn1, gi);
        const uint32_t sb0 = fp16_to_fp32_bits(q_row0[kb].d);
        const uint32_t sb1 = fp16_to_fp32_bits(q_row1[kb].d);
        __asm__ volatile(
            "fbcx.ps  f15, %[s0]\n\t"
            "fmadd.ps f20, f10, f15, f20, rne\n\t"
            "fmadd.ps f28, f6, f15, f28, rne\n\t"
            "fbcx.ps  f15, %[s1]\n\t"
            "fmadd.ps f21, f16, f15, f21, rne\n\t"
            "fmadd.ps f29, f7, f15, f29, rne\n\t"
            :
            : [s0] "r"(sb0), [s1] "r"(sb1)
            : "f15", "f20", "f21", "f28", "f29");
    }
}

// Row q_row0 against columns c0/c1 into f20 (c0) and f28 (c1).
static inline void __attribute__((always_inline)) pq2_dot_tile12_w(const block_pq2_0 * q_row0,
                                                                   const float *       c0,
                                                                   const float *       c1,
                                                                   int64_t             n_blocks) {
    static const int32_t __attribute__((aligned(64))) shift_pattern[8] = { 0, 2, 4, 6, 0, 2, 4, 6 };
    const uint64_t                                    gi               = PQ2_GATHER_WINDOW_IDX;
    __asm__ volatile("flw.ps f30, %[s]\n\t" : : [s] "m"(*(const int32_t (*)[8]) shift_pattern) : "f30");
    if (n_blocks > 0) {
        PQ2V_ACT_PROLOGUE2(c0, c1);
    }
    for (int64_t kb = 0; kb < n_blocks; kb++) {
        const uint8_t * qs0 = q_row0[kb].qs;
        const float *   b0  = c0 + kb * QK_PQ2_0;
        const float *   b1  = c1 + kb * QK_PQ2_0;
        const float *   bn0 = (kb + 1 < n_blocks) ? b0 + QK_PQ2_0 : c0;
        const float *   bn1 = (kb + 1 < n_blocks) ? b1 + QK_PQ2_0 : c1;
        __asm__ volatile("fbci.pi f10, 0\n\tfbci.pi f6, 0" ::: "f10", "f6");
        PQ2V_BLOCK12(qs0, b0, b1, bn0, bn1, gi);
        const uint32_t sb0 = fp16_to_fp32_bits(q_row0[kb].d);
        __asm__ volatile(
            "fbcx.ps  f15, %[s0]\n\t"
            "fmadd.ps f20, f10, f15, f20, rne\n\t"
            "fmadd.ps f28, f6, f15, f28, rne\n\t"
            :
            : [s0] "r"(sb0)
            : "f15", "f20", "f28");
    }
}

// Horizontal sum of one 8-lane accumulator (same tree as pq2_dot_reduce).
#define PQ2V_HSUM(ACC, OUT)                                                                 \
    __asm__ __volatile__(                                                                   \
        "fswizz.ps f1, " ACC ", 0xB1 \n\t"                                                  \
        "fadd.ps   f2, " ACC ", f1, rne \n\t"                                               \
        "fswizz.ps f3, f2, 0x4E \n\t"                                                       \
        "fadd.ps   f4, f2, f3, rne \n\t"                                                    \
        "fmvz.x.ps t0, f4, 4 \n\t"                                                          \
        "fbcx.ps   f5, t0 \n\t"                                                             \
        "fadd.ps   %[v], f4, f5, rne \n\t"                                                  \
        : [v] "=f"(OUT)::"t0", "f1", "f2", "f3", "f4", "f5", "f20", "f21", "f28", "f29")

#endif  // PQ2_OPS_H
