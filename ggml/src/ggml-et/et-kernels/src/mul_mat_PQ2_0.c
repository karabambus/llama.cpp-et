//******************************************************************************
// MUL_MAT kernel for PQ2_0 weights (Prism ternary, one fp16 scale per 128 weights)
// C[M,N] = A[M,K] * B[K,N], F32 activations and output.
//
// Rows are striped over all launched harts (m = hart_id, hart_id + 2048, ...); every hart computes whole dot
// products, two rows interleaved so two weight lines are in flight per hart (the stream is latency-bound at one),
// so no partial sums cross harts and each output element is published with one global atomic store.
// The dot product unpacks the 2-bit codes in the vector lanes (pq2_ops.h). K must be a multiple of 128
// (supports_op guarantees it; the kernel refuses otherwise instead of reading past the row).
//******************************************************************************

#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"
#include "pq2_ops.h"
#include "quants.h"

#include <stdint.h>

#define STRIDE_M 2048 /* 32 shires x 32 minions x 2 harts */

static inline size_t tensor_bytes(const struct ggml_tensor * t) {
    return (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

// Row pairing for the 2x3 tile. A hart's rows are STRIDE_M rows apart, a multiple of 128 bytes, so both rows' weight lines
// fall into the same set of the hart's 2-set x 4-way L1; with three activation lines in that set as well (the columns are
// also a multiple of 128 bytes apart) the set would need five ways. Pairing row r with row r + d, where d * nb01 is 64
// modulo 128, puts the two weight lines into opposite sets at every chunk. Returns d, or 0 when no d <= 64 exists.
static inline int64_t pq2_pair_offset(size_t nb01) {
    for (int64_t d = 1; d <= 64; d++) {
        if (((size_t) d * nb01) % 128 == 64) {
            return d;
        }
    }
    return 0;
}

// One 2x3 tile (rows r0, r1 against columns c0..c2) including the horizontal sums, bias and output stores.
static inline void __attribute__((always_inline)) pq2_rows23(const char * src0_ptr2, size_t nb01, int64_t r0, int64_t r1,
                                                             const float * b_c0, const float * b_c1, const float * b_c2,
                                                             const float * bias_c0, const float * bias_c1,
                                                             const float * bias_c2, float * d_c0, float * d_c1, float * d_c2,
                                                             int64_t K_blocks) {
    const block_pq2_0 * q_row0 = (const block_pq2_0 *) (src0_ptr2 + r0 * nb01);
    const block_pq2_0 * q_row1 = (const block_pq2_0 *) (src0_ptr2 + r1 * nb01);
    float               r0c0, r1c0, r0c1, r1c1, r0c2, r1c2;
    pq2_dot_reset23();
    pq2_dot_tile23_w(q_row0, q_row1, b_c0, b_c1, b_c2, K_blocks);
    PQ2V_HSUM("f20", r0c0);
    PQ2V_HSUM("f21", r1c0);
    PQ2V_HSUM("f28", r0c1);
    PQ2V_HSUM("f29", r1c1);
    PQ2V_HSUM("f14", r0c2);
    PQ2V_HSUM("f31", r1c2);
    if (bias_c0) {
        r0c0 += bias_c0[r0];
        r1c0 += bias_c0[r1];
        r0c1 += bias_c1[r0];
        r1c1 += bias_c1[r1];
        r0c2 += bias_c2[r0];
        r1c2 += bias_c2[r1];
    }
    atomic_store_f32((volatile float *) (d_c0 + r0), r0c0);
    atomic_store_f32((volatile float *) (d_c0 + r1), r1c0);
    atomic_store_f32((volatile float *) (d_c1 + r0), r0c1);
    atomic_store_f32((volatile float *) (d_c1 + r1), r1c1);
    atomic_store_f32((volatile float *) (d_c2 + r0), r0c2);
    atomic_store_f32((volatile float *) (d_c2 + r1), r1c2);
}

// K-split tail: tail row t (row tail0 + t, t < rem) is computed as two K halves, half 0 on hart t and half 1 on hart t + rem,
// for the one column pass of an N = 1, 2 or 3 launch. Each half publishes its ncol partials in the row's record with
// global atomics, then swaps the record's flag to this launch's generation; the half that finds the generation already
// there is the second, and it stores h0 + h1 (+ bias) per column in that fixed order, so the result does not depend on
// which half finished first. Records are 32 bytes {flag, c0h0, c0h1, c1h0, c1h1, c2h0, c2h1, -}; every access is a global
// atomic, and a flag left by an earlier launch never equals a later generation.
#define PQ2_TAIL_REC_BYTES 32

static inline void pq2_tail_publish(const struct ggml_et_mm_q8_params * params, int64_t t, int64_t half, int ncol,
                                    const float * part, const float * const * bias_c, float * const * d_c, int64_t row) {
    volatile uint32_t * rec = (volatile uint32_t *) (uintptr_t) (params->tail_scratch + (uint64_t) t * PQ2_TAIL_REC_BYTES);
    for (int c = 0; c < ncol; c++) {
        uint32_t bits;
        __builtin_memcpy(&bits, &part[c], sizeof bits);
        et_global_swap_w(rec + 1 + 2 * c + half, bits);
    }
    FENCE;
    if (et_global_swap_w(rec, params->tail_gen) != params->tail_gen) {
        return;  // first half: the other one combines
    }
    for (int c = 0; c < ncol; c++) {
        const uint32_t b0 = et_global_add_w(rec + 1 + 2 * c, 0);
        const uint32_t b1 = et_global_add_w(rec + 2 + 2 * c, 0);
        float          p0, p1;
        __builtin_memcpy(&p0, &b0, sizeof p0);
        __builtin_memcpy(&p1, &b1, sizeof p1);
        float sum = p0 + p1;
        if (bias_c[c]) {
            sum += bias_c[c][row];
        }
        atomic_store_f32((volatile float *) (d_c[c] + row), sum);
    }
}

// The half of tail row t that hart h computes: blocks [kb0, kb0 + nkb). Returns 0 when h has no tail work.
static inline int pq2_tail_part(uint64_t hart_id, int64_t rem, int64_t K_blocks, int64_t * t, int64_t * half, int64_t * kb0,
                                int64_t * nkb) {
    const int64_t h = (int64_t) hart_id;
    if (h >= 2 * rem) {
        return 0;
    }
    const int64_t kb_h = (K_blocks + 1) / 2;
    *t    = h % rem;
    *half = h / rem;
    *kb0  = *half ? kb_h : 0;
    *nkb  = *half ? K_blocks - kb_h : kb_h;
    return 1;
}

static inline void pq2_tail_ksplit(const struct ggml_et_mm_q8_params * params, const char * src0_ptr2, size_t nb01,
                                   const float * b_col, const float * bias_n, float * c_col, int64_t tail0, int64_t rem,
                                   int64_t K_blocks, int windowed, uint64_t hart_id) {
    int64_t t, half, kb0, nkb;
    if (!pq2_tail_part(hart_id, rem, K_blocks, &t, &half, &kb0, &nkb)) {
        return;
    }
    const int64_t       row    = tail0 + t;
    const block_pq2_0 * q_part = (const block_pq2_0 *) (src0_ptr2 + row * nb01) + kb0;
    const float *       b_part = b_col + kb0 * QK_PQ2_0;
    const float part = windowed ? pq2_dot_compute_w(q_part, b_part, nkb) : pq2_dot_compute(q_part, b_part, nkb);
    pq2_tail_publish(params, t, half, 1, &part, &bias_n, &c_col, row);
}

int entry_point(struct ggml_et_mm_q8_params * params, void * env) {
    (void) env;
    const uint64_t hart_id = get_hart_id();

    const int64_t K    = params->src0.ne[0];
    const int64_t M    = params->src0.ne[1];
    const int64_t N    = params->src1.ne[1];
    const int64_t ne02 = params->src0.ne[2];
    const int64_t ne03 = params->src0.ne[3];
    const int64_t ne12 = params->src1.ne[2];
    const int64_t ne13 = params->src1.ne[3];

    const size_t nb01 = params->src0.nb[1];
    const size_t nb02 = params->src0.nb[2];
    const size_t nb03 = params->src0.nb[3];

    const size_t nb11 = params->src1.nb[1];
    const size_t nb12 = params->src1.nb[2];
    const size_t nb13 = params->src1.nb[3];

    const size_t nbd1 = params->dst.nb[1];
    const size_t nbd2 = params->dst.nb[2];
    const size_t nbd3 = params->dst.nb[3];

    // Optional residual bias (same shape and strides as dst); bias.data == NULL means none.
    const char * bias_base = (const char *) params->bias.data;
    const size_t nbb1      = params->bias.nb[1];
    const size_t nbb2      = params->bias.nb[2];
    const size_t nbb3      = params->bias.nb[3];

    if (K % QK_PQ2_0 != 0) {
        return -1;
    }
    const int64_t K_blocks = K / QK_PQ2_0;

    // Broadcasting ratios
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    // No entry evict: the uberkernel's dispatcher publishes src1 and the bias; after a standalone launch the firmware's
    // cleanup leaves no input line in any L1 or L2 (KB 04 §1.1).

    // The windowed gather (fg32b.ps) is bit-identical to fgb.ps only at even chunk addresses: blocks are 34 bytes,
    // so every chunk address is even exactly when the weight base and the row/batch strides are even (pq2_ops.h).
    const int windowed = ((((uintptr_t) params->src0.data) | nb01 | nb02 | nb03) & 1) == 0;
    const int64_t pair_d = pq2_pair_offset(nb01);

    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int64_t i3 = 0; i3 < ne13; i3++) {
        const int64_t i03       = i3 / r3;
        const char *  src0_ptr3 = (const char *) params->src0.data + i03 * nb03;
        const char *  src1_ptr3 = (const char *) params->src1.data + i3 * nb13;
        char *        dst_ptr3  = (char *) params->dst.data + i3 * nbd3;
        const char *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (const char *) 0;

        for (int64_t i2 = 0; i2 < ne12; i2++) {
            const int64_t i02       = i2 / r2;
            const char *  src0_ptr2 = src0_ptr3 + i02 * nb02;
            const char *  src1_ptr2 = src1_ptr3 + i2 * nb12;
            char *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
            const char *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (const char *) 0;

            int64_t n = 0;
            // Two token columns per weight pass on the windowed path (each weight line is loaded once for both); when
            // exactly three columns remain (odd N > 1: a 3-token speculative verification, or the tail of N = 5, 7)
            // they share one pass instead of a pair pass plus a separate one-column pass.
            while (windowed && n + 1 < N) {
                if (N - n == 3) {
                    const float * b_c0    = (const float *) (src1_ptr2 + n * nb11);
                    const float * b_c1    = (const float *) (src1_ptr2 + (n + 1) * nb11);
                    const float * b_c2    = (const float *) (src1_ptr2 + (n + 2) * nb11);
                    const float * bias_c0 = bias_ptr2 ? (const float *) (bias_ptr2 + n * nbb1) : (const float *) 0;
                    const float * bias_c1 = bias_ptr2 ? (const float *) (bias_ptr2 + (n + 1) * nbb1) : (const float *) 0;
                    const float * bias_c2 = bias_ptr2 ? (const float *) (bias_ptr2 + (n + 2) * nbb1) : (const float *) 0;
                    float *       d_c0    = (float *) (dst_ptr2 + n * nbd1);
                    float *       d_c1    = (float *) (dst_ptr2 + (n + 1) * nbd1);
                    float *       d_c2    = (float *) (dst_ptr2 + (n + 2) * nbd1);

                    // Region A (whole rounds of 2 * STRIDE_M rows): pair slot p = hart_id + k * STRIDE_M maps to rows r0 and
                    // r0 + pair_d in opposite L1 sets, a bijection on the region; every hart keeps its row count. The rest
                    // of the rows keep the STRIDE_M pairing (plus single rows) of the other tiles.
                    const int64_t full = pair_d ? M / (2 * STRIDE_M) : 0;
                    for (int64_t k = 0; k < full; k++) {
                        const int64_t p  = k * STRIDE_M + (int64_t) hart_id;
                        const int64_t r0 = (p / pair_d) * 2 * pair_d + (p % pair_d);
                        pq2_rows23(src0_ptr2, nb01, r0, r0 + pair_d, b_c0, b_c1, b_c2, bias_c0, bias_c1, bias_c2, d_c0, d_c1,
                                   d_c2, K_blocks);
                    }
                    int64_t m = full * 2 * STRIDE_M + (int64_t) hart_id;
                    for (; m + STRIDE_M < M; m += 2 * STRIDE_M) {
                        pq2_rows23(src0_ptr2, nb01, m, m + STRIDE_M, b_c0, b_c1, b_c2, bias_c0, bias_c1, bias_c2, d_c0, d_c1,
                                   d_c2, K_blocks);
                    }
                    const int64_t rem3 = M % (2 * STRIDE_M);
                    if (params->tail_scratch && N == 3 && ne12 * ne13 == 1 && K_blocks >= 2 && rem3 > 0 &&
                        rem3 <= STRIDE_M / 2 && m >= M - rem3) {
                        int64_t t, half, kb0, nkb;
                        if (pq2_tail_part(hart_id, rem3, K_blocks, &t, &half, &kb0, &nkb)) {
                            const int64_t       row    = M - rem3 + t;
                            const block_pq2_0 * q_part = (const block_pq2_0 *) (src0_ptr2 + row * nb01) + kb0;
                            float               part[3];
                            pq2_dot_reset23();
                            pq2_dot_tile13_w(q_part, b_c0 + kb0 * QK_PQ2_0, b_c1 + kb0 * QK_PQ2_0, b_c2 + kb0 * QK_PQ2_0, nkb);
                            PQ2V_HSUM("f20", part[0]);
                            PQ2V_HSUM("f28", part[1]);
                            PQ2V_HSUM("f14", part[2]);
                            const float * const bias3[3] = { bias_c0, bias_c1, bias_c2 };
                            float * const       dst3[3]  = { d_c0, d_c1, d_c2 };
                            pq2_tail_publish(params, t, half, 3, part, bias3, dst3, row);
                        }
                        m = M;
                    }
                    for (; m < M; m += STRIDE_M) {
                        const block_pq2_0 * q_row = (const block_pq2_0 *) (src0_ptr2 + m * nb01);
                        float               r0c0, r0c1, r0c2;
                        pq2_dot_reset23();
                        pq2_dot_tile13_w(q_row, b_c0, b_c1, b_c2, K_blocks);
                        PQ2V_HSUM("f20", r0c0);
                        PQ2V_HSUM("f28", r0c1);
                        PQ2V_HSUM("f14", r0c2);
                        if (bias_c0) {
                            r0c0 += bias_c0[m];
                            r0c1 += bias_c1[m];
                            r0c2 += bias_c2[m];
                        }
                        atomic_store_f32((volatile float *) (d_c0 + m), r0c0);
                        atomic_store_f32((volatile float *) (d_c1 + m), r0c1);
                        atomic_store_f32((volatile float *) (d_c2 + m), r0c2);
                    }
                    n += 3;
                    continue;
                }
                const float * b_c0    = (const float *) (src1_ptr2 + n * nb11);
                const float * b_c1    = (const float *) (src1_ptr2 + (n + 1) * nb11);
                const float * bias_c0 = bias_ptr2 ? (const float *) (bias_ptr2 + n * nbb1) : (const float *) 0;
                const float * bias_c1 = bias_ptr2 ? (const float *) (bias_ptr2 + (n + 1) * nbb1) : (const float *) 0;
                float *       d_c0    = (float *) (dst_ptr2 + n * nbd1);
                float *       d_c1    = (float *) (dst_ptr2 + (n + 1) * nbd1);

                int64_t m = (int64_t) hart_id;
                for (; m + STRIDE_M < M; m += 2 * STRIDE_M) {
                    const block_pq2_0 * q_row0 = (const block_pq2_0 *) (src0_ptr2 + m * nb01);
                    const block_pq2_0 * q_row1 = (const block_pq2_0 *) (src0_ptr2 + (m + STRIDE_M) * nb01);
                    float               r0c0, r1c0, r0c1, r1c1;
                    pq2_dot_reset22();
                    pq2_dot_tile22_w(q_row0, q_row1, b_c0, b_c1, K_blocks);
                    PQ2V_HSUM("f20", r0c0);
                    PQ2V_HSUM("f21", r1c0);
                    PQ2V_HSUM("f28", r0c1);
                    PQ2V_HSUM("f29", r1c1);
                    if (bias_c0) {
                        r0c0 += bias_c0[m];
                        r1c0 += bias_c0[m + STRIDE_M];
                        r0c1 += bias_c1[m];
                        r1c1 += bias_c1[m + STRIDE_M];
                    }
                    atomic_store_f32((volatile float *) (d_c0 + m), r0c0);
                    atomic_store_f32((volatile float *) (d_c0 + m + STRIDE_M), r1c0);
                    atomic_store_f32((volatile float *) (d_c1 + m), r0c1);
                    atomic_store_f32((volatile float *) (d_c1 + m + STRIDE_M), r1c1);
                }
                const int64_t rem2 = M % (2 * STRIDE_M);
                if (params->tail_scratch && N == 2 && ne12 * ne13 == 1 && K_blocks >= 2 && rem2 > 0 &&
                    rem2 <= STRIDE_M / 2 && m >= M - rem2) {
                    int64_t t, half, kb0, nkb;
                    if (pq2_tail_part(hart_id, rem2, K_blocks, &t, &half, &kb0, &nkb)) {
                        const int64_t       row    = M - rem2 + t;
                        const block_pq2_0 * q_part = (const block_pq2_0 *) (src0_ptr2 + row * nb01) + kb0;
                        float               part[2];
                        pq2_dot_reset22();
                        pq2_dot_tile12_w(q_part, b_c0 + kb0 * QK_PQ2_0, b_c1 + kb0 * QK_PQ2_0, nkb);
                        PQ2V_HSUM("f20", part[0]);
                        PQ2V_HSUM("f28", part[1]);
                        const float * const bias2[2] = { bias_c0, bias_c1 };
                        float * const       dst2[2]  = { d_c0, d_c1 };
                        pq2_tail_publish(params, t, half, 2, part, bias2, dst2, row);
                    }
                    m = M;
                }
                for (; m < M; m += STRIDE_M) {
                    const block_pq2_0 * q_row = (const block_pq2_0 *) (src0_ptr2 + m * nb01);
                    float               r0c0, r0c1;
                    pq2_dot_reset22();
                    pq2_dot_tile12_w(q_row, b_c0, b_c1, K_blocks);
                    PQ2V_HSUM("f20", r0c0);
                    PQ2V_HSUM("f28", r0c1);
                    if (bias_c0) {
                        r0c0 += bias_c0[m];
                        r0c1 += bias_c1[m];
                    }
                    atomic_store_f32((volatile float *) (d_c0 + m), r0c0);
                    atomic_store_f32((volatile float *) (d_c1 + m), r0c1);
                }
                n += 2;
            }
            for (; n < N; n++) {
                const float * b_col  = (const float *) (src1_ptr2 + n * nb11);
                const float * bias_n = bias_ptr2 ? (const float *) (bias_ptr2 + n * nbb1) : (const float *) 0;
                float *       c_col  = (float *) (dst_ptr2 + n * nbd1);

                // Two of this hart's rows at a time (two lines in flight per hart), then the odd row if any.
                int64_t m = (int64_t) hart_id;
                for (; m + STRIDE_M < M; m += 2 * STRIDE_M) {
                    const block_pq2_0 * q_row0 = (const block_pq2_0 *) (src0_ptr2 + m * nb01);
                    const block_pq2_0 * q_row1 = (const block_pq2_0 *) (src0_ptr2 + (m + STRIDE_M) * nb01);
                    float               s0, s1;
                    pq2_dot_reset2();
                    if (windowed) {
                        pq2_dot_tile2_w(q_row0, q_row1, b_col, K_blocks);
                    } else {
                        pq2_dot_tile2(q_row0, q_row1, b_col, K_blocks);
                    }
                    pq2_dot_reduce2(&s0, &s1);
                    if (bias_n) {
                        s0 += bias_n[m];
                        s1 += bias_n[m + STRIDE_M];
                    }
                    atomic_store_f32((volatile float *) (c_col + m), s0);
                    atomic_store_f32((volatile float *) (c_col + m + STRIDE_M), s1);
                }
                // Tail rows: when at most STRIDE_M / 2 rows are left after the whole rounds, the old schedule ran them as single
                // rows on harts 0..rem-1 while the rest idled, and a lone hart's time follows its instruction count (KB 08
                // lesson 38). Each such row is split into two K halves on harts t and t + rem.
                const int64_t rem = M % (2 * STRIDE_M);
                if (params->tail_scratch && N == 1 && ne12 * ne13 == 1 && K_blocks >= 2 && rem > 0 && rem <= STRIDE_M / 2) {
                    pq2_tail_ksplit(params, src0_ptr2, nb01, b_col, bias_n, c_col, M - rem, rem, K_blocks, windowed, hart_id);
                    m = M;
                }
                for (; m < M; m += STRIDE_M) {
                    const block_pq2_0 * q_row = (const block_pq2_0 *) (src0_ptr2 + m * nb01);

                    float sum = windowed ? pq2_dot_compute_w(q_row, b_col, K_blocks)
                                         : pq2_dot_compute(q_row, b_col, K_blocks);
                    if (bias_n) {
                        sum += bias_n[m];
                    }
                    atomic_store_f32((volatile float *) (c_col + m), sum);
                }
            }
        }
    }

    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));

    // Under the uberkernel the next instruction's consumer-side evict publishes dst; nothing to do here.
    return 0;
}
