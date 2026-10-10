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

#ifndef ET_UBERKERNEL
    evict_region_past_l2(params->src1.data, tensor_bytes(&params->src1));
    if (params->bias.data) {
        evict_region_past_l2(params->bias.data, tensor_bytes(&params->bias));
    }
#endif

    // The windowed gather (fg32b.ps) is bit-identical to fgb.ps only at even chunk addresses: blocks are 34 bytes,
    // so every chunk address is even exactly when the weight base and the row/batch strides are even (pq2_ops.h).
    const int windowed = ((((uintptr_t) params->src0.data) | nb01 | nb02 | nb03) & 1) == 0;

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
            // Two token columns per weight pass on the windowed path (each weight line is loaded once for both).
            for (; windowed && n + 1 < N; n += 2) {
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
