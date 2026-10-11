// ET kernel entry-point parameter structs and tensor helpers.

#ifndef GGML_TENSOR_H
#define GGML_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#include "ggml.h"

struct ggml_et_binary_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

// bias.data == NULL -> unfused MUL_MAT; otherwise dst = mat_mul(...) + bias.
struct ggml_et_mm_q8_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
    struct ggml_tensor bias;
    uint64_t           tail_scratch;  // PQ2_0 GEMV: per-row records for the K-split tail (0 = off), see mul_mat_PQ2_0.c
    uint32_t           tail_gen;      // this launch's generation, unique per launch and never 0
    uint32_t           tail_pad;
};

struct ggml_et_mul_mat_id_params {
    struct ggml_tensor src0;  // [K, M, n_expert]
    struct ggml_tensor src1;  // [K, n_expert_used, batch]
    struct ggml_tensor src2;  // [n_expert_used, batch] (I32 expert indices)
    struct ggml_tensor dst;   // [M, n_expert_used, batch, 1]
    uint64_t shire_mask;      // launch shire mask when the host narrowed it (GGML_ET_MMID_SHIRE_MASK); 0 = use
                              // kernel_env->shire_mask. Needed because the firmware's env mask stays 0xFFFFFFFF for a
                              // narrowed launch (039b diag, 2026-08-27), so work partitioning must not trust it.
};

// Thread partition for a launch whose host narrowed the shire mask (params->shire_mask != 0, contiguous low bits
// e.g. 0x1 / 0xF). platform.h's get_num_threads()/get_relative_thread_id() are STUBS (manual_popcountll == 32,
// manual_ctzll == 0): right for the chip-wide 0xFFFFFFFF every production launch uses, silently 32x too many
// threads for a narrowed one (039b diag, 2026-08-27: a 1-shire launch computed 1/32 of its units). Returns the
// thread count; *thread_id < 0 means this hart is outside the mask. hart_id = get_hart_id() (platform.h is
// included after this header, so the caller passes it).
static inline int ggml_et_mask_threads(uint64_t mask, uint64_t hart_id, int * thread_id) {
    int pop = 0, ctz = 0;
    for (uint64_t x = mask; x; x &= x - 1) { pop++; }
    for (uint64_t x = mask; x && !(x & 1); x >>= 1) { ctz++; }
    const int nthr = pop * 64;
    const int tid  = (int) hart_id - ctz * 64;
    *thread_id = (tid >= 0 && tid < nthr) ? tid : -1;
    return nthr;
}

// ne[i] == 1 axes are skipped: their stride is unobservable.
static inline int ggml_tensor_is_contiguous(const struct ggml_tensor * t, int type_size) {
    int64_t expected = type_size;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (t->ne[i] > 1 && (int64_t) t->nb[i] != expected) {
            return 0;
        }
        expected *= t->ne[i];
    }
    return 1;
}

#endif  // GGML_TENSOR_H
