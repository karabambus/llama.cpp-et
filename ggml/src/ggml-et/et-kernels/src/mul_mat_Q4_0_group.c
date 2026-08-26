//******************************************************************************
// Grouped Q4_0 MUL_MAT kernel (T2 fusion 3)
//
// Runs a group of independent Q4_0 mul_mats that share one activation operand as a
// SINGLE launch: the q/k/v projections of a decode/verify step (N < 53, the scalar
// kernel's regime). Each member executes exactly the code it would have executed on
// its own (mul_mat_Q4_0.c is included with its entry point renamed), so the result is
// bit-identical to the separate launches; the saving is two host launches + gaps per
// layer. Every hart walks the members in the same order, which keeps the two harts of
// a minion in step through the K-split semaphore rendezvous (the K-split decision is a
// function of the member's own shape, uniform across harts). See mul_mat_Q8_0_group.c.
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"

#ifdef ET_UBERKERNEL
#  define ET_GROUP_ENTRY_NAME mul_mat_Q4_0_group_entry
#else
#  define ET_GROUP_ENTRY_NAME entry_point
#endif
#undef entry_point

#define entry_point mul_mat_q4_0_single
#include "mul_mat_Q4_0.c"
#undef entry_point

#define ET_MUL_MAT_GROUP_MAX 3

struct ggml_et_mul_mat_group_params {
    int32_t n_mats;                                  // 2 or 3
    int32_t reserved;                                // keep the tensor array 8B-aligned
    struct ggml_tensor src0[ET_MUL_MAT_GROUP_MAX];   // weights, one per sub-matmul
    struct ggml_tensor src1;                         // the shared activation
    struct ggml_tensor dst[ET_MUL_MAT_GROUP_MAX];    // outputs, disjoint
    struct ggml_tensor bias[ET_MUL_MAT_GROUP_MAX];   // residual bias; .data == 0 means none
};

int ET_GROUP_ENTRY_NAME(struct ggml_et_mul_mat_group_params * params, void * env) {
    if (params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }
    const int32_t n_mats = params->n_mats;
    if (n_mats < 1 || n_mats > ET_MUL_MAT_GROUP_MAX) {
        return -1;
    }
    for (int32_t g = 0; g < n_mats; g++) {
        struct ggml_et_mm_q8_params sub;
        sub.src0 = params->src0[g];
        sub.src1 = params->src1;
        sub.dst  = params->dst[g];
        sub.bias = params->bias[g];
        const int rc = mul_mat_q4_0_single(&sub, env);
        if (rc != 0) {
            return rc;
        }
    }
    return 0;
}
