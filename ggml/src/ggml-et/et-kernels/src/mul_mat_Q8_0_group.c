//******************************************************************************
// Bare Metal grouped Q8_0 MUL_MAT kernel
//
// Runs a group of independent Q8_0 mul_mats that share one activation operand as a
// SINGLE dispatch. In llama decode the graph emits such groups twice per layer: the
// Q/K/V projections and the FFN gate/up projections. They read the same normed hidden
// vector, write disjoint outputs, and have no dependency on one another.
//
// Under the uberkernel the saving is a chip-wide barrier per member, not a host launch:
// the interpreter runs et_barrier_global() before EVERY instruction, so collapsing three
// instructions into one removes two barriers. Across 16 layers that is 48 of ~347.
//
// Each sub-matmul executes exactly the code it would have executed on its own, so the
// result is bit-identical to the separate dispatches.
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"

// The uberkernel build compiles every kernel with -Dentry_point=<kernel>_entry so symbols
// do not collide between translation units. That macro would also rewrite the inner
// #define below, so capture the caller-supplied name and restore it after the include.
// The uberkernel build sets ET_UBERKERNEL and -Dentry_point=<kernel>_entry. We cannot keep
// that macro across the include below (it would rewrite the inner rename), and undef'ing it
// loses the name, so select the exported name explicitly per build.
#ifdef ET_UBERKERNEL
#  define ET_GROUP_ENTRY_NAME mul_mat_Q8_0_group_entry
#else
#  define ET_GROUP_ENTRY_NAME entry_point
#endif
#undef entry_point

// The single-matmul kernel is reused verbatim: including it here renames only its entry
// symbol, so the board-validated dot, its K-split and its hart partitioning are shared
// with the standalone mul_mat_Q8_0.elf rather than duplicated.
#define entry_point mul_mat_q8_0_single
#include "mul_mat_Q8_0.c"
#undef entry_point

#define ET_MUL_MAT_GROUP_MAX 3

struct ggml_et_mul_mat_group_params {
    int32_t n_mats;                                  // 2 (gate/up) or 3 (Q/K/V)
    int32_t reserved;                                // keep the tensor array 8B-aligned
    struct ggml_tensor src0[ET_MUL_MAT_GROUP_MAX];   // weights, one per sub-matmul
    struct ggml_tensor src1;                         // the shared activation
    struct ggml_tensor dst[ET_MUL_MAT_GROUP_MAX];    // outputs, disjoint
    struct ggml_tensor bias[ET_MUL_MAT_GROUP_MAX];   // residual bias; .data == 0 means none
};

int ET_GROUP_ENTRY_NAME(struct ggml_et_mul_mat_group_params * params, void * env) {
    if (params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;  // Invalid pointer
    }

    const int32_t n_mats = params->n_mats;
    if (n_mats < 1 || n_mats > ET_MUL_MAT_GROUP_MAX) {
        return -1;
    }

    // Every hart walks the group in the same order and runs the same number of
    // sub-matmuls. That is what keeps the two harts of a minion in step: the K-split
    // path rendezvouses per row through a counting semaphore, so both harts must issue
    // an identical sequence of post/wait pairs. The K-split decision is a function of
    // the sub-matmul's own shape, which is uniform across harts, so both take the same
    // branch for every member -- including when members differ in row count (Q/K/V is
    // 2048/512/512), because the row loop strides by minion and both harts of a minion
    // share minion_id and therefore trip count.
    for (int32_t g = 0; g < n_mats; g++) {
        struct ggml_et_mm_q8_params sub;
        sub.src0 = params->src0[g];
        sub.src1 = params->src1;
        sub.dst  = params->dst[g];
        sub.bias = params->bias[g];

        const int rc = mul_mat_q8_0_single(&sub, env);
        if (rc != 0) {
            return rc;
        }
    }

    return 0;
}
