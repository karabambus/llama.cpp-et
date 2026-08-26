#include "ggml_tensor.h"
#include "platform.h"
#include <stdint.h>

//******************************************************************************
// ARGSORT F32 kernel (backlog 036 — MoE routing on device)
// dst[j, i1, i2, i3] (I32) = index of the j-th smallest (ASC) / largest (DESC)
// element of src0[:, i1, i2, i3]. Order comes from dst->op_params[0]
// (ggml_sort_order). One row per hart; rows are striped over all harts.
//
// Work array: each hart sorts an index array in its private 32 KiB slice of the
// shire's L2 scratchpad (64 harts x 32 KiB = 2 MiB of the 2.5 MiB L2 SCP), so
// rows up to ARGSORT_MAX_NE0 elements need no stack and no DRAM scratch.
// Publishing: dst rows are not cache-line aligned in general (ne0*4 bytes), so
// neighbouring rows written by other harts share lines — publish every index
// with a global atomic swap instead of plain (write-back, non-coherent) stores.
// Ties: broken by the smaller index (deterministic). The CPU reference uses an
// unstable std::sort, so tie order may differ there; test data has no ties.
//******************************************************************************

#define ARGSORT_MAX_NE0     8192
#define ARGSORT_SCRATCH_B   (ARGSORT_MAX_NE0 * (int64_t) sizeof(int32_t))  // 32 KiB per hart

struct ggml_et_argsort_params {
    struct ggml_tensor src0;  // F32 input [ne0, ne1, ne2, ne3]
    struct ggml_tensor dst;   // I32 output, same shape
};

// "a sorts before b": ASC by value, DESC by value; equal values by index.
static inline int __attribute__((always_inline)) sorts_before(const float * v, int32_t a, int32_t b, int desc) {
    const float va = v[a], vb = v[b];
    if (va != vb) {
        return desc ? (va > vb) : (va < vb);
    }
    return a < b;
}

static inline void sift_down(int32_t * idx, const float * v, int64_t root, int64_t n, int desc) {
    for (;;) {
        int64_t child = 2 * root + 1;
        if (child >= n) {
            return;
        }
        // pick the child that sorts later (max-heap w.r.t. sorts_before)
        if (child + 1 < n && sorts_before(v, idx[child], idx[child + 1], desc)) {
            child++;
        }
        if (!sorts_before(v, idx[root], idx[child], desc)) {
            return;
        }
        int32_t t = idx[root]; idx[root] = idx[child]; idx[child] = t;
        root = child;
    }
}

static inline void heapsort_idx(int32_t * idx, const float * v, int64_t n, int desc) {
    for (int64_t i = n / 2 - 1; i >= 0; --i) {
        sift_down(idx, v, i, n, desc);
    }
    for (int64_t end = n - 1; end > 0; --end) {
        int32_t t = idx[0]; idx[0] = idx[end]; idx[end] = t;
        sift_down(idx, v, 0, end, desc);
    }
}

int entry_point(struct ggml_et_argsort_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    if (!kernel_env) {
        return -1;
    }
    int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) {
        return 0;
    }
    if (params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }
    struct ggml_tensor * src0 = &params->src0;
    struct ggml_tensor * dst  = &params->dst;
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        return -1;
    }
    const float * src0_data = (const float *) src0->data;
    int32_t *     dst_data  = (int32_t *) dst->data;
    if (!src0_data || !dst_data) {
        return -1;
    }
    const int64_t ne0 = src0->ne[0];
    if (ne0 > ARGSORT_MAX_NE0 || src0->nb[0] != sizeof(float) || dst->nb[0] != sizeof(int32_t)) {
        return -1;
    }
    const int   desc = (dst->op_params[0] == GGML_SORT_ORDER_DESC);
    const int64_t ne1 = src0->ne[1], ne2 = src0->ne[2], ne3 = src0->ne[3];
    const size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    const uint64_t hart_in_shire = get_hart_id() & 63;
    int32_t * idx = (int32_t *) et_shire_l2scp_local(hart_in_shire * ARGSORT_SCRATCH_B);

    const int64_t total_rows = ne1 * ne2 * ne3;
    for (int64_t ir = thread_id; ir < total_rows; ir += num_threads) {
        const int64_t i3 = ir / (ne2 * ne1);
        const int64_t i2 = (ir - i3 * ne2 * ne1) / ne1;
        const int64_t i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
        const float * v = (const float *) ((const char *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
        int32_t * out   = (int32_t *) ((char *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);
        for (int64_t j = 0; j < ne0; ++j) {
            idx[j] = (int32_t) j;
        }
        heapsort_idx(idx, v, ne0, desc);
        for (int64_t j = 0; j < ne0; ++j) {
            et_global_swap_w((volatile void *) (out + j), (uint32_t) idx[j]);
        }
    }
    FENCE;
    return 0;
}
