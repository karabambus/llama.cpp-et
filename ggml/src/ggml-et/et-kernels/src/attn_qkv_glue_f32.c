//******************************************************************************
// ATTN_QKV_GLUE F32 kernel (T2 fusion 2 - attention glue after the q/k/v projections)
//
// One launch runs up to three independent "jobs", each a per-(token, head) row
// pipeline over D floats:
//   y = rms_norm(x) * w            (RMS_NORM + MUL, optional: w == NULL skips it)
//   z = rope(y, pos[t])            (ROPE, optional: pos == NULL skips it)
//   dst[t, h] = z                  (F32 output, optional)
//   kv[idx[t]][h*D .. h*D+D) = z   (SET_ROWS into the KV cache, F16 or F32, optional)
// The Qwen3 graph per layer is: Q = norm+rope -> dst; K = norm+rope -> kv;
// V = copy -> kv, i.e. RMS_NORM, MUL, ROPE, RMS_NORM, MUL, ROPE, SET_ROWS,
// SET_ROWS (8 launches) become one.
//
// Numerics are those of the per-op kernels: the norm is rms_norm_mul_f32's
// (x * powf(mean_sq + eps, -0.5)) * w, the rotation and its cos/sin cache come
// from rope_f32.c (included below), the F16 conversion is set_rows_f32's
// fcvt.f16.ps + fsch.ps line store.
//
// Publication contract: every work unit owns whole 64-byte lines of dst and of
// its KV row segment (the host only fuses when D*4 (F32) / D*2 (F16) is a
// multiple of 64 and all bases/strides are line-aligned), so plain stores are
// safe - the same contract as set_rows_f32's cache-aligned path. dst may alias
// the job's src (ggml-alloc in-place chains): each row is read completely into
// a private buffer before it is written, by the same hart.
//******************************************************************************

#include <stdint.h>

#ifdef ET_UBERKERNEL
#    define ET_GLUE_ENTRY_NAME attn_qkv_glue_f32_entry
#else
#    define ET_GLUE_ENTRY_NAME entry_point
#endif
#undef entry_point

// rope_f32.c supplies rope_params_t, compute_rope_cache() and rope_yarn_corr_dims(); its own entry point is
// renamed so the board-validated cache builder is shared instead of copied.
#define entry_point rope_f32_single
#include "rope_f32.c"
#undef entry_point

#define GLUE_MAX_JOBS  3
#define GLUE_MAX_D     256
#define GLUE_SCRATCH_B 4096  // per-hart L2 SCP slice: 2*MAX_ROPE_HALF_DIMS + 2*GLUE_MAX_D floats = 4 KiB

struct ggml_et_qkv_job {
    const void * src;      // F32 [D, H, N]; row (t, h) at src + t*src_nb2 + h*src_nb1
    const void * w;        // F32 [D] rms-norm weight (NULL: no norm)
    const void * pos;      // I32 [N] positions (NULL: no rope)
    void *       dst;      // F32 [D, H, N] output (NULL: not written)
    void *       kv;       // KV cache rows [D*H, kv_rows] (NULL: no set_rows)
    const void * idx;      // I64 [N] destination rows in kv
    uint64_t     src_nb1, src_nb2;
    uint64_t     dst_nb1, dst_nb2;
    uint64_t     kv_nb1, idx_nb0;
    int64_t      kv_rows;  // bounds for idx
    int32_t      D, H, N;
    int32_t      kv_f16;   // 1: kv rows are F16 (convert), 0: F32
    float        eps;
    int32_t      reserved;
};

struct ggml_et_qkv_glue_params {
    struct ggml_et_qkv_job job[GLUE_MAX_JOBS];
    rope_params_t          rope;  // shared by every job with pos != NULL
    int32_t                n_jobs;
    int32_t                reserved;
};

// 8 F32 -> 8 F16 line-store, exactly as set_rows_f32's copy_cache_aligned_f16.
static inline void glue_store_f16_8(uint16_t * dst, const float * src) {
    __asm__ volatile(
        "flw.ps      f2, %[s]         \n\t"
        "fcvt.f16.ps f3, f2           \n\t"
        "fsch.ps     f3, f1(%[d])     \n\t"
        :
        : [s] "m"(*(const float (*)[8]) src), [d] "r"(dst)
        : "f2", "f3", "memory");
}

int ET_GLUE_ENTRY_NAME(struct ggml_et_qkv_glue_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    if (!kernel_env) {
        return -1;
    }
    const int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    const int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) {
        return 0;
    }
    if (params == 0 || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }
    const int32_t n_jobs = params->n_jobs;
    if (n_jobs < 1 || n_jobs > GLUE_MAX_JOBS) {
        return -1;
    }

    int64_t total = 0;
    int     any_rope = 0;
    for (int32_t j = 0; j < n_jobs; ++j) {
        const struct ggml_et_qkv_job * job = &params->job[j];
        if (!job->src || job->D < 8 || job->D > GLUE_MAX_D || (job->D & 7) != 0 || job->H < 1 || job->N < 1) {
            return -1;
        }
        if (job->kv && !job->idx) {
            return -1;
        }
        if (job->pos) {
            any_rope = 1;
        }
        total += (int64_t) job->N * job->H;
    }

    // rope constants (shared): same derivation as rope_f32.c's entry point
    const rope_params_t * rp        = &params->rope;
    float                 corr_dims[2] = { 0.0f, 0.0f };
    float                 theta_scale  = 1.0f;
    int32_t               n_dims       = 0;
    int                   is_neox      = 0;
    if (any_rope) {
        n_dims = rp->n_dims;
        if (n_dims <= 0 || (n_dims & 1) != 0 || n_dims / 2 > MAX_ROPE_HALF_DIMS || (rp->mode & GGML_ROPE_TYPE_MROPE)) {
            return -1;
        }
        rope_yarn_corr_dims(n_dims, rp->n_ctx_orig, rp->freq_base, rp->beta_fast, rp->beta_slow, corr_dims);
        theta_scale = et_powf(rp->freq_base, et_fdiv(-2.0f, (float) n_dims));
        is_neox     = (rp->mode & GGML_ROPE_TYPE_NEOX) != 0;
    }
    const int32_t half_dims = n_dims / 2;

    // hart 0 of every minion first (the two harts of a minion share one pipeline)
    const int64_t half_threads = num_threads / 2;
    const int64_t slot = (thread_id & 1) ? (half_threads + (thread_id >> 1)) : (thread_id >> 1);
    if (slot >= total) {
        return 0;
    }

    // Per-hart scratch lives in a private slice of the shire's L2 SCP (64 harts x 4 KiB = 256 KiB of the 2.5 MiB),
    // not on the stack: cos/sin caches + two D-float row buffers are ~4 KiB, more than the firmware U-mode stack
    // tolerates once both harts of every minion are active (N >= 32 faulted with a smashed return address).
    const uint64_t hart_in_shire = get_hart_id() & 63;
    float * scratch   = (float *) et_shire_l2scp_local(hart_in_shire * GLUE_SCRATCH_B);
    float * cos_cache = scratch;                          // MAX_ROPE_HALF_DIMS floats
    float * sin_cache = scratch + MAX_ROPE_HALF_DIMS;     // MAX_ROPE_HALF_DIMS floats
    float * y         = scratch + 2 * MAX_ROPE_HALF_DIMS; // GLUE_MAX_D floats (32B aligned: offsets are multiples of 1 KiB)
    float * z         = y + GLUE_MAX_D;                   // GLUE_MAX_D floats
    int32_t  last_pos    = 0;
    int      cache_valid = 0;
    uint32_t f16_offs[8] __attribute__((aligned(32))) = { 0, 2, 4, 6, 8, 10, 12, 14 };

    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int64_t u = slot; u < total; u += num_threads) {
        int32_t j = 0;
        int64_t v = u;
        while (v >= (int64_t) params->job[j].N * params->job[j].H) {
            v -= (int64_t) params->job[j].N * params->job[j].H;
            ++j;
        }
        const struct ggml_et_qkv_job * job = &params->job[j];
        const int32_t D = job->D;
        const int64_t t = v / job->H;
        const int64_t h = v % job->H;
        const float * x = (const float *) ((const char *) job->src + t * job->src_nb2 + h * job->src_nb1);

        // 1. rms-norm * w  (or plain copy) -> y
        if (job->w) {
            const float * wgt = (const float *) job->w;
            __asm__ volatile("fbci.pi f10, 0" ::: "f10");
            for (int32_t i0 = 0; i0 < D; i0 += 8) {
                __asm__ volatile(
                    "flw.ps f11, %[x_vec]\n"
                    "fmadd.ps f10, f11, f11, f10\n"
                    :
                    : [x_vec] "m"(*(const float (*)[8]) & x[i0])
                    : "f10", "f11");
            }
            float sum;
            __asm__ __volatile__(
                "fswizz.ps f1, f10, 0xB1 \n\t"
                "fadd.ps   f2, f10, f1, rne \n\t"
                "fswizz.ps f3, f2, 0x4E \n\t"
                "fadd.ps   f4, f2, f3, rne \n\t"
                "fmvz.x.ps t0, f4, 4 \n\t"
                "fbcx.ps   f5, t0 \n\t"
                "fadd.ps   %[vout], f4, f5, rne \n\t"
                : [vout] "=f"(sum)::"t0", "f1", "f2", "f3", "f4", "f5");
            const float inv_d  = et_fdiv(1.0f, (float) D);
            const float scale  = et_powf(sum * inv_d + job->eps, -0.5f);
            if (!(scale > 0.0f)) {
                __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
                return -1;
            }
            uint32_t scale_bits;
            __asm__ volatile("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
            __asm__ volatile("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");
            for (int32_t i0 = 0; i0 < D; i0 += 8) {
                __asm__ volatile(
                    "flw.ps f12, %[x_vec]\n"
                    "flw.ps f15, %[w_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fmul.ps f14, f14, f15\n"
                    "fsw.ps f14, %[result]\n"
                    : [result] "=m"(*(float (*)[8]) & y[i0])
                    : [x_vec] "m"(*(const float (*)[8]) & x[i0]), [w_vec] "m"(*(const float (*)[8]) & wgt[i0])
                    : "f12", "f14", "f15");
            }
        } else {
            for (int32_t i0 = 0; i0 < D; i0 += 8) {
                __asm__ volatile(
                    "flw.ps f12, %[x_vec]\n"
                    "fsw.ps f12, %[result]\n"
                    : [result] "=m"(*(float (*)[8]) & y[i0])
                    : [x_vec] "m"(*(const float (*)[8]) & x[i0])
                    : "f12");
            }
        }

        // 2. rope -> z (else z := y)
        const float * out = y;
        if (job->pos) {
            const int32_t pos = *(const int32_t *) ((const char *) job->pos + t * sizeof(int32_t)) + rp->n_past;
            if (!cache_valid || pos != last_pos) {
                compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos, NULL, rp->freq_scale, corr_dims,
                                   rp->ext_factor, rp->attn_factor);
                last_pos    = pos;
                cache_valid = 1;
            }
            // rope_sincos_block8 saves/restores the mask itself; re-arm the full mask afterwards
            __asm__ volatile("mov.m.x m0, x0, 0xFF");
            for (int32_t d = n_dims; d < D; ++d) {
                z[d] = y[d];
            }
            if (is_neox) {
                for (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
                    __asm__ volatile(
                        "flw.ps f0, %[x0_src]       \n\t"
                        "flw.ps f1, %[x1_src]       \n\t"
                        "flw.ps f2, %[sin_cache]    \n\t"
                        "flw.ps f3, %[cos_cache]    \n\t"
                        "fmul.ps f4, f0, f3         \n\t"
                        "fmul.ps f5, f0, f2         \n\t"
                        "fnmsub.ps f4, f1, f2, f4   \n\t"
                        "fmadd.ps f5, f1, f3, f5    \n\t"
                        "fsw.ps f4, %[x0_dst]       \n\t"
                        "fsw.ps f5, %[x1_dst]       \n\t"
                        : [x0_dst] "=m"(*(float (*)[8]) & z[dim_idx]),
                          [x1_dst] "=m"(*(float (*)[8]) & z[dim_idx + half_dims])
                        : [x0_src] "m"(*(const float (*)[8]) & y[dim_idx]),
                          [x1_src] "m"(*(const float (*)[8]) & y[dim_idx + half_dims]),
                          [sin_cache] "m"(*(const float (*)[8]) & sin_cache[dim_idx]),
                          [cos_cache] "m"(*(const float (*)[8]) & cos_cache[dim_idx])
                        : "f0", "f1", "f2", "f3", "f4", "f5", "memory");
                }
            } else {
                for (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                    const int32_t d  = pair_idx * 2;
                    const float   x0 = y[d];
                    const float   x1 = y[d + 1];
                    z[d]             = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                    z[d + 1]         = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
                }
            }
            out = z;
        }

        // 3. stores
        if (job->dst) {
            float * d = (float *) ((char *) job->dst + t * job->dst_nb2 + h * job->dst_nb1);
            for (int32_t i0 = 0; i0 < D; i0 += 8) {
                __asm__ volatile(
                    "flw.ps f12, %[src]\n"
                    "fsw.ps f12, %[result]\n"
                    : [result] "=m"(*(float (*)[8]) & d[i0])
                    : [src] "m"(*(const float (*)[8]) & out[i0])
                    : "f12", "memory");
            }
        }
        if (job->kv) {
            const int64_t r = *(const int64_t *) ((const char *) job->idx + t * job->idx_nb0);
            if (r < 0 || r >= job->kv_rows) {
                __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
                return -1;
            }
            char * row = (char *) job->kv + r * job->kv_nb1;
            if (job->kv_f16) {
                uint16_t * d = (uint16_t *) row + h * D;
                __asm__ volatile("flw.ps f1, %[o]\n" : : [o] "m"(*(const uint32_t (*)[8]) f16_offs) : "f1");
                for (int32_t i0 = 0; i0 < D; i0 += 8) {
                    glue_store_f16_8(d + i0, &out[i0]);
                }
            } else {
                float * d = (float *) row + h * D;
                for (int32_t i0 = 0; i0 < D; i0 += 8) {
                    __asm__ volatile(
                        "flw.ps f12, %[src]\n"
                        "fsw.ps f12, %[result]\n"
                        : [result] "=m"(*(float (*)[8]) & d[i0])
                        : [src] "m"(*(const float (*)[8]) & out[i0])
                        : "f12", "memory");
                }
            }
        }
    }

    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
    FENCE;
    return 0;
}
