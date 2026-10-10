#include "ggml_tensor.h"
#include "platform.h"

#include <stdint.h>

struct ggml_et_ssm_conv_params {
    struct ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    struct ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    struct ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

int entry_point(struct ggml_et_ssm_conv_params * params, void * env) {
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
    struct ggml_tensor * src1 = &params->src1;
    struct ggml_tensor * dst  = &params->dst;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1;
    }

    const float * src0_data = (const float *) src0->data;
    const float * src1_data = (const float *) src1->data;
    float *       dst_data  = (float *) dst->data;

    if (!src0_data || !src1_data || !dst_data) {
        return -1;
    }

    const int64_t nc  = src1->ne[0];
    const int64_t ncs = src0->ne[0];
    const int64_t nr  = src0->ne[1];
    const int64_t n_t = dst->ne[1];
    const int64_t n_s = dst->ne[2];

    if (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != sizeof(float) ||
        src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || src0->nb[1] != (size_t) ncs * sizeof(float) ||
        src1->nb[1] != (size_t) nc * sizeof(float)) {
        return -1;
    }

    // Save and set vector mask to all 8 lanes
    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    // Four taps, whole 16-channel lines: work items are (16 channels = one 64-byte output line, block of SSMC_TB
    // tokens), dealt over all threads, so a 512-token batch keeps every hart busy. An item never covers half a line: two
    // harts writing halves of one line through their own L1s lose each other's half. Each 8-channel half of an item
    // gathers its tap weights once into f14-f17 and slides the window s[ch, t .. t+3] through f10-f13: one 8-channel
    // gather per token instead of 32 scalar loads and stores. Each output adds its taps in the same order (fmadd from
    // +0.0, tap 0 first) as the per-tap form below. f10-f18 and the gather offsets f20/f21 stay live across the asm
    // statements of an item; no compiler float code runs between them.
    if (nc == 4 && nr % 16 == 0) {
#define SSMC_TB 32
        int32_t __attribute__((aligned(32))) offs_s[8], offs_c[8];
        for (int e = 0; e < 8; ++e) {
            offs_s[e] = (int32_t) (e * ncs * (int64_t) sizeof(float));
            offs_c[e] = (int32_t) (e * nc * (int64_t) sizeof(float));
        }
        __asm__ volatile("flw.ps f20, %[os]\n\tflw.ps f21, %[oc]\n"
                         :
                         : [os] "m"(*(const int32_t (*)[8]) offs_s), [oc] "m"(*(const int32_t (*)[8]) offs_c)
                         : "f20", "f21");

        const int64_t n_line = nr / 16;
        const int64_t n_tblk = (n_t + SSMC_TB - 1) / SSMC_TB;
        const int64_t n_item = n_s * n_tblk * n_line;

        for (int64_t item = thread_id; item < n_item; item += num_threads) {
            const int64_t line = item % n_line;
            const int64_t tb   = (item / n_line) % n_tblk;
            const int64_t i3   = item / (n_line * n_tblk);
            const int64_t t0   = tb * SSMC_TB;
            const int64_t t1   = t0 + SSMC_TB < n_t ? t0 + SSMC_TB : n_t;

            for (int64_t grp = 2 * line; grp < 2 * line + 2; ++grp) {
                const float * sb = (const float *) ((const char *) src0_data + i3 * src0->nb[2]) + grp * 8 * ncs;
                const float * cb = src1_data + grp * 8 * nc;
                char *        xb = (char *) dst_data + i3 * dst->nb[2] + grp * 8 * (int64_t) sizeof(float);
                const size_t  xs = dst->nb[1];

                __asm__ volatile(
                    "fgw.ps f14, f21(%[c0])\n\t"
                    "fgw.ps f15, f21(%[c1])\n\t"
                    "fgw.ps f16, f21(%[c2])\n\t"
                    "fgw.ps f17, f21(%[c3])\n\t"
                    "fgw.ps f10, f20(%[s0])\n\t"
                    "fgw.ps f11, f20(%[s1])\n\t"
                    "fgw.ps f12, f20(%[s2])\n\t"
                    :
                    : [c0] "r"(cb), [c1] "r"(cb + 1), [c2] "r"(cb + 2), [c3] "r"(cb + 3), [s0] "r"(sb + t0),
                      [s1] "r"(sb + t0 + 1), [s2] "r"(sb + t0 + 2)
                    : "f10", "f11", "f12", "f14", "f15", "f16", "f17", "memory");

#define SSMC_TOKEN(W0, W1, W2, W3, src, out)                                       \
    __asm__ volatile("fgw.ps   " W3 ", f20(%[sp])\n\t"                            \
                     "fbci.pi  f18, 0\n\t"                                       \
                     "fmadd.ps f18, " W0 ", f14, f18\n\t"                         \
                     "fmadd.ps f18, " W1 ", f15, f18\n\t"                         \
                     "fmadd.ps f18, " W2 ", f16, f18\n\t"                         \
                     "fmadd.ps f18, " W3 ", f17, f18\n\t"                         \
                     "fsw.ps   f18, %[o]\n\t"                                    \
                     : [o] "=m"(*(float (*)[8])(out))                             \
                     : [sp] "r"(src)                                             \
                     : "f10", "f11", "f12", "f13", "f18", "memory")

                int64_t t = t0;
                for (; t + 4 <= t1; t += 4) {
                    SSMC_TOKEN("f10", "f11", "f12", "f13", sb + t + 3, xb + (t + 0) * xs);
                    SSMC_TOKEN("f11", "f12", "f13", "f10", sb + t + 4, xb + (t + 1) * xs);
                    SSMC_TOKEN("f12", "f13", "f10", "f11", sb + t + 5, xb + (t + 2) * xs);
                    SSMC_TOKEN("f13", "f10", "f11", "f12", sb + t + 6, xb + (t + 3) * xs);
                }
                for (; t < t1; ++t) {
                    // Remainder tokens reload the whole window (f10-f12 are not rotated here).
                    __asm__ volatile("fgw.ps f10, f20(%[s0])\n\t"
                                     "fgw.ps f11, f20(%[s1])\n\t"
                                     "fgw.ps f12, f20(%[s2])\n\t"
                                     :
                                     : [s0] "r"(sb + t), [s1] "r"(sb + t + 1), [s2] "r"(sb + t + 2)
                                     : "f10", "f11", "f12", "memory");
                    SSMC_TOKEN("f10", "f11", "f12", "f13", sb + t + 3, xb + t * xs);
                }
            }
#undef SSMC_TOKEN
        }
        __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
        return 0;
#undef SSMC_TB
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    const int64_t chunk    = 16;
    const int64_t n_chunks = (nr + chunk - 1) / chunk;

    for (int64_t i3 = 0; i3 < n_s; ++i3) {
        for (int64_t i2 = 0; i2 < n_t; ++i2) {
            const float * s = (const float *) ((const char *) src0_data + i2 * src0->nb[0] + i3 * src0->nb[2]);
            float *       x = (float *) ((char *) dst_data + i2 * dst->nb[1] + i3 * dst->nb[2]);

            for (int64_t ci = thread_id; ci < n_chunks; ci += num_threads) {
                const int64_t i1_start = ci * chunk;
                const int64_t i1_end   = i1_start + chunk < nr ? i1_start + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_start;
                for (; i1 + 8 <= i1_end; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    float tmp_s[8], tmp_c[8];
                    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    for (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        for (int j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_data[(i1 + j) * nc + i0];
                        }

                        __asm__ volatile(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [out] "=m"(*(float (*)[8]) acc)
                            : [acc] "m"(*(const float (*)[8]) acc), [sv] "m"(*(const float (*)[8]) tmp_s),
                              [cv] "m"(*(const float (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ volatile(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(float (*)[8])(x + i1))
                        : [acc] "m"(*(const float (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                for (; i1 < i1_end; ++i1) {
                    const float * c     = src1_data + i1 * nc;
                    const float * s_row = s + i1 * ncs;
                    float         sumf  = 0.0f;
                    for (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_row[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));

    return 0;
}
