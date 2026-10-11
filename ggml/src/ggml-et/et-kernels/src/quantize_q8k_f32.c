//******************************************************************************
// F32 -> Q8_K quantisation, bit-exact with the CPU's quantize_row_q8_K_ref (ggml-quants.c).
//
// Per 256-element block: max = the signed value of the FIRST element with the largest |x| (strict '>', scan order);
// amax == 0 gives d = 0 and zero quants; otherwise iscale = -127/max and d = 1/iscale, both correctly rounded
// (exact integer mantissa division: fdiv traps on this core), q = MIN(127, RNE(RN(iscale * x))) with the product and
// the round-to-nearest-even step as separate instructions (the CPU's base library is built without FMA, so the
// product is rounded before the +12582912 trick), and bsums = sums of 16 consecutive quants.
//
// Work split: a block is 292 bytes and does not end on a cache line, but 16 blocks are exactly 73 lines (4,672 bytes),
// so each hart owns runs of 16 consecutive destination blocks: one writer per 64-byte line (dst 64-byte aligned;
// otherwise thread 0 writes everything). Operands are finite (the CPU's float compares and the integer compares on
// |x| bits agree for every non-NaN value).
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"

#include <stdint.h>

#define Q8K_QK      256
#define Q8K_BLOCK   292  // float d, int8 qs[256], int16 bsums[16]
#define Q8K_RUN     16   // blocks per hart run: 16 * 292 = 73 * 64

struct ggml_et_q8k_params {
    struct ggml_tensor src0;  // F32, ne0 % 256 == 0, unit element stride
    struct ggml_tensor dst;   // Q8_K, contiguous, same shape
};

static inline uint32_t q8k_f2u(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return u;
}

static inline float q8k_u2f(uint32_t u) {
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

// Correctly rounded a / b (round to nearest even) for finite, non-zero a and b. Subnormal operands are normalised;
// a quotient that overflows gives +-inf and one below the normal range gives +-0 (neither occurs for the quantiser's
// operands unless |max| < 3.7e-37).
static float q8k_div_rn(float a, float b) {
    const uint32_t ua = q8k_f2u(a), ub = q8k_f2u(b);
    const uint32_t s  = (ua ^ ub) & 0x80000000u;
    if ((ub & 0x7fffffffu) == 0x7f800000u) {
        return q8k_u2f(s);  // a / inf
    }
    int32_t  ea = (int32_t) ((ua >> 23) & 0xff), eb = (int32_t) ((ub >> 23) & 0xff);
    uint64_t ma = ua & 0x7fffffu, mb = ub & 0x7fffffu;
    if (ea == 0) {
        ea = 1;
        while (!(ma & 0x800000u)) { ma <<= 1; ea--; }
    } else {
        ma |= 0x800000u;
    }
    if (eb == 0) {
        eb = 1;
        while (!(mb & 0x800000u)) { mb <<= 1; eb--; }
    } else {
        mb |= 0x800000u;
    }
    const uint64_t num = ma << 25;  // ma, mb in [2^23, 2^24): quotient in (2^24, 2^26)
    const uint64_t q   = num / mb;
    const uint64_t r   = num % mb;
    int32_t        e   = ea - eb + 127;
    uint32_t       m, guard, rest;
    if (q >= (1ull << 25)) {  // ma >= mb
        m     = (uint32_t) (q >> 2);
        guard = (uint32_t) (q >> 1) & 1u;
        rest  = (uint32_t) (q & 1u) | (r != 0);
    } else {
        e    -= 1;
        m     = (uint32_t) (q >> 1);
        guard = (uint32_t) (q & 1u);
        rest  = (r != 0);
    }
    if (guard && (rest || (m & 1u))) {
        m++;
        if (m == (1u << 24)) {
            m >>= 1;
            e++;
        }
    }
    if (e >= 255) {
        return q8k_u2f(s | 0x7f800000u);
    }
    if (e <= 0) {
        return q8k_u2f(s);
    }
    return q8k_u2f(s | ((uint32_t) e << 23) | (m & 0x7fffffu));
}

// nearest_int(iscale * x): one rounded product, then round half to even through the +12582912 trick.
static inline int32_t q8k_round(float iscale, float x) {
    float       p, v;
    const float magic = 12582912.0f;
    __asm__ volatile("fmul.s %0, %1, %2, rne" : "=f"(p) : "f"(iscale), "f"(x));
    __asm__ volatile("fadd.s %0, %1, %2, rne" : "=f"(v) : "f"(p), "f"(magic));
    const uint32_t i = q8k_f2u(v);
    return (int32_t) (i & 0x007fffffu) - 0x00400000;
}

static void q8k_block(const float * x, uint8_t * blk) {
    uint32_t best = 0, maxbits = 0;
    for (int j = 0; j < Q8K_QK; ++j) {
        const uint32_t u = q8k_f2u(x[j]);
        const uint32_t a = u & 0x7fffffffu;
        if (a > best) {
            best    = a;
            maxbits = u;
        }
    }
    float *   d     = (float *) blk;
    int8_t *  qs    = (int8_t *) (blk + 4);
    int16_t * bsums = (int16_t *) (blk + 4 + Q8K_QK);
    if (best == 0) {
        *d = 0.0f;
        for (int j = 0; j < Q8K_QK; ++j) qs[j] = 0;
        for (int j = 0; j < Q8K_QK / 16; ++j) bsums[j] = 0;
        return;
    }
    const float iscale = q8k_div_rn(-127.0f, q8k_u2f(maxbits));
    for (int j = 0; j < Q8K_QK / 16; ++j) {
        int32_t sum = 0;
        for (int i = 0; i < 16; ++i) {
            int32_t v = q8k_round(iscale, x[j * 16 + i]);
            v         = v < 127 ? v : 127;
            qs[j * 16 + i] = (int8_t) v;
            sum += v;
        }
        bsums[j] = (int16_t) sum;
    }
    *d = q8k_div_rn(1.0f, iscale);
}

int entry_point(struct ggml_et_q8k_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    if (!kernel_env || !params || ((uint64_t) params & 0x7) != 0) {
        return -1;
    }
    const int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    const int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) {
        return 0;
    }

    const struct ggml_tensor * src = &params->src0;
    const struct ggml_tensor * dst = &params->dst;
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_Q8_K || !src->data || !dst->data ||
        src->nb[0] != sizeof(float) || src->ne[0] % Q8K_QK != 0 || dst->ne[0] != src->ne[0] ||
        dst->ne[1] != src->ne[1] || dst->ne[2] != src->ne[2] || dst->ne[3] != src->ne[3] ||
        dst->nb[1] != (size_t) (src->ne[0] / Q8K_QK) * Q8K_BLOCK || dst->nb[2] != dst->nb[1] * (size_t) dst->ne[1] ||
        dst->nb[3] != dst->nb[2] * (size_t) dst->ne[2]) {
        return -1;
    }

    const int64_t bpr   = src->ne[0] / Q8K_QK;  // blocks per row
    const int64_t nrows = src->ne[1] * src->ne[2] * src->ne[3];
    const int64_t nblk  = bpr * nrows;
    const int64_t nruns = (nblk + Q8K_RUN - 1) / Q8K_RUN;
    const int     split = ((uint64_t) dst->data % 64) == 0;

    for (int64_t run = split ? thread_id : 0; run < nruns; run += split ? num_threads : 1) {
        if (!split && thread_id != 0) {
            break;
        }
        const int64_t b1 = (run + 1) * Q8K_RUN < nblk ? (run + 1) * Q8K_RUN : nblk;
        for (int64_t b = run * Q8K_RUN; b < b1; ++b) {
            const int64_t row = b / bpr;
            const int64_t i1  = row % src->ne[1];
            const int64_t i2  = (row / src->ne[1]) % src->ne[2];
            const int64_t i3  = row / (src->ne[1] * src->ne[2]);
            const float * x   = (const float *) ((const char *) src->data + i1 * src->nb[1] + i2 * src->nb[2] +
                                                 i3 * src->nb[3]) + (b % bpr) * Q8K_QK;
            uint8_t *     out = (uint8_t *) dst->data + b * Q8K_BLOCK;  // dst contiguous: block b at b * 292
            q8k_block(x, out);
        }
    }
    return 0;
}
