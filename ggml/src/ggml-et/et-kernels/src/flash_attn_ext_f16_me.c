//******************************************************************************
// Flash Attention with TensorFMA16A32 for QK^T
//
// Uses the matrix engine for the QK^T dot products (F16×F16→F32),
// scalar code for online softmax and V accumulation.
//
// Hart 0: tensor engine (Q load, K load from SCP, FMA, softmax, V accum)
// Hart 1: pack K into double-buffered L2 SCP panels, flush for tensor_load
//
// Requirements:
//   - Q: F32 (converted to F16 internally)
//   - K, V: F16
//   - dk must be a multiple of 32 (TensorFMA16A32 K-tile)
//   - dv ≤ 512 (accumulator in shire-local L2 SCP)
//
// Parallelization: each minion independently processes one (qpos, head, batch)
// row, round-robin across all minion hart-0s. Hart 1 assists with K packing.
//******************************************************************************

#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"
#include "tensor.h"

#include <etsoc/common/utils.h>
#include <stdint.h>
#include <string.h>

#define NUM_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32

// QK^T tiles: 16 KV positions at a time, K in chunks of 32 F16
#define TILE_KV 16
#define TILE_K  32

// L1 scratchpad layout: A (Q) in lines 0-15, B (K interleaved) in lines 16-31
#define A_L1_START 0
#define B_L1_START 16

// Max head dimensions
#define FA_DV_MAX 512  // max value head dim (dv)
#define FA_DK_MAX 512  // max key head dim (dk) - some models use hsk > hsv

typedef uint16_t et_fp16_t;

#define ET_NEG_INF_F (-3.402823466e+38f)

// L2 SCP layout per minion:
//   [0..2047]       accumulator (FA_DV_MAX * sizeof(float))
//   [2048..4095]    kpanel buffer 0 (32 × 32 × 2 = 2048 bytes)
//   [4096..6143]    kpanel buffer 1 (2048 bytes)
//   [6144..6207]    stats line - (M_p at +0, S_p at +4), own cache line
// Double-buffering ensures hart 0 finishes buf[N%2] before hart 1
// overwrites it at chunk N+2.
//
// The stats line reserves a cache-line-aligned slot for split-KV softmax
// partials (M_p, S_p). With k_splits=1 the slot is currently unused; step 2
// will populate it and use peer minions' slots during the reduction.
#define SCP_ACC_OFF     0
#define SCP_ACC_STRIDE  (FA_DV_MAX * sizeof(float))       // 2048
#define SCP_KPANEL_SIZE (32 * 32 * sizeof(et_fp16_t))     // 2048
#define SCP_KP0_OFF     SCP_ACC_STRIDE                    // 2048
#define SCP_KP1_OFF     (SCP_KP0_OFF + SCP_KPANEL_SIZE)   // 4096
#define SCP_STATS_OFF   (SCP_KP1_OFF + SCP_KPANEL_SIZE)   // 6144
#define SCP_STATS_SIZE  64                                // own cache line
#define SCP_PER_MINION  (SCP_STATS_OFF + SCP_STATS_SIZE)  // 6208

struct ggml_et_flash_attn_ext_params {
    struct ggml_tensor src0;  // Q (F32)
    struct ggml_tensor src1;  // K (F16)
    struct ggml_tensor src2;  // V (F16)
    struct ggml_tensor mask;  // mask (F16 or F32), zeroed when absent
    struct ggml_tensor dst;   // Output (F32)
    float              scale;
    int32_t            has_mask;
    int32_t            gqa_tiles;  // GQA tile path: 0 off, 1 batches >= GQ_MIN_NQ, 2 >= 2 queries with KV split, 3 every batch
};

static inline float get_mask_val(const struct ggml_tensor * mask, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    const char * base = (const char *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                        (iq3 % mask->ne[3]) * mask->nb[3];

    if (mask->type == GGML_TYPE_F32) {
        return *(const float *) (base + ik1 * mask->nb[0]);
    }
    return fp16_to_fp32(*(const uint16_t *) (base + ik1 * mask->nb[0]));
}

static inline const char * get_mask_row_base(const struct ggml_tensor * mask, int64_t iq1, int64_t iq2, int64_t iq3) {
    return (const char *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
           (iq3 % mask->ne[3]) * mask->nb[3];
}

static inline float get_mask_val_from_base(const struct ggml_tensor * mask, const char * base, int64_t ik1) {
    if (mask->type == GGML_TYPE_F32) {
        return *(const float *) (base + ik1 * mask->nb[0]);
    }
    return fp16_to_fp32(*(const uint16_t *) (base + ik1 * mask->nb[0]));
}

// Pack K rows for TensorLoadTranspose16 (even/odd deinterleave)
static inline void __attribute__((always_inline)) pack_k_for_transpose16(et_fp16_t *  out,
                                                                         const char * k_base,
                                                                         int64_t      kv_start,
                                                                         int64_t      dk_start,
                                                                         int64_t      kv_count,
                                                                         int64_t      nb1_k) {
    unsigned long old_mask;
    __asm__ volatile(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        : [ms] "=&r"(old_mask)
        :
        :);

    for (int j = 0; j < (int) kv_count; ++j) {
        const et_fp16_t * k_row    = (const et_fp16_t *) (k_base + (kv_start + j) * nb1_k) + dk_start;
        et_fp16_t *       even_row = out + (j * 2) * 32;
        et_fp16_t *       odd_row  = out + (j * 2 + 1) * 32;
        __asm__ volatile(
            "flw.ps    f2, 0(%[src0])  \n\t"  // load row[0..15]
            "flw.ps    f3, 0(%[src1])  \n\t"  // load row[16..31]
            "fpackreph.pi f4, f2       \n\t"  // even_lo from src0
            "fpackreph.pi f6, f3       \n\t"  // even_lo from src1 (interleaved)
            "fsrli.pi  f5, f2, 16      \n\t"  // shift src0 for odd
            "fsrli.pi  f7, f3, 16      \n\t"  // shift src1 for odd (interleaved)
            "fpackreph.pi f5, f5       \n\t"  // odd from src0
            "fpackreph.pi f7, f7       \n\t"  // odd from src1
            "mov.m.x   m0, x0, 0x0F   \n\t"
            "fcmovm.ps f4, f4, f6      \n\t"  // merge even halves
            "fcmovm.ps f5, f5, f7      \n\t"  // merge odd halves
            "mov.m.x   m0, x0, 0xFF   \n\t"
            "fsw.ps    f4, 0(%[even])  \n\t"
            "fsw.ps    f5, 0(%[odd])   \n\t"
            :
            : [src0] "r"(k_row), [src1] "r"(k_row + 16), [even] "r"(even_row), [odd] "r"(odd_row)
            : "f2", "f3", "f4", "f5", "f6", "f7", "memory");
    }

    __asm__ volatile("mova.m.x  %[ms]            \n\t" : : [ms] "r"(old_mask));

    for (int j = (int) kv_count; j < TILE_KV; ++j) {
        et_fp16_t * even_row = out + (j * 2) * 32;
        et_fp16_t * odd_row  = out + (j * 2 + 1) * 32;
        for (int l = 0; l < TILE_K / 2; ++l) {
            even_row[l] = 0;
            odd_row[l]  = 0;
        }
    }
}

// Build interleaved B panel for TensorFMA16A32 (weights @ V).
static inline void __attribute__((always_inline)) pack_v_interleaved(et_fp16_t *  out,
                                                                     const char * v_head,
                                                                     int64_t      kv_base,
                                                                     int64_t      dv_start,
                                                                     int64_t      kv_count,
                                                                     int64_t      nb1_v) {
    for (int k = 0; k < TILE_KV; ++k) {
        const int         l   = k >> 1;
        const int         r   = k & 1;
        et_fp16_t * const dst = out + l * 32 + r;
        if (k < (int) kv_count) {
            const et_fp16_t * v_row = (const et_fp16_t *) (v_head + (kv_base + k) * nb1_v) + dv_start;
            for (int n = 0; n < 16; ++n) {
                dst[n * 2] = v_row[n];
            }
        } else {
            for (int n = 0; n < 16; ++n) {
                dst[n * 2] = 0;
            }
        }
    }
}

// Prefetch KV rows for one chunk into L2.
static inline void __attribute__((always_inline)) prefetch_kv_to_l2(const char * head,
                                                                    int64_t      kv_start,
                                                                    int64_t      d_start,
                                                                    int64_t      kv_count,
                                                                    int64_t      nb1) {
    const void * base = (const void *) (head + kv_start * nb1 + d_start * 2);
    l2_prefetch(base, (uint64_t) kv_count, (uint64_t) nb1);
}

static inline void __attribute__((always_inline)) convert_q_row_f32_to_f16(et_fp16_t *   dst,
                                                                           const float * src,
                                                                           int64_t       n) {
    static const int32_t __attribute__((aligned(32))) offsets[8] = { 0, 2, 4, 6, 8, 10, 12, 14 };

    unsigned long old_mask;
    __asm__ volatile(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 0xFF      \n\t"
        "flw.ps    f1, 0(%[offs])    \n\t"
        : [ms] "=&r"(old_mask)
        : [offs] "r"(offsets)
        : "f1");

    for (int64_t d = 0; d < n; d += 8) {
        __asm__ volatile(
            "flw.ps      f2, 0(%[src])    \n\t"
            "fcvt.f16.ps f3, f2           \n\t"
            "fsch.ps     f3, f1(%[dst])   \n\t"
            :
            : [src] "r"(src + d), [dst] "r"(dst + d)
            : "f2", "f3", "memory");
    }

    __asm__ volatile("mova.m.x  %[ms]             \n\t" : : [ms] "r"(old_mask));
}

static inline void __attribute__((always_inline)) zero_acc_vec(float * acc, int64_t dv) {
    const float   zero = 0.0f;
    unsigned long old_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(old_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");
    __asm__ volatile("fbc.ps  f2, 0(%[z])" ::[z] "r"(&zero) : "f2");

    for (int64_t d = 0; d < dv; d += 8) {
        __asm__ volatile("fsw.ps  f2, 0(%[a])     \n\t" ::[a] "r"(acc + d) : "f2", "memory");
    }

    __asm__ volatile("mova.m.x %0" ::"r"(old_mask));
}

static inline void __attribute__((always_inline)) scale_acc_vec(float * acc, int64_t dv, float scale) {
    unsigned long old_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(old_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int64_t d = 0; d < dv; d += 8) {
        __asm__ volatile(
            "fbc.ps    f2, 0(%[s])    \n\t"
            "flw.ps    f3, 0(%[a])    \n\t"
            "fmul.ps   f3, f3, f2     \n\t"
            "fsw.ps    f3, 0(%[a])    \n\t"
            :
            : [s] "r"(&scale), [a] "r"(acc + d)
            : "f2", "f3", "memory");
    }

    __asm__ volatile("mova.m.x %0" ::"r"(old_mask));
}

static inline void __attribute__((always_inline)) normalize_store_vec(float * out,
                                                                      float * acc,
                                                                      int64_t dv,
                                                                      float   inv,
                                                                      int     use_fast_store) {
    unsigned long old_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(old_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int64_t d = 0; d < dv; d += 8) {
        __asm__ volatile(
            "fbc.ps    f2, 0(%[inv])   \n\t"
            "flw.ps    f3, 0(%[a])     \n\t"
            "fmul.ps   f3, f3, f2      \n\t"
            "fsw.ps    f3, 0(%[a])     \n\t"
            :
            : [inv] "r"(&inv), [a] "r"(acc + d)
            : "f2", "f3", "memory");
        if (use_fast_store) {
            __asm__ volatile(
                "flw.ps  f4, 0(%[a])     \n\t"
                "fsw.ps  f4, 0(%[o])     \n\t"
                :
                : [a] "r"(acc + d), [o] "r"(out + d)
                : "f4", "memory");
        } else {
            atomic_store_f32((volatile float *) &out[d + 0], acc[d + 0]);
            atomic_store_f32((volatile float *) &out[d + 1], acc[d + 1]);
            atomic_store_f32((volatile float *) &out[d + 2], acc[d + 2]);
            atomic_store_f32((volatile float *) &out[d + 3], acc[d + 3]);
            atomic_store_f32((volatile float *) &out[d + 4], acc[d + 4]);
            atomic_store_f32((volatile float *) &out[d + 5], acc[d + 5]);
            atomic_store_f32((volatile float *) &out[d + 6], acc[d + 6]);
            atomic_store_f32((volatile float *) &out[d + 7], acc[d + 7]);
        }
    }

    __asm__ volatile("mova.m.x %0" ::"r"(old_mask));
}

// Bytes from a view's first element to the end of its last one (the strided span, not the packed size).
static inline size_t fa_span_bytes(const struct ggml_tensor * t) {
    size_t span = t->nb[0];
    for (int i = 0; i < 4; ++i) {
        if (t->ne[i] > 1) {
            span += (size_t) (t->ne[i] - 1) * t->nb[i];
        }
    }
    return span;
}

// This hart's share of a past-L2 walk split over the 64 harts of a shire: lines h, h + 64, h + 128, ...
static inline void fa_evict_split(const void * addr, size_t bytes, uint64_t h) {
    if (!addr || bytes == 0) {
        return;
    }
    const uint64_t CL     = 64;
    const uint64_t base   = (uint64_t) addr & ~(CL - 1);
    const uint64_t end    = ((uint64_t) addr + bytes + CL - 1) & ~(CL - 1);
    const uint64_t nlines = (end - base) / CL;
    for (uint64_t off = h; off < nlines; off += 16 * 64) {
        uint64_t batch = (nlines - off + 63) / 64;
        if (batch > 16) {
            batch = 16;
        }
        evict_past_l2((const void *) (base + off * CL), batch, 64 * CL);
    }
}

// Evict a byte range from L1D to L2 SCP, splitting into batches of ≤16
// cache lines (the hw limit for evict_to_l2). Use before a barrier when
// another minion in the shire needs to read the region, or after a barrier
// on the reader side to drop stale L1D copies before reading peer data.
static inline void __attribute__((always_inline)) evict_range_to_l2(const void * addr, int64_t bytes) {
    if (bytes <= 0) {
        return;
    }
    int64_t      lines = (bytes + 63) / 64;
    const char * p     = (const char *) addr;
    while (lines > 0) {
        int64_t batch = lines > 16 ? 16 : lines;
        evict_to_l2((const void *) p, (uint64_t) batch, 64);
        p += batch * 64;
        lines -= batch;
    }
}

// Split-KV online merge inner loop:
//
//   for d in [0, dv) step 8:
//       acc[d..d+8] = alpha_own * acc[d..d+8] + alpha_peer * peer_acc[d..d+8]
//
// Runs on the reducer (k_split == 0) after all tensor_fma ops for the row are
// complete, so f0..f31 are dead at entry. We still bracket the loop in inline
// asm with explicit f2/f3/f4/f5 clobbers to lock register usage down — per the
// MM register lifetime rule, never let the compiler mingle FP ops into code
// that sits anywhere near a tensor engine output window.
static inline void __attribute__((always_inline)) merge_rescale_add_asm(float *       acc,
                                                                        const float * peer_acc,
                                                                        int64_t       dv,
                                                                        float         alpha_own,
                                                                        float         alpha_peer) {
    unsigned long old_mask;
    __asm__ volatile(
        "mova.x.m  %[ms]              \n\t"
        "mov.m.x   m0, x0, 0xFF       \n\t"
        "fbc.ps    f4, 0(%[ao])       \n\t"  // broadcast alpha_own
        "fbc.ps    f5, 0(%[ap])       \n\t"  // broadcast alpha_peer
        : [ms] "=&r"(old_mask)
        : [ao] "r"(&alpha_own), [ap] "r"(&alpha_peer)
        : "f4", "f5");

    for (int64_t d = 0; d < dv; d += 8) {
        __asm__ volatile(
            "flw.ps    f2, 0(%[a])      \n\t"  // own
            "flw.ps    f3, 0(%[p])      \n\t"  // peer
            "fmul.ps   f2, f2, f4       \n\t"  // own *= alpha_own
            "fmul.ps   f3, f3, f5       \n\t"  // peer *= alpha_peer
            "fadd.ps   f2, f2, f3       \n\t"
            "fsw.ps    f2, 0(%[a])      \n\t"
            :
            : [a] "r"(acc + d), [p] "r"(peer_acc + d)
            : "f2", "f3", "memory");
    }

    __asm__ volatile("mova.m.x %0" ::"r"(old_mask));
}

//******************************************************************************
// GQA tile path (GGML_ET_FA_GQA: mode 1 batches of >= GQ_MIN_NQ queries, mode 2 >= 2 queries, mode 3 every batch)
//
// A team is (batch, KV head, two consecutive query positions p0, p1). Its KV tiles run on ks minions' hart 0 (ks = 1
// in mode 1 and for large batches; up to 32 of one shire in modes 2-3, merged between two shire barriers). The 16 tile
// columns are the GQA query heads of the two positions: column j < 8 is head ik2*gqa + j at p0, column 8 + j the same
// head at p1 (lanes >= gqa are zero padding). QK^T takes K as the A operand (16 KV rows by plain loads straight from
// the cache) and Q^T as B (a panel built once per team in the L2 scratchpad and streamed through TenB), so the score
// of KV row i, column j lands in f(2i) lane j (p0) or f(2i+1) lane j - 8 (p1) and the softmax runs lane-wise, one lane
// per query row, with no horizontal reductions. P x V takes the 2*gqa probability rows as A and V by INTERLEAVE16 as
// B, as the row path does. With ks = 1 each query row sees the row path's arithmetic in the row path's order: the same
// fp16 products in the same pair order, scale, mask, max, exp2, the same sum tree, rescale and accumulation.
//
// The row path re-reads K and V once per query row and packs K on hart 1; here a K/V tile is read once per 2*gqa
// rows and hart 1 only takes the merge barriers.
//******************************************************************************

#define GQ_LANES  8   // lanes of one position's half of the tile (b_num_col 3: 16 columns = two halves)
#define GQ_MIN_NQ 64  // mode 1: smaller batches keep the row path and its KV split: without a split the GQA path walks
                      // a team's whole KV on one minion, which loses below about 48 queries (r324: N = 16 at 8.7 k positions
                      // 46 ms against the row path's 15.6)

// L1 scratchpad lines
#define GQ_KB0 0   // K chunk buffer 0 (A, load id 0)
#define GQ_KB1 16  // K chunk buffer 1 (A, load id 0)
#define GQ_VB0 0   // V buffer 0 (B, INTERLEAVE16, load id 1), P x V phase only
#define GQ_VB1 8   // V buffer 1
#define GQ_PL1 32  // probability rows (A), up to 16 lines

// L2 scratchpad per minion
#define GQ_QT_OFF     0                                    // Q^T panel: dk/32 chunks x 16 lines
#define GQ_QT_CHUNK   (16 * 64)
#define GQ_ACC_OFF    (GQ_QT_OFF + (FA_DK_MAX / TILE_K) * GQ_QT_CHUNK)
#define GQ_ACC_ROW    (FA_DV_MAX * sizeof(float))          // one accumulator row per query row
#define GQ_P_OFF      (GQ_ACC_OFF + 2 * GQ_LANES * GQ_ACC_ROW)  // probability staging, 16 lines
#define GQ_SC_OFF     (GQ_P_OFF + 16 * 64)                 // spilled QK^T registers f0..f31
#define GQ_ST_OFF     (GQ_SC_OFF + 32 * 32)                // per lane: M, S, alpha, tile max / tile sum
#define GQ_ST_M       0
#define GQ_ST_S       (2 * GQ_LANES)
#define GQ_ST_A       (4 * GQ_LANES)
#define GQ_ST_T       (6 * GQ_LANES)
#define GQ_PER_MINION (GQ_ST_OFF + 8 * GQ_LANES * sizeof(float))
_Static_assert(GQ_PER_MINION % 64 == 0, "per-minion slices start on cache lines");
_Static_assert(MINIONS_PER_SHIRE * GQ_PER_MINION <= 2621440, "L2 scratchpad per shire");

#define GQ_OPEN   0  // mask all +0 (or no mask): scores unchanged
#define GQ_MIXED  1  // per-element mask
#define GQ_MASKED 2  // mask all -inf (or the position does not exist): the half contributes nothing

// All f registers: the tensor engine writes them behind the compiler's back. Placed right after every FMA issue and
// every FMA wait, so no compiler-held float value lives across the window in which the engine writes.
#define GQ_CLOBBER_F                                                                                              \
    __asm__ volatile("" ::: "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", \
                     "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", \
                     "f28", "f29", "f30", "f31", "memory")

static inline int fa_gqa_eligible(const struct ggml_et_flash_attn_ext_params * p) {
    const struct ggml_tensor * q = &p->src0;
    const struct ggml_tensor * k = &p->src1;
    const struct ggml_tensor * v = &p->src2;
    const struct ggml_tensor * m = &p->mask;

    if (!p->gqa_tiles || k->ne[2] <= 0 || q->ne[2] % k->ne[2] != 0) {
        return 0;
    }
    const int64_t gqa = q->ne[2] / k->ne[2];
    // mode 1: >= GQ_MIN_NQ queries; mode 2: >= 2 (single-token decode keeps the row path, which spreads one position
    // over 24 shires where the split GQA path would use 4); mode 3: every batch
    const int64_t min_nq = p->gqa_tiles == 3 ? 1 : p->gqa_tiles == 2 ? 2 : GQ_MIN_NQ;
    if (gqa < 2 || gqa > GQ_LANES || q->ne[1] < min_nq) {
        return 0;
    }
    if (q->ne[0] % TILE_K != 0 || q->ne[0] > FA_DK_MAX || v->ne[0] % 16 != 0 || v->ne[0] > FA_DV_MAX) {
        return 0;
    }
    if (k->ne[1] % TILE_KV != 0) {
        return 0;
    }
    if ((((uint64_t) k->data) | k->nb[1] | k->nb[2] | k->nb[3]) & 63) {
        return 0;
    }
    if ((((uint64_t) v->data) | v->nb[1] | v->nb[2] | v->nb[3]) & 63) {
        return 0;
    }
    if (p->has_mask) {
        if (m->type != GGML_TYPE_F16 || m->nb[0] != 2 || m->ne[2] != 1 || m->ne[0] < k->ne[1] ||
            m->ne[1] < q->ne[1]) {
            return 0;
        }
        if ((((uint64_t) m->data) | m->nb[1] | m->nb[3]) & 7) {
            return 0;
        }
    }
    return 1;
}

static inline int gq_mask_class(const char * mrow, int64_t kv_base) {
    if (!mrow) {
        return GQ_OPEN;
    }
    const volatile uint64_t * w = (const volatile uint64_t *) (mrow + kv_base * 2);
    const uint64_t            a = w[0], b = w[1], c = w[2], d = w[3];
    if ((a | b | c | d) == 0) {
        return GQ_OPEN;
    }
    const uint64_t ninf = 0xFC00FC00FC00FC00ULL;
    if (a == ninf && b == ninf && c == ninf && d == ninf) {
        return GQ_MASKED;
    }
    return GQ_MIXED;
}

// Q^T panel: chunk c, line l, word j = (Q_j[32c + 2l], Q_j[32c + 2l + 1]) as fp16, Q_j = the query row of column j.
static void gq_build_qt(char * qt, const char * q_data, const struct ggml_tensor * q, int64_t dk, int64_t gqa,
                        int64_t ik2, int64_t p0, int p1_valid, int64_t iq3) {
    static const int32_t __attribute__((aligned(32))) offs[8] = { 0, 2, 64, 66, 128, 130, 192, 194 };

    zero_acc_vec((float *) qt, (dk / TILE_K) * 16 * 16);

    unsigned long old_mask;
    __asm__ volatile(
        "mova.x.m  %[ms]          \n\t"
        "mov.m.x   m0, x0, 0xFF   \n\t"
        "flw.ps    f1, 0(%[o])    \n\t"
        : [ms] "=&r"(old_mask)
        : [o] "r"(offs)
        : "f1");
    for (int h = 0; h < 2; ++h) {
        if (h == 1 && !p1_valid) {
            break;
        }
        for (int64_t lane = 0; lane < gqa; ++lane) {
            const float * src = (const float *) (q_data + (p0 + h) * q->nb[1] + (ik2 * gqa + lane) * q->nb[2] +
                                                 iq3 * q->nb[3]);
            char * col = qt + (h * GQ_LANES + lane) * 4;
            for (int64_t d0 = 0; d0 < dk; d0 += 8) {
                char * dst = col + (d0 / TILE_K) * GQ_QT_CHUNK + ((d0 % TILE_K) / 2) * 64;
                __asm__ volatile(
                    "flw.ps      f2, 0(%[s])    \n\t"
                    "fcvt.f16.ps f3, f2         \n\t"
                    "fsch.ps     f3, f1(%[d])   \n\t"
                    :
                    : [s] "r"(src + d0), [d] "r"(dst)
                    : "f2", "f3", "memory");
            }
        }
    }
    __asm__ volatile("mova.m.x  %[ms]          \n\t" : : [ms] "r"(old_mask));

    FENCE;
    for (int64_t c = 0; c < dk / TILE_K; ++c) {
        flush_to_l2(qt + c * GQ_QT_CHUNK, 16, 64);
    }
    WAIT_CACHEOPS;
}

// One position's half of the tile, after QK^T: scale, mask, lane max, rescale, exp2, probability rows, lane sums.
// sc holds the spilled QK^T registers (f(2i + h) at sc + (2i + h) * 32).
static void gq_softmax_half(char * scp, int h, int cls, const char * mrow, int64_t kv_base, int64_t gqa, float scale,
                            int64_t dv) {
    float *       sc   = (float *) (scp + GQ_SC_OFF) + h * GQ_LANES;  // row i at sc + i * 16 floats
    float *       st   = (float *) (scp + GQ_ST_OFF);
    float *       M    = st + GQ_ST_M + h * GQ_LANES;
    float *       S    = st + GQ_ST_S + h * GQ_LANES;
    float *       T    = st + GQ_ST_T + h * GQ_LANES;
    char *        prow = scp + GQ_P_OFF + h * gqa * 64;
    const float   log2e = 1.4426950408889634f;
    unsigned long ms;

    if (cls == GQ_MASKED) {
        // exp2(-inf) everywhere: zero probabilities, M and S unchanged (the row path skips such a tile).
        for (int64_t r = 0; r < gqa; ++r) {
            volatile uint64_t * w = (volatile uint64_t *) (prow + r * 64);
            w[0] = 0;
            w[1] = 0;
            w[2] = 0;
            w[3] = 0;
        }
        return;
    }

    // scale, in place
    __asm__ volatile(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        "fbc.ps    f16, 0(%[ps])    \n\t"
        "flw.ps    f0, 0(%[sc])     \n\t"
        "flw.ps    f1, 64(%[sc])    \n\t"
        "flw.ps    f2, 128(%[sc])   \n\t"
        "flw.ps    f3, 192(%[sc])   \n\t"
        "flw.ps    f4, 256(%[sc])   \n\t"
        "flw.ps    f5, 320(%[sc])   \n\t"
        "flw.ps    f6, 384(%[sc])   \n\t"
        "flw.ps    f7, 448(%[sc])   \n\t"
        "flw.ps    f8, 512(%[sc])   \n\t"
        "flw.ps    f9, 576(%[sc])   \n\t"
        "flw.ps    f10, 640(%[sc])  \n\t"
        "flw.ps    f11, 704(%[sc])  \n\t"
        "flw.ps    f12, 768(%[sc])  \n\t"
        "flw.ps    f13, 832(%[sc])  \n\t"
        "flw.ps    f14, 896(%[sc])  \n\t"
        "flw.ps    f15, 960(%[sc])  \n\t"
        "fmul.ps   f0, f0, f16      \n\t"
        "fmul.ps   f1, f1, f16      \n\t"
        "fmul.ps   f2, f2, f16      \n\t"
        "fmul.ps   f3, f3, f16      \n\t"
        "fmul.ps   f4, f4, f16      \n\t"
        "fmul.ps   f5, f5, f16      \n\t"
        "fmul.ps   f6, f6, f16      \n\t"
        "fmul.ps   f7, f7, f16      \n\t"
        "fmul.ps   f8, f8, f16      \n\t"
        "fmul.ps   f9, f9, f16      \n\t"
        "fmul.ps   f10, f10, f16    \n\t"
        "fmul.ps   f11, f11, f16    \n\t"
        "fmul.ps   f12, f12, f16    \n\t"
        "fmul.ps   f13, f13, f16    \n\t"
        "fmul.ps   f14, f14, f16    \n\t"
        "fmul.ps   f15, f15, f16    \n\t"
        "fsw.ps    f0, 0(%[sc])     \n\t"
        "fsw.ps    f1, 64(%[sc])    \n\t"
        "fsw.ps    f2, 128(%[sc])   \n\t"
        "fsw.ps    f3, 192(%[sc])   \n\t"
        "fsw.ps    f4, 256(%[sc])   \n\t"
        "fsw.ps    f5, 320(%[sc])   \n\t"
        "fsw.ps    f6, 384(%[sc])   \n\t"
        "fsw.ps    f7, 448(%[sc])   \n\t"
        "fsw.ps    f8, 512(%[sc])   \n\t"
        "fsw.ps    f9, 576(%[sc])   \n\t"
        "fsw.ps    f10, 640(%[sc])  \n\t"
        "fsw.ps    f11, 704(%[sc])  \n\t"
        "fsw.ps    f12, 768(%[sc])  \n\t"
        "fsw.ps    f13, 832(%[sc])  \n\t"
        "fsw.ps    f14, 896(%[sc])  \n\t"
        "fsw.ps    f15, 960(%[sc])  \n\t"
        "mova.m.x  %[ms]            \n\t"
        : [ms] "=&r"(ms)
        : [sc] "r"(sc), [ps] "r"(&scale)
        : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16",
          "memory");

    // mask, the row path's rule: a value equal to its -FLT_MAX sentinel or NaN sets the score to the sentinel,
    // anything else is added (an all-zero tile adds +0 everywhere, which changes no exp2 result: skipped)
    if (cls == GQ_MIXED) {
        for (int64_t i = 0; i < TILE_KV; ++i) {
            float mv = fp16_to_fp32(*(const uint16_t *) (mrow + (kv_base + i) * 2));
            float * row = sc + i * 16;
            if (mv == ET_NEG_INF_F || mv != mv) {
                const float sent = ET_NEG_INF_F;
                __asm__ volatile(
                    "mova.x.m  %[ms]          \n\t"
                    "mov.m.x   m0, x0, 0xFF   \n\t"
                    "fbc.ps    f0, 0(%[v])    \n\t"
                    "fsw.ps    f0, 0(%[r])    \n\t"
                    "mova.m.x  %[ms]          \n\t"
                    : [ms] "=&r"(ms)
                    : [v] "r"(&sent), [r] "r"(row)
                    : "f0", "memory");
            } else {
                __asm__ volatile(
                    "mova.x.m  %[ms]          \n\t"
                    "mov.m.x   m0, x0, 0xFF   \n\t"
                    "fbc.ps    f1, 0(%[v])    \n\t"
                    "flw.ps    f0, 0(%[r])    \n\t"
                    "fadd.ps   f0, f0, f1     \n\t"
                    "fsw.ps    f0, 0(%[r])    \n\t"
                    "mova.m.x  %[ms]          \n\t"
                    : [ms] "=&r"(ms)
                    : [v] "r"(&mv), [r] "r"(row)
                    : "f0", "f1", "memory");
            }
        }
    }

    // lane max over the 16 KV rows
    __asm__ volatile(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        "flw.ps    f0, 0(%[sc])     \n\t"
        "flw.ps    f1, 64(%[sc])    \n\t"
        "flw.ps    f2, 128(%[sc])   \n\t"
        "flw.ps    f3, 192(%[sc])   \n\t"
        "flw.ps    f4, 256(%[sc])   \n\t"
        "flw.ps    f5, 320(%[sc])   \n\t"
        "flw.ps    f6, 384(%[sc])   \n\t"
        "flw.ps    f7, 448(%[sc])   \n\t"
        "flw.ps    f8, 512(%[sc])   \n\t"
        "flw.ps    f9, 576(%[sc])   \n\t"
        "flw.ps    f10, 640(%[sc])  \n\t"
        "flw.ps    f11, 704(%[sc])  \n\t"
        "flw.ps    f12, 768(%[sc])  \n\t"
        "flw.ps    f13, 832(%[sc])  \n\t"
        "flw.ps    f14, 896(%[sc])  \n\t"
        "flw.ps    f15, 960(%[sc])  \n\t"
        "fmax.ps   f0, f0, f1       \n\t"
        "fmax.ps   f2, f2, f3       \n\t"
        "fmax.ps   f4, f4, f5       \n\t"
        "fmax.ps   f6, f6, f7       \n\t"
        "fmax.ps   f8, f8, f9       \n\t"
        "fmax.ps   f10, f10, f11    \n\t"
        "fmax.ps   f12, f12, f13    \n\t"
        "fmax.ps   f14, f14, f15    \n\t"
        "fmax.ps   f0, f0, f2       \n\t"
        "fmax.ps   f4, f4, f6       \n\t"
        "fmax.ps   f8, f8, f10      \n\t"
        "fmax.ps   f12, f12, f14    \n\t"
        "fmax.ps   f0, f0, f4       \n\t"
        "fmax.ps   f8, f8, f12      \n\t"
        "fmax.ps   f0, f0, f8       \n\t"
        "fsw.ps    f0, 0(%[t])      \n\t"
        "mova.m.x  %[ms]            \n\t"
        : [ms] "=&r"(ms)
        : [sc] "r"(sc), [t] "r"(T)
        : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15",
          "memory");

    // a half with no score above the -FLT_MAX sentinel is skipped, as the row path skips such a tile
    {
        int64_t live = 0;
        for (int64_t lane = 0; lane < gqa; ++lane) {
            live += T[lane] > ET_NEG_INF_F;
        }
        if (live == 0) {
            gq_softmax_half(scp, h, GQ_MASKED, mrow, kv_base, gqa, scale, dv);
            return;
        }
    }

    // a lane whose tile max exceeds its running max rescales its accumulator row and sum (the row path's rule)
    float * acc_h = (float *) (scp + GQ_ACC_OFF) + h * gqa * (GQ_ACC_ROW / sizeof(float));
    for (int64_t lane = 0; lane < gqa; ++lane) {
        const float tm = T[lane];
        if (tm > M[lane]) {
            const float rescale = et_exp2f((M[lane] - tm) * log2e);
            scale_acc_vec(acc_h + lane * (GQ_ACC_ROW / sizeof(float)), dv, rescale);
            S[lane] *= rescale;
            M[lane] = tm;
        }
    }

    // p = exp2((s - M) * log2e) per lane; fp16 copies scattered into the probability rows (lane j -> row h*gqa + j,
    // halfword i); lane sums in the row path's tree order ((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7)), a_k = p_k + p_{k+8}
    static const int32_t __attribute__((aligned(32))) prow_offs[8] = { 0, 64, 128, 192, 256, 320, 384, 448 };
    const uint64_t lane_mask = (1ULL << gqa) - 1;
    __asm__ volatile(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        "flw.ps    f16, 0(%[pm])    \n\t"
        "fbc.ps    f17, 0(%[pl])    \n\t"
        "flw.ps    f18, 0(%[po])    \n\t"
        "flw.ps    f0, 0(%[sc])   \n\t"
        "flw.ps    f1, 64(%[sc])   \n\t"
        "flw.ps    f2, 128(%[sc])   \n\t"
        "flw.ps    f3, 192(%[sc])   \n\t"
        "flw.ps    f4, 256(%[sc])   \n\t"
        "flw.ps    f5, 320(%[sc])   \n\t"
        "flw.ps    f6, 384(%[sc])   \n\t"
        "flw.ps    f7, 448(%[sc])   \n\t"
        "flw.ps    f8, 512(%[sc])   \n\t"
        "flw.ps    f9, 576(%[sc])   \n\t"
        "flw.ps    f10, 640(%[sc])   \n\t"
        "flw.ps    f11, 704(%[sc])   \n\t"
        "flw.ps    f12, 768(%[sc])   \n\t"
        "flw.ps    f13, 832(%[sc])   \n\t"
        "flw.ps    f14, 896(%[sc])   \n\t"
        "flw.ps    f15, 960(%[sc])   \n\t"
        "fsub.ps   f0, f0, f16   \n\t"
        "fsub.ps   f1, f1, f16   \n\t"
        "fsub.ps   f2, f2, f16   \n\t"
        "fsub.ps   f3, f3, f16   \n\t"
        "fsub.ps   f4, f4, f16   \n\t"
        "fsub.ps   f5, f5, f16   \n\t"
        "fsub.ps   f6, f6, f16   \n\t"
        "fsub.ps   f7, f7, f16   \n\t"
        "fsub.ps   f8, f8, f16   \n\t"
        "fsub.ps   f9, f9, f16   \n\t"
        "fsub.ps   f10, f10, f16   \n\t"
        "fsub.ps   f11, f11, f16   \n\t"
        "fsub.ps   f12, f12, f16   \n\t"
        "fsub.ps   f13, f13, f16   \n\t"
        "fsub.ps   f14, f14, f16   \n\t"
        "fsub.ps   f15, f15, f16   \n\t"
        "fmul.ps   f0, f0, f17   \n\t"
        "fmul.ps   f1, f1, f17   \n\t"
        "fmul.ps   f2, f2, f17   \n\t"
        "fmul.ps   f3, f3, f17   \n\t"
        "fmul.ps   f4, f4, f17   \n\t"
        "fmul.ps   f5, f5, f17   \n\t"
        "fmul.ps   f6, f6, f17   \n\t"
        "fmul.ps   f7, f7, f17   \n\t"
        "fmul.ps   f8, f8, f17   \n\t"
        "fmul.ps   f9, f9, f17   \n\t"
        "fmul.ps   f10, f10, f17   \n\t"
        "fmul.ps   f11, f11, f17   \n\t"
        "fmul.ps   f12, f12, f17   \n\t"
        "fmul.ps   f13, f13, f17   \n\t"
        "fmul.ps   f14, f14, f17   \n\t"
        "fmul.ps   f15, f15, f17   \n\t"
        "fexp.ps   f0, f0        \n\t"
        "fexp.ps   f1, f1        \n\t"
        "fexp.ps   f2, f2        \n\t"
        "fexp.ps   f3, f3        \n\t"
        "fexp.ps   f4, f4        \n\t"
        "fexp.ps   f5, f5        \n\t"
        "fexp.ps   f6, f6        \n\t"
        "fexp.ps   f7, f7        \n\t"
        "fexp.ps   f8, f8        \n\t"
        "fexp.ps   f9, f9        \n\t"
        "fexp.ps   f10, f10        \n\t"
        "fexp.ps   f11, f11        \n\t"
        "fexp.ps   f12, f12        \n\t"
        "fexp.ps   f13, f13        \n\t"
        "fexp.ps   f14, f14        \n\t"
        "fexp.ps   f15, f15        \n\t"
        "mov.m.x   m0, %[lm], 0     \n\t"
        "fcvt.f16.ps f19, f0      \n\t"
        "addi      t0, %[pr], 0    \n\t"
        "fsch.ps   f19, f18(t0)    \n\t"
        "fcvt.f16.ps f19, f1      \n\t"
        "addi      t0, %[pr], 2    \n\t"
        "fsch.ps   f19, f18(t0)    \n\t"
        "fcvt.f16.ps f19, f2      \n\t"
        "addi      t0, %[pr], 4    \n\t"
        "fsch.ps   f19, f18(t0)    \n\t"
        "fcvt.f16.ps f19, f3      \n\t"
        "addi      t0, %[pr], 6    \n\t"
        "fsch.ps   f19, f18(t0)    \n\t"
        "fcvt.f16.ps f19, f4      \n\t"
        "addi      t0, %[pr], 8    \n\t"
        "fsch.ps   f19, f18(t0)    \n\t"
        "fcvt.f16.ps f19, f5      \n\t"
        "addi      t0, %[pr], 10    \n\t"
        "fsch.ps   f19, f18(t0)    \n\t"
        "fcvt.f16.ps f19, f6      \n\t"
        "addi      t0, %[pr], 12    \n\t"
        "fsch.ps   f19, f18(t0)    \n\t"
        "fcvt.f16.ps f19, f7      \n\t"
        "addi      t0, %[pr], 14    \n\t"
        "fsch.ps   f19, f18(t0)    \n\t"
        "fcvt.f16.ps f19, f8      \n\t"
        "addi      t0, %[pr], 16    \n\t"
        "fsch.ps   f19, f18(t0)    \n\t"
        "fcvt.f16.ps f19, f9      \n\t"
        "addi      t0, %[pr], 18    \n\t"
        "fsch.ps   f19, f18(t0)    \n\t"
        "fcvt.f16.ps f19, f10      \n\t"
        "addi      t0, %[pr], 20    \n\t"
        "fsch.ps   f19, f18(t0)    \n\t"
        "fcvt.f16.ps f19, f11      \n\t"
        "addi      t0, %[pr], 22    \n\t"
        "fsch.ps   f19, f18(t0)    \n\t"
        "fcvt.f16.ps f19, f12      \n\t"
        "addi      t0, %[pr], 24    \n\t"
        "fsch.ps   f19, f18(t0)    \n\t"
        "fcvt.f16.ps f19, f13      \n\t"
        "addi      t0, %[pr], 26    \n\t"
        "fsch.ps   f19, f18(t0)    \n\t"
        "fcvt.f16.ps f19, f14      \n\t"
        "addi      t0, %[pr], 28    \n\t"
        "fsch.ps   f19, f18(t0)    \n\t"
        "fcvt.f16.ps f19, f15      \n\t"
        "addi      t0, %[pr], 30    \n\t"
        "fsch.ps   f19, f18(t0)    \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        "fadd.ps   f0, f0, f8, rne \n\t"
        "fadd.ps   f1, f1, f9, rne \n\t"
        "fadd.ps   f2, f2, f10, rne \n\t"
        "fadd.ps   f3, f3, f11, rne \n\t"
        "fadd.ps   f4, f4, f12, rne \n\t"
        "fadd.ps   f5, f5, f13, rne \n\t"
        "fadd.ps   f6, f6, f14, rne \n\t"
        "fadd.ps   f7, f7, f15, rne \n\t"
        "fadd.ps   f0, f0, f1, rne \n\t"
        "fadd.ps   f2, f2, f3, rne \n\t"
        "fadd.ps   f4, f4, f5, rne \n\t"
        "fadd.ps   f6, f6, f7, rne \n\t"
        "fadd.ps   f0, f0, f2, rne  \n\t"
        "fadd.ps   f4, f4, f6, rne  \n\t"
        "fadd.ps   f0, f0, f4, rne  \n\t"
        "fsw.ps    f0, 0(%[t])      \n\t"
        "mova.m.x  %[ms]            \n\t"
        : [ms] "=&r"(ms)
        : [sc] "r"(sc), [pm] "r"(M), [pl] "r"(&log2e), [po] "r"(prow_offs), [lm] "r"(lane_mask), [pr] "r"(prow),
          [t] "r"(T)
        : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16",
          "f17", "f18", "f19", "t0", "memory");

    for (int64_t lane = 0; lane < gqa; ++lane) {
        S[lane] += T[lane];
    }
}

// After a P x V FMA: acc[r][0..15] = O + acc for the rows one P x V FMA wrote (O in f(2r), f(2r + 1), r < rows <= 12),
// in the row path's operand order. The accumulator lives in the L2 scratchpad: each group of four rows issues its eight
// loads into the free registers f24..f31 before the adds use them. m0 must be all lanes.
static inline void __attribute__((always_inline)) gq_acc_add(float * acc_row0, int64_t rows) {
    const int64_t RS = GQ_ACC_ROW / sizeof(float);
    if (rows >= 4) {
        __asm__ volatile(
            "flw.ps    f24, 0(%[a0])        \n\t"
            "flw.ps    f25, 32(%[a0])       \n\t"
            "flw.ps    f26, 0(%[a1])        \n\t"
            "flw.ps    f27, 32(%[a1])       \n\t"
            "flw.ps    f28, 0(%[a2])        \n\t"
            "flw.ps    f29, 32(%[a2])       \n\t"
            "flw.ps    f30, 0(%[a3])        \n\t"
            "flw.ps    f31, 32(%[a3])       \n\t"
            "fadd.ps   f0, f0, f24         \n\t"
            "fadd.ps   f1, f1, f25         \n\t"
            "fadd.ps   f2, f2, f26         \n\t"
            "fadd.ps   f3, f3, f27         \n\t"
            "fadd.ps   f4, f4, f28         \n\t"
            "fadd.ps   f5, f5, f29         \n\t"
            "fadd.ps   f6, f6, f30         \n\t"
            "fadd.ps   f7, f7, f31         \n\t"
            "fsw.ps    f0, 0(%[a0])         \n\t"
            "fsw.ps    f1, 32(%[a0])        \n\t"
            "fsw.ps    f2, 0(%[a1])         \n\t"
            "fsw.ps    f3, 32(%[a1])        \n\t"
            "fsw.ps    f4, 0(%[a2])         \n\t"
            "fsw.ps    f5, 32(%[a2])        \n\t"
            "fsw.ps    f6, 0(%[a3])         \n\t"
            "fsw.ps    f7, 32(%[a3])        \n\t"
            :
            : [a0] "r"(acc_row0 + 0 * RS), [a1] "r"(acc_row0 + 1 * RS), [a2] "r"(acc_row0 + 2 * RS), [a3] "r"(acc_row0 + 3 * RS)
            : "f0", "f1", "f24", "f25", "f2", "f3", "f26", "f27", "f4", "f5", "f28", "f29", "f6", "f7", "f30", "f31", "memory");
    } else if (rows == 3) {
        __asm__ volatile(
            "flw.ps    f24, 0(%[a0])        \n\t"
            "flw.ps    f25, 32(%[a0])       \n\t"
            "flw.ps    f26, 0(%[a1])        \n\t"
            "flw.ps    f27, 32(%[a1])       \n\t"
            "flw.ps    f28, 0(%[a2])        \n\t"
            "flw.ps    f29, 32(%[a2])       \n\t"
            "fadd.ps   f0, f0, f24         \n\t"
            "fadd.ps   f1, f1, f25         \n\t"
            "fadd.ps   f2, f2, f26         \n\t"
            "fadd.ps   f3, f3, f27         \n\t"
            "fadd.ps   f4, f4, f28         \n\t"
            "fadd.ps   f5, f5, f29         \n\t"
            "fsw.ps    f0, 0(%[a0])         \n\t"
            "fsw.ps    f1, 32(%[a0])        \n\t"
            "fsw.ps    f2, 0(%[a1])         \n\t"
            "fsw.ps    f3, 32(%[a1])        \n\t"
            "fsw.ps    f4, 0(%[a2])         \n\t"
            "fsw.ps    f5, 32(%[a2])        \n\t"
            :
            : [a0] "r"(acc_row0 + 0 * RS), [a1] "r"(acc_row0 + 1 * RS), [a2] "r"(acc_row0 + 2 * RS)
            : "f0", "f1", "f24", "f25", "f2", "f3", "f26", "f27", "f4", "f5", "f28", "f29", "memory");
    } else if (rows == 2) {
        __asm__ volatile(
            "flw.ps    f24, 0(%[a0])        \n\t"
            "flw.ps    f25, 32(%[a0])       \n\t"
            "flw.ps    f26, 0(%[a1])        \n\t"
            "flw.ps    f27, 32(%[a1])       \n\t"
            "fadd.ps   f0, f0, f24         \n\t"
            "fadd.ps   f1, f1, f25         \n\t"
            "fadd.ps   f2, f2, f26         \n\t"
            "fadd.ps   f3, f3, f27         \n\t"
            "fsw.ps    f0, 0(%[a0])         \n\t"
            "fsw.ps    f1, 32(%[a0])        \n\t"
            "fsw.ps    f2, 0(%[a1])         \n\t"
            "fsw.ps    f3, 32(%[a1])        \n\t"
            :
            : [a0] "r"(acc_row0 + 0 * RS), [a1] "r"(acc_row0 + 1 * RS)
            : "f0", "f1", "f24", "f25", "f2", "f3", "f26", "f27", "memory");
    } else if (rows == 1) {
        __asm__ volatile(
            "flw.ps    f24, 0(%[a0])        \n\t"
            "flw.ps    f25, 32(%[a0])       \n\t"
            "fadd.ps   f0, f0, f24         \n\t"
            "fadd.ps   f1, f1, f25         \n\t"
            "fsw.ps    f0, 0(%[a0])         \n\t"
            "fsw.ps    f1, 32(%[a0])        \n\t"
            :
            : [a0] "r"(acc_row0 + 0 * RS)
            : "f0", "f1", "f24", "f25", "memory");
    }
    if (rows >= 8) {
        __asm__ volatile(
            "flw.ps    f24, 0(%[a0])        \n\t"
            "flw.ps    f25, 32(%[a0])       \n\t"
            "flw.ps    f26, 0(%[a1])        \n\t"
            "flw.ps    f27, 32(%[a1])       \n\t"
            "flw.ps    f28, 0(%[a2])        \n\t"
            "flw.ps    f29, 32(%[a2])       \n\t"
            "flw.ps    f30, 0(%[a3])        \n\t"
            "flw.ps    f31, 32(%[a3])       \n\t"
            "fadd.ps   f8, f8, f24         \n\t"
            "fadd.ps   f9, f9, f25         \n\t"
            "fadd.ps   f10, f10, f26         \n\t"
            "fadd.ps   f11, f11, f27         \n\t"
            "fadd.ps   f12, f12, f28         \n\t"
            "fadd.ps   f13, f13, f29         \n\t"
            "fadd.ps   f14, f14, f30         \n\t"
            "fadd.ps   f15, f15, f31         \n\t"
            "fsw.ps    f8, 0(%[a0])         \n\t"
            "fsw.ps    f9, 32(%[a0])        \n\t"
            "fsw.ps    f10, 0(%[a1])         \n\t"
            "fsw.ps    f11, 32(%[a1])        \n\t"
            "fsw.ps    f12, 0(%[a2])         \n\t"
            "fsw.ps    f13, 32(%[a2])        \n\t"
            "fsw.ps    f14, 0(%[a3])         \n\t"
            "fsw.ps    f15, 32(%[a3])        \n\t"
            :
            : [a0] "r"(acc_row0 + 4 * RS), [a1] "r"(acc_row0 + 5 * RS), [a2] "r"(acc_row0 + 6 * RS), [a3] "r"(acc_row0 + 7 * RS)
            : "f8", "f9", "f24", "f25", "f10", "f11", "f26", "f27", "f12", "f13", "f28", "f29", "f14", "f15", "f30", "f31", "memory");
    } else if (rows == 7) {
        __asm__ volatile(
            "flw.ps    f24, 0(%[a0])        \n\t"
            "flw.ps    f25, 32(%[a0])       \n\t"
            "flw.ps    f26, 0(%[a1])        \n\t"
            "flw.ps    f27, 32(%[a1])       \n\t"
            "flw.ps    f28, 0(%[a2])        \n\t"
            "flw.ps    f29, 32(%[a2])       \n\t"
            "fadd.ps   f8, f8, f24         \n\t"
            "fadd.ps   f9, f9, f25         \n\t"
            "fadd.ps   f10, f10, f26         \n\t"
            "fadd.ps   f11, f11, f27         \n\t"
            "fadd.ps   f12, f12, f28         \n\t"
            "fadd.ps   f13, f13, f29         \n\t"
            "fsw.ps    f8, 0(%[a0])         \n\t"
            "fsw.ps    f9, 32(%[a0])        \n\t"
            "fsw.ps    f10, 0(%[a1])         \n\t"
            "fsw.ps    f11, 32(%[a1])        \n\t"
            "fsw.ps    f12, 0(%[a2])         \n\t"
            "fsw.ps    f13, 32(%[a2])        \n\t"
            :
            : [a0] "r"(acc_row0 + 4 * RS), [a1] "r"(acc_row0 + 5 * RS), [a2] "r"(acc_row0 + 6 * RS)
            : "f8", "f9", "f24", "f25", "f10", "f11", "f26", "f27", "f12", "f13", "f28", "f29", "memory");
    } else if (rows == 6) {
        __asm__ volatile(
            "flw.ps    f24, 0(%[a0])        \n\t"
            "flw.ps    f25, 32(%[a0])       \n\t"
            "flw.ps    f26, 0(%[a1])        \n\t"
            "flw.ps    f27, 32(%[a1])       \n\t"
            "fadd.ps   f8, f8, f24         \n\t"
            "fadd.ps   f9, f9, f25         \n\t"
            "fadd.ps   f10, f10, f26         \n\t"
            "fadd.ps   f11, f11, f27         \n\t"
            "fsw.ps    f8, 0(%[a0])         \n\t"
            "fsw.ps    f9, 32(%[a0])        \n\t"
            "fsw.ps    f10, 0(%[a1])         \n\t"
            "fsw.ps    f11, 32(%[a1])        \n\t"
            :
            : [a0] "r"(acc_row0 + 4 * RS), [a1] "r"(acc_row0 + 5 * RS)
            : "f8", "f9", "f24", "f25", "f10", "f11", "f26", "f27", "memory");
    } else if (rows == 5) {
        __asm__ volatile(
            "flw.ps    f24, 0(%[a0])        \n\t"
            "flw.ps    f25, 32(%[a0])       \n\t"
            "fadd.ps   f8, f8, f24         \n\t"
            "fadd.ps   f9, f9, f25         \n\t"
            "fsw.ps    f8, 0(%[a0])         \n\t"
            "fsw.ps    f9, 32(%[a0])        \n\t"
            :
            : [a0] "r"(acc_row0 + 4 * RS)
            : "f8", "f9", "f24", "f25", "memory");
    }
    if (rows >= 12) {
        __asm__ volatile(
            "flw.ps    f24, 0(%[a0])        \n\t"
            "flw.ps    f25, 32(%[a0])       \n\t"
            "flw.ps    f26, 0(%[a1])        \n\t"
            "flw.ps    f27, 32(%[a1])       \n\t"
            "flw.ps    f28, 0(%[a2])        \n\t"
            "flw.ps    f29, 32(%[a2])       \n\t"
            "flw.ps    f30, 0(%[a3])        \n\t"
            "flw.ps    f31, 32(%[a3])       \n\t"
            "fadd.ps   f16, f16, f24         \n\t"
            "fadd.ps   f17, f17, f25         \n\t"
            "fadd.ps   f18, f18, f26         \n\t"
            "fadd.ps   f19, f19, f27         \n\t"
            "fadd.ps   f20, f20, f28         \n\t"
            "fadd.ps   f21, f21, f29         \n\t"
            "fadd.ps   f22, f22, f30         \n\t"
            "fadd.ps   f23, f23, f31         \n\t"
            "fsw.ps    f16, 0(%[a0])         \n\t"
            "fsw.ps    f17, 32(%[a0])        \n\t"
            "fsw.ps    f18, 0(%[a1])         \n\t"
            "fsw.ps    f19, 32(%[a1])        \n\t"
            "fsw.ps    f20, 0(%[a2])         \n\t"
            "fsw.ps    f21, 32(%[a2])        \n\t"
            "fsw.ps    f22, 0(%[a3])         \n\t"
            "fsw.ps    f23, 32(%[a3])        \n\t"
            :
            : [a0] "r"(acc_row0 + 8 * RS), [a1] "r"(acc_row0 + 9 * RS), [a2] "r"(acc_row0 + 10 * RS), [a3] "r"(acc_row0 + 11 * RS)
            : "f16", "f17", "f24", "f25", "f18", "f19", "f26", "f27", "f20", "f21", "f28", "f29", "f22", "f23", "f30", "f31", "memory");
    } else if (rows == 11) {
        __asm__ volatile(
            "flw.ps    f24, 0(%[a0])        \n\t"
            "flw.ps    f25, 32(%[a0])       \n\t"
            "flw.ps    f26, 0(%[a1])        \n\t"
            "flw.ps    f27, 32(%[a1])       \n\t"
            "flw.ps    f28, 0(%[a2])        \n\t"
            "flw.ps    f29, 32(%[a2])       \n\t"
            "fadd.ps   f16, f16, f24         \n\t"
            "fadd.ps   f17, f17, f25         \n\t"
            "fadd.ps   f18, f18, f26         \n\t"
            "fadd.ps   f19, f19, f27         \n\t"
            "fadd.ps   f20, f20, f28         \n\t"
            "fadd.ps   f21, f21, f29         \n\t"
            "fsw.ps    f16, 0(%[a0])         \n\t"
            "fsw.ps    f17, 32(%[a0])        \n\t"
            "fsw.ps    f18, 0(%[a1])         \n\t"
            "fsw.ps    f19, 32(%[a1])        \n\t"
            "fsw.ps    f20, 0(%[a2])         \n\t"
            "fsw.ps    f21, 32(%[a2])        \n\t"
            :
            : [a0] "r"(acc_row0 + 8 * RS), [a1] "r"(acc_row0 + 9 * RS), [a2] "r"(acc_row0 + 10 * RS)
            : "f16", "f17", "f24", "f25", "f18", "f19", "f26", "f27", "f20", "f21", "f28", "f29", "memory");
    } else if (rows == 10) {
        __asm__ volatile(
            "flw.ps    f24, 0(%[a0])        \n\t"
            "flw.ps    f25, 32(%[a0])       \n\t"
            "flw.ps    f26, 0(%[a1])        \n\t"
            "flw.ps    f27, 32(%[a1])       \n\t"
            "fadd.ps   f16, f16, f24         \n\t"
            "fadd.ps   f17, f17, f25         \n\t"
            "fadd.ps   f18, f18, f26         \n\t"
            "fadd.ps   f19, f19, f27         \n\t"
            "fsw.ps    f16, 0(%[a0])         \n\t"
            "fsw.ps    f17, 32(%[a0])        \n\t"
            "fsw.ps    f18, 0(%[a1])         \n\t"
            "fsw.ps    f19, 32(%[a1])        \n\t"
            :
            : [a0] "r"(acc_row0 + 8 * RS), [a1] "r"(acc_row0 + 9 * RS)
            : "f16", "f17", "f24", "f25", "f18", "f19", "f26", "f27", "memory");
    } else if (rows == 9) {
        __asm__ volatile(
            "flw.ps    f24, 0(%[a0])        \n\t"
            "flw.ps    f25, 32(%[a0])       \n\t"
            "fadd.ps   f16, f16, f24         \n\t"
            "fadd.ps   f17, f17, f25         \n\t"
            "fsw.ps    f16, 0(%[a0])         \n\t"
            "fsw.ps    f17, 32(%[a0])        \n\t"
            :
            : [a0] "r"(acc_row0 + 8 * RS)
            : "f16", "f17", "f24", "f25", "memory");
    }
}

// Spill the 32 QK^T output registers: f(k) -> sc + 32 k.
static inline void __attribute__((always_inline)) gq_spill_scores(float * sc) {
    unsigned long ms;
    __asm__ volatile(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        "fsw.ps    f0, 0(%[sc])  \n\t"
        "fsw.ps    f1, 32(%[sc])  \n\t"
        "fsw.ps    f2, 64(%[sc])  \n\t"
        "fsw.ps    f3, 96(%[sc])  \n\t"
        "fsw.ps    f4, 128(%[sc])  \n\t"
        "fsw.ps    f5, 160(%[sc])  \n\t"
        "fsw.ps    f6, 192(%[sc])  \n\t"
        "fsw.ps    f7, 224(%[sc])  \n\t"
        "fsw.ps    f8, 256(%[sc])  \n\t"
        "fsw.ps    f9, 288(%[sc])  \n\t"
        "fsw.ps    f10, 320(%[sc])  \n\t"
        "fsw.ps    f11, 352(%[sc])  \n\t"
        "fsw.ps    f12, 384(%[sc])  \n\t"
        "fsw.ps    f13, 416(%[sc])  \n\t"
        "fsw.ps    f14, 448(%[sc])  \n\t"
        "fsw.ps    f15, 480(%[sc])  \n\t"
        "fsw.ps    f16, 512(%[sc])  \n\t"
        "fsw.ps    f17, 544(%[sc])  \n\t"
        "fsw.ps    f18, 576(%[sc])  \n\t"
        "fsw.ps    f19, 608(%[sc])  \n\t"
        "fsw.ps    f20, 640(%[sc])  \n\t"
        "fsw.ps    f21, 672(%[sc])  \n\t"
        "fsw.ps    f22, 704(%[sc])  \n\t"
        "fsw.ps    f23, 736(%[sc])  \n\t"
        "fsw.ps    f24, 768(%[sc])  \n\t"
        "fsw.ps    f25, 800(%[sc])  \n\t"
        "fsw.ps    f26, 832(%[sc])  \n\t"
        "fsw.ps    f27, 864(%[sc])  \n\t"
        "fsw.ps    f28, 896(%[sc])  \n\t"
        "fsw.ps    f29, 928(%[sc])  \n\t"
        "fsw.ps    f30, 960(%[sc])  \n\t"
        "fsw.ps    f31, 992(%[sc])  \n\t"
        "mova.m.x  %[ms]            \n\t"
        : [ms] "=&r"(ms)
        : [sc] "r"(sc)
        : "memory");
}

struct gq_team {
    int64_t iq3, ik2, p0;
    int     p1_valid;
};

static inline struct gq_team gq_team_of(int64_t t, int64_t nq, int64_t nhk) {
    const int64_t tpb = (nq + 1) / 2;  // teams per (batch, KV head)
    struct gq_team tm;
    tm.iq3      = t / (nhk * tpb);
    tm.ik2      = (t / tpb) % nhk;
    tm.p0       = 2 * (t % tpb);
    tm.p1_valid = tm.p0 + 1 < nq;
    return tm;
}

// Accumulate KV tiles [tile_lo, tile_hi) of team tm into this minion's L2-scratchpad slot: per lane the running max M
// and sum S, per query row the unnormalised accumulator. An empty range leaves M at the sentinel and S, acc at 0.
static void gq_team_slab(struct ggml_et_flash_attn_ext_params * params, char * scp, struct gq_team tm, int64_t tile_lo,
                         int64_t tile_hi) {
    const struct ggml_tensor * q    = &params->src0;
    const struct ggml_tensor * k    = &params->src1;
    const struct ggml_tensor * v    = &params->src2;
    const struct ggml_tensor * mask = params->has_mask ? &params->mask : (const struct ggml_tensor *) 0;

    const int64_t dk    = q->ne[0];
    const int64_t dv    = v->ne[0];
    const int64_t gqa   = q->ne[2] / k->ne[2];
    const int64_t nc    = dk / TILE_K;
    const float   scale = params->scale;
    const int64_t iq3 = tm.iq3, ik2 = tm.ik2, p0 = tm.p0;
    const int     p1_valid = tm.p1_valid;

    // P x V in one FMA of 2*gqa rows when the outputs leave f30/f31 free for the accumulator, else one per position
    const int64_t pv_rows   = (2 * gqa <= 12) ? 2 * gqa : gqa;
    const int64_t pv_groups = (2 * gqa) / pv_rows;

    char *  qt  = scp + GQ_QT_OFF;
    float * acc = (float *) (scp + GQ_ACC_OFF);
    float * st  = (float *) (scp + GQ_ST_OFF);
    float * sc  = (float *) (scp + GQ_SC_OFF);

    for (int64_t i = 0; i < 2 * GQ_LANES; ++i) {
        st[GQ_ST_M + i] = ET_NEG_INF_F;
        st[GQ_ST_S + i] = 0.0f;
    }
    for (int64_t r = 0; r < 2 * gqa; ++r) {
        zero_acc_vec(acc + r * (GQ_ACC_ROW / sizeof(float)), dv);
    }
    if (tile_lo >= tile_hi) {
        return;
    }

    gq_build_qt(qt, (const char *) q->data, q, dk, gqa, ik2, p0, p1_valid, iq3);

    const char * mrow[2] = { 0, 0 };
    if (mask) {
        const char * mb = (const char *) mask->data + (iq3 % mask->ne[3]) * mask->nb[3];
        mrow[0]         = mb + p0 * mask->nb[1];
        mrow[1]         = p1_valid ? mb + (p0 + 1) * mask->nb[1] : 0;
    }
    const char *   k_head = (const char *) k->data + ik2 * k->nb[2] + iq3 * k->nb[3];
    const char *   v_head = (const char *) v->data + ik2 * v->nb[2] + iq3 * v->nb[3];
    const uint64_t nb1_k  = (uint64_t) k->nb[1];
    const uint64_t nb1_v  = (uint64_t) v->nb[1];

    for (int64_t kv_base = tile_lo * TILE_KV; kv_base < tile_hi * TILE_KV; kv_base += TILE_KV) {
        const int cls0 = gq_mask_class(mrow[0], kv_base);
        const int cls1 = p1_valid ? gq_mask_class(mrow[1], kv_base) : GQ_MASKED;
        if (cls0 == GQ_MASKED && cls1 == GQ_MASKED) {
            continue;
        }

        // QK^T: A = 16 K rows of chunk c (load id 0, double-buffered), B = Q^T chunk c through TenB
        const char * kt = k_head + kv_base * (int64_t) nb1_k;
        tensor_load(false, false, GQ_KB0, TENSOR_LOAD_PLAIN, 0, (uint64_t) kt, 0, 15, nb1_k, 0);
        for (int64_t c = 0; c < nc; ++c) {
            tensor_wait(TENSOR_LOAD_WAIT_0);
            if (c + 1 < nc) {
                tensor_load(false, false, ((c + 1) & 1) ? GQ_KB1 : GQ_KB0, TENSOR_LOAD_PLAIN, 0,
                            (uint64_t) (kt + (c + 1) * 64), 0, 15, nb1_k, 0);
            }
            tensor_load_setup_b(false, (uint64_t) (qt + c * GQ_QT_CHUNK), 15, 64, 1);
            tensor_fma(false, 3, 15, 15, 0, false, false, false, true, 0, (c & 1) ? GQ_KB1 : GQ_KB0,
                       TENSOR_FMA_OP_FP16, c == 0);
            GQ_CLOBBER_F;
            tensor_wait(TENSOR_FMA_WAIT);
            GQ_CLOBBER_F;
        }
        gq_spill_scores(sc);
        GQ_CLOBBER_F;

        gq_softmax_half(scp, 0, cls0, mrow[0], kv_base, gqa, scale, dv);
        gq_softmax_half(scp, 1, cls1, mrow[1], kv_base, gqa, scale, dv);

        // P x V: A = the 2*gqa probability rows, B = V by INTERLEAVE16 (load id 1, double-buffered)
        FENCE;
        flush_to_l2(scp + GQ_P_OFF, (uint64_t) (2 * gqa), 64);
        WAIT_CACHEOPS;
        tensor_load(false, false, GQ_PL1, TENSOR_LOAD_PLAIN, 0, (uint64_t) (scp + GQ_P_OFF), 0,
                    (uint64_t) (2 * gqa - 1), 64, 0);
        const uintptr_t v_base = (uintptr_t) v_head + kv_base * (int64_t) nb1_v;
        uint64_t        b_cur  = GQ_VB0;
        tensor_load(false, false, b_cur, TENSOR_LOAD_INTERLEAVE16, 0, (uint64_t) v_base, 0, 7, nb1_v, 1);
        tensor_wait(TENSOR_LOAD_WAIT_0);
        tensor_wait(TENSOR_LOAD_WAIT_1);

        unsigned long ms;
        __asm__ volatile("mova.x.m  %[ms]          \n\t"
                         "mov.m.x   m0, x0, 0xFF   \n\t"
                         : [ms] "=&r"(ms));
        for (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
            const uint64_t b_nxt = (b_cur == GQ_VB0) ? GQ_VB1 : GQ_VB0;
            if (dv_off + 16 < dv) {
                tensor_load(false, false, b_nxt, TENSOR_LOAD_INTERLEAVE16, 0, (uint64_t) (v_base + (dv_off + 16) * 2),
                            0, 7, nb1_v, 1);
            }
            for (int64_t grp = 0; grp < pv_groups; ++grp) {
                tensor_fma(false, 3, (uint64_t) (pv_rows - 1), 7, 0, false, false, false, false, b_cur,
                           (uint64_t) (GQ_PL1 + grp * pv_rows), TENSOR_FMA_OP_FP16, true);
                GQ_CLOBBER_F;
                tensor_wait(TENSOR_FMA_WAIT);
                GQ_CLOBBER_F;
                gq_acc_add(acc + grp * pv_rows * (GQ_ACC_ROW / sizeof(float)) + dv_off, pv_rows);
                GQ_CLOBBER_F;
            }
            if (dv_off + 16 < dv) {
                tensor_wait(TENSOR_LOAD_WAIT_1);
                b_cur = b_nxt;
            }
        }
        __asm__ volatile("mova.m.x  %[ms]          \n\t" : : [ms] "r"(ms));
    }

}

// Normalise and store each query row of a team whose whole KV ran on this minion (whole 64-byte lines, one writer).
static void gq_store_rows(struct ggml_et_flash_attn_ext_params * params, char * scp, struct gq_team tm) {
    const struct ggml_tensor * dst = &params->dst;
    const int64_t              gqa = params->src0.ne[2] / params->src1.ne[2];
    const int64_t              dv  = params->src2.ne[0];
    float *                    acc = (float *) (scp + GQ_ACC_OFF);
    const float *              st  = (const float *) (scp + GQ_ST_OFF);
    for (int64_t r = 0; r < 2 * gqa; ++r) {
        const int64_t h = r / gqa;
        if (h == 1 && !tm.p1_valid) {
            break;
        }
        const int64_t lane  = r % gqa;
        float *       out   = (float *) ((char *) dst->data + (tm.ik2 * gqa + lane) * dst->nb[1] + (tm.p0 + h) * dst->nb[2] +
                                       tm.iq3 * dst->nb[3]);
        const float   S     = st[GQ_ST_S + h * GQ_LANES + lane];
        const float   S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        normalize_store_vec(out, acc + r * (GQ_ACC_ROW / sizeof(float)), dv, S_inv, 1);
    }
}

// One output line (row r, 16 floats at chunk c) of a team split over ks members: the members' partials are combined
// with the softmax rule (M* = max M_m, alpha_m = exp2((M_m - M*) log2e), S* = sum alpha_m S_m, acc* = sum alpha_m acc_m)
// and normalised. Members whose range saw no unmasked score (M_m at the sentinel) contribute nothing.
static void gq_merge_line(struct ggml_et_flash_attn_ext_params * params, struct gq_team tm, int64_t slot0, int64_t ks,
                          int64_t r, int64_t c) {
    const struct ggml_tensor * dst  = &params->dst;
    const int64_t              gqa  = params->src0.ne[2] / params->src1.ne[2];
    const int64_t              h    = r / gqa;
    const int64_t              lane = r % gqa;
    const int64_t              k    = h * GQ_LANES + lane;
    const float                log2e = 1.4426950408889634f;

    // drop this minion's L1 copies of the peers' stats and accumulator lines (16 members per evict, slots GQ_PER_MINION apart)
    for (int64_t m0 = 0; m0 < ks; m0 += 16) {
        const int64_t cnt  = ks - m0 < 16 ? ks - m0 : 16;
        char *        base = (char *) et_shire_l2scp_local((slot0 + m0) * GQ_PER_MINION);
        evict_to_l2(base + GQ_ST_OFF, (uint64_t) cnt, GQ_PER_MINION);
        evict_to_l2(base + GQ_ST_OFF + 64, (uint64_t) cnt, GQ_PER_MINION);
        evict_to_l2(base + GQ_ACC_OFF + r * GQ_ACC_ROW + c * 64, (uint64_t) cnt, GQ_PER_MINION);
    }
    WAIT_CACHEOPS;
    __asm__ volatile("" ::: "memory");  // WAIT_CACHEOPS has no memory clobber: keep the peer loads after it

    float Mx = ET_NEG_INF_F;
    for (int64_t m = 0; m < ks; ++m) {
        const float * st = (const float *) ((char *) et_shire_l2scp_local((slot0 + m) * GQ_PER_MINION) + GQ_ST_OFF);
        const float   Mm = st[GQ_ST_M + k];
        Mx               = Mm > Mx ? Mm : Mx;
    }

    // running sum of alpha_m * acc_m in memory: compiler float code (exp2) runs between the asm statements
    float         sum[16] __attribute__((aligned(64)));
    float         Sx = 0.0f;
    unsigned long ms;
    zero_acc_vec(sum, 16);
    for (int64_t m = 0; m < ks; ++m) {
        const char *  slot = (const char *) et_shire_l2scp_local((slot0 + m) * GQ_PER_MINION);
        const float * st   = (const float *) (slot + GQ_ST_OFF);
        const float   Mm   = st[GQ_ST_M + k];
        if (!(Mm > ET_NEG_INF_F)) {
            continue;
        }
        const float alpha = et_exp2f((Mm - Mx) * log2e);
        Sx += alpha * st[GQ_ST_S + k];
        const float * a = (const float *) (slot + GQ_ACC_OFF + r * GQ_ACC_ROW + c * 64);
        __asm__ volatile("mova.x.m  %[ms]              \n\t"
                         "mov.m.x   m0, x0, 0xFF       \n\t"
                         "fbc.ps    f22, 0(%[al])      \n\t"
                         "flw.ps    f20, 0(%[s])       \n\t"
                         "flw.ps    f21, 32(%[s])      \n\t"
                         "flw.ps    f23, 0(%[a])       \n\t"
                         "flw.ps    f24, 32(%[a])      \n\t"
                         "fmadd.ps  f20, f23, f22, f20 \n\t"
                         "fmadd.ps  f21, f24, f22, f21 \n\t"
                         "fsw.ps    f20, 0(%[s])       \n\t"
                         "fsw.ps    f21, 32(%[s])      \n\t"
                         "mova.m.x  %[ms]              \n\t"
                         : [ms] "=&r"(ms)
                         : [al] "r"(&alpha), [a] "r"(a), [s] "r"(sum)
                         : "f20", "f21", "f22", "f23", "f24", "memory");
    }
    const float inv = Sx == 0.0f ? 0.0f : et_fdiv(1.0f, Sx);
    float *     out = (float *) ((char *) dst->data + (tm.ik2 * gqa + lane) * dst->nb[1] + (tm.p0 + h) * dst->nb[2] +
                             tm.iq3 * dst->nb[3]) + c * 16;
    __asm__ volatile("mova.x.m  %[ms]            \n\t"
                     "mov.m.x   m0, x0, 0xFF     \n\t"
                     "fbc.ps    f22, 0(%[iv])    \n\t"
                     "flw.ps    f20, 0(%[s])     \n\t"
                     "flw.ps    f21, 32(%[s])    \n\t"
                     "fmul.ps   f20, f20, f22    \n\t"
                     "fmul.ps   f21, f21, f22    \n\t"
                     "fsw.ps    f20, 0(%[o])     \n\t"
                     "fsw.ps    f21, 32(%[o])    \n\t"
                     "mova.m.x  %[ms]            \n\t"
                     : [ms] "=&r"(ms)
                     : [iv] "r"(&inv), [o] "r"(out), [s] "r"(sum)
                     : "f20", "f21", "f22", "memory");
}

static int fa_gqa_tiles(struct ggml_et_flash_attn_ext_params * params, uint64_t shire_id, uint64_t local_minion,
                        int is_hart1) {
    const int64_t nq       = params->src0.ne[1];
    const int64_t no       = params->src0.ne[3];
    const int64_t nhk      = params->src1.ne[2];
    const int64_t gqa      = params->src0.ne[2] / nhk;
    const int64_t dv       = params->src2.ne[0];
    const int64_t nk_tiles = params->src1.ne[1] / TILE_KV;
    const int64_t n_teams  = no * nhk * ((nq + 1) / 2);

    // ks members per team, as the row path picks its split: minimise passes x (tiles per member + one for the merge).
    // Mode 1 keeps ks = 1 (each team's whole KV on one minion: no merge, no barrier).
    int64_t ks = 1;
    if (params->gqa_tiles >= 2) {
        int64_t best = -1;
        for (int64_t s = 1; s <= MINIONS_PER_SHIRE && s <= nk_tiles; s *= 2) {
            const int64_t passes = (n_teams * s + NUM_COMPUTE_SHIRES * MINIONS_PER_SHIRE - 1) /
                                   (NUM_COMPUTE_SHIRES * MINIONS_PER_SHIRE);
            const int64_t cost   = passes * ((nk_tiles + s - 1) / s + (s > 1 ? 1 : 0));
            if (best < 0 || cost < best) {
                best = cost;
                ks   = s;
            }
        }
    }
    const int64_t tps    = MINIONS_PER_SHIRE / ks;  // teams per shire per pass
    const int64_t passes = (n_teams + NUM_COMPUTE_SHIRES * tps - 1) / (NUM_COMPUTE_SHIRES * tps);
    const int64_t ts     = (int64_t) local_minion / ks;  // team slot in the shire
    const int64_t km     = (int64_t) local_minion % ks;  // member within the team
    char *        scp    = (char *) et_shire_l2scp_local(local_minion * GQ_PER_MINION);

    if (ks == 1) {
        if (is_hart1) {
            return 0;
        }
        for (int64_t p = 0; p < passes; ++p) {
            const int64_t t = (p * NUM_COMPUTE_SHIRES + (int64_t) shire_id) * tps + ts;
            if (t < n_teams) {
                const struct gq_team tm = gq_team_of(t, nq, nhk);
                gq_team_slab(params, scp, tm, 0, nk_tiles);
                gq_store_rows(params, scp, tm);
            }
        }
        FENCE;
        return 0;
    }

    // Split teams: every hart of the shire takes both shire barriers of every pass, work or no work.
    const int64_t per = (nk_tiles + ks - 1) / ks;
    for (int64_t p = 0; p < passes; ++p) {
        const int64_t t    = (p * NUM_COMPUTE_SHIRES + (int64_t) shire_id) * tps + ts;
        const int     have = t < n_teams;
        struct gq_team tm  = { 0, 0, 0, 0 };
        if (have) {
            tm = gq_team_of(t, nq, nhk);
        }
        if (!is_hart1 && have) {
            const int64_t lo = km * per;
            const int64_t hi = lo + per < nk_tiles ? lo + per : nk_tiles;
            gq_team_slab(params, scp, tm, lo, hi);
            // publish this member's partial to the shire's L2 for the merge
            FENCE;
            evict_to_l2(scp + GQ_ST_OFF, 2, 64);
            for (int64_t r = 0; r < 2 * gqa; ++r) {
                evict_range_to_l2(scp + GQ_ACC_OFF + r * GQ_ACC_ROW, dv * (int64_t) sizeof(float));
            }
            WAIT_CACHEOPS;
        }
        et_barrier(ET_BARRIER_SHIRE);  // A: every member's partial is in L2
        if (!is_hart1 && have) {
            const int64_t rows   = tm.p1_valid ? 2 * gqa : gqa;
            const int64_t chunks = dv / 16;
            for (int64_t piece = km; piece < rows * chunks; piece += ks) {
                gq_merge_line(params, tm, ts * ks, ks, piece / chunks, piece % chunks);
            }
            FENCE;
        }
        et_barrier(ET_BARRIER_SHIRE);  // B: the merge has read every slot before the next pass rewrites them
    }
    return 0;
}

int entry_point(struct ggml_et_flash_attn_ext_params * params, void * env) {
    (void) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    if (shire_id >= NUM_COMPUTE_SHIRES) {
        return 0;
    }

    const int is_hart1     = hart_id & 1;
    uint64_t  local_minion = (hart_id >> 1) & 0x1F;

    struct ggml_tensor * q = &params->src0;
    struct ggml_tensor * k = &params->src1;
    struct ggml_tensor * v = &params->src2;
#ifndef ET_UBERKERNEL
    // Standalone launch: drop stale copies of the inputs, as the uberkernel's dispatcher does between instructions:
    // every minion pushes its whole L1 to L2, then the 64 harts of the shire split the walk of Q, the mask and the K/V
    // views past L2 (line h, h + 64, ...), so each line is walked once per shire instead of once per hart. The shire
    // barrier between the two steps keeps a hart's L2 walk from racing another hart's L1 evict of the same line; the
    // entry shire barrier below orders the walk before any read.
    cache_ops_priv_evict_l1(0, to_L2);
    WAIT_CACHEOPS;
    et_barrier(ET_BARRIER_SHIRE);
    {
        const uint64_t h = hart_id & 63;
        fa_evict_split(q->data, fa_span_bytes(q), h);
        if (params->has_mask) {
            fa_evict_split(params->mask.data, fa_span_bytes(&params->mask), h);
        }
        fa_evict_split(k->data, fa_span_bytes(k), h);
        fa_evict_split(v->data, fa_span_bytes(v), h);
        WAIT_CACHEOPS;
    }
#endif
    struct ggml_tensor * dst      = &params->dst;
    const int32_t        has_mask = params->has_mask;
    struct ggml_tensor * mask     = has_mask ? &params->mask : (struct ggml_tensor *) 0;

    const char * q_data   = (const char *) q->data;
    const char * k_data   = (const char *) k->data;
    const char * v_data   = (const char *) v->data;
    char *       dst_data = (char *) dst->data;

    // et_barrier(ET_BARRIER_GLOBAL);
    // evict_region_past_l2(q_data,   tensor_bytes_fa(q));
    // evict_region_past_l2(k_data,   tensor_bytes_fa(k));
    // evict_region_past_l2(v_data,   tensor_bytes_fa(v));
    // if (mask) {
    // evict_region_past_l2(mask->data, tensor_bytes_fa(mask));
    // }
    // et_barrier(ET_BARRIER_GLOBAL);

    const int64_t dk  = q->ne[0];
    const int64_t nq  = q->ne[1];
    const int64_t nhq = q->ne[2];
    const int64_t no  = q->ne[3];
    const int64_t nk  = k->ne[1];
    const int64_t nhk = k->ne[2];
    const int64_t dv  = v->ne[0];

    if (dv > FA_DV_MAX || dk > FA_DK_MAX) {
        return -1;
    }
    if (k->nb[0] != 2 || v->nb[0] != 2) {
        return -1;
    }
    if ((dk % 8) != 0 || (dv % 16) != 0) {
        return -1;
    }

    const int64_t gqa_ratio      = nhq / nhk;
    const int64_t total_rows     = nq * nhq * no;
    const float   scale          = params->scale;
    const int     use_fast_store = (dv % 16 == 0);

    // Split-KV team layout (mirrors mul_mat_f16_matrix_engine.c)
    //
    // When total_rows is small compared to the total minion count (typical
    // for decode: nq=1, nhq small), we group k_splits minions within the
    // same shire into a team that cooperates on one row by splitting the
    // KV dimension. Each team member computes a partial (M_p, S_p, acc_p)
    // over its KV slab; the k_split==0 member merges the partials with the
    // softmax combine rule.
    //
    // k_splits is a power of two, capped at MINIONS_PER_SHIRE (so a team
    // never spans shires — L2 SCP is shire-local) and at nk_tiles (so each
    // team member gets at least one KV tile).
    //
    // A minion's time is set by the KV tiles it walks (each tile is a chain of dependent loads), so pick the
    // k_splits that minimises passes × (tiles per slab + one tile's worth for the merge): 72 decode rows at
    // 3.5 k positions take three passes over 7 tiles at k_splits 32 instead of two over 14 at 16. Ties keep
    // the smaller split (fewer merges).
    const int64_t nk_tiles = (nk + TILE_KV - 1) / TILE_KV;
    int64_t       k_splits = 1;
    {
        int64_t best_cost = -1;
        for (int64_t ks = 1; ks <= MINIONS_PER_SHIRE && ks <= nk_tiles; ks *= 2) {
            const int64_t teams  = (int64_t) NUM_COMPUTE_SHIRES * (MINIONS_PER_SHIRE / ks);
            const int64_t passes = (total_rows + teams - 1) / teams;
            const int64_t cost   = passes * ((nk_tiles + ks - 1) / ks + (ks > 1 ? 1 : 0));
            if (best_cost < 0 || cost < best_cost) {
                best_cost = cost;
                k_splits  = ks;
            }
        }
    }

    const int64_t tiles_per_shire = MINIONS_PER_SHIRE / k_splits;
    const int64_t k_split         = (int64_t) local_minion % k_splits;
    const int64_t local_tile_idx  = (int64_t) local_minion / k_splits;
    const int64_t tiles_stride    = (int64_t) NUM_COMPUTE_SHIRES * tiles_per_shire;

    // KV slab for this k_split. With k_splits=1 this is the full range.
    const int64_t tiles_per_split_rounded = (nk_tiles + k_splits - 1) / k_splits;
    const int64_t tile_start              = k_split * tiles_per_split_rounded;
    int64_t       tile_end                = tile_start + tiles_per_split_rounded;
    if (tile_end > nk_tiles) {
        tile_end = nk_tiles;
    }
    const int64_t kv_start = tile_start * TILE_KV;
    int64_t       kv_end   = tile_end * TILE_KV;
    if (kv_end > nk) {
        kv_end = nk;
    }

    // L2 SCP pointers for this minion
    uint64_t    scp_base  = local_minion * SCP_PER_MINION;
    et_fp16_t * scp_kp[2] = {
        (et_fp16_t *) et_shire_l2scp_local(scp_base + SCP_KP0_OFF),
        (et_fp16_t *) et_shire_l2scp_local(scp_base + SCP_KP1_OFF),
    };

    // Hart 1 does K-panel packing
    //
    // When k_splits > 1, hart 1 must also participate in the two shire
    // barriers that bracket the merge phase (one before and one after, so
    // the reducer can read peer partials safely and the writers know when
    // their acc/stats slab is free to reuse). Hart 1 has no useful work
    // between those barriers.
    //
    // All teams in a shire must iterate the same number of times so the
    // per-iter shire barriers stay balanced. Teams whose assigned row is
    // past total_rows still call the barriers but skip the packing work.

    et_barrier(ET_BARRIER_SHIRE);
    // et_barrier(ET_BARRIER_GLOBAL);

    // The decision depends on launch parameters only, so every hart takes the same path.
    if (fa_gqa_eligible(params)) {
        return fa_gqa_tiles(params, shire_id, local_minion, is_hart1);
    }

    if (is_hart1) {
        // et_barrier(ET_BARRIER_GLOBAL);
        // et_barrier(ET_BARRIER_SHIRE);

        uint32_t      chunk_id = 0;
        const int64_t row_base = (int64_t) shire_id + local_tile_idx * NUM_COMPUTE_SHIRES;

        int64_t max_iters;
        if (k_splits > 1) {
            max_iters = (total_rows + tiles_stride - 1) / tiles_stride;
        } else {
            max_iters = (row_base >= total_rows) ? 0 : ((total_rows - row_base - 1) / tiles_stride + 1);
        }

        for (int64_t iter = 0; iter < max_iters; iter++) {
            const int64_t row      = row_base + iter * tiles_stride;
            const int     has_work = (row < total_rows);

            if (has_work) {
                const int64_t iq3 = row / (nhq * nq);
                const int64_t rem = row % (nhq * nq);
                const int64_t iq2 = rem / nq;
                const int64_t ik2 = iq2 / gqa_ratio;

                const char * k_head = k_data + ik2 * k->nb[2] + iq3 * k->nb[3];

                for (int64_t kv_base = kv_start; kv_base < kv_end; kv_base += TILE_KV) {
                    const int64_t kv_count = (kv_base + TILE_KV <= nk) ? TILE_KV : (nk - kv_base);

                    for (int64_t dk_chunk = 0; dk_chunk < dk; dk_chunk += TILE_K) {
                        int buf = chunk_id & 1;

                        // Back-pressure: before overwriting buf[buf] on chunk N
                        // (which will displace chunk N-2), wait for hart 0 to
                        // post that it's done with chunk N-2. Gates both
                        // directions of double-buffering.
                        //
                        // NOTE: we use et_sem_* (FCC 0 only) rather than
                        // et_barrier(ET_BARRIER_MINION) here because the
                        // minion barrier for minion 0 shares FLB 0 with
                        // ET_BARRIER_SHIRE. Mixing them deadlocks. See
                        // feedback_flb_collision.
                        if (chunk_id >= 2) {
                            et_sem_wait(ET_BARRIER_MINION);
                        }

                        // Prefetch K data for this chunk
                        prefetch_kv_to_l2(k_head, kv_base, dk_chunk, kv_count, k->nb[1]);

                        pack_k_for_transpose16(scp_kp[buf], k_head, kv_base, dk_chunk, kv_count, k->nb[1]);

                        FENCE;
                        flush_to_l2(scp_kp[buf], 16, 64);
                        flush_to_l2((et_fp16_t *) ((char *) scp_kp[buf] + 1024), 16, 64);
                        WAIT_CACHEOPS;

                        // Signal: this buf is ready for hart 0 to consume.
                        et_sem_post(ET_BARRIER_MINION);

                        chunk_id++;
                    }
                }
            }

            // Shire barriers for split-KV merge (hart 1 is a passive arrival).
            if (k_splits > 1) {
                et_barrier(ET_BARRIER_SHIRE);  // A: team has written its partial
                et_barrier(ET_BARRIER_SHIRE);  // B: reducer has finished merge
            }
        }

        // Self-drain phantom FCC 0 credits left by the wait-skip on the
        // first 2 chunks. Hart 1 issued chunk_id posts but only
        // (chunk_id - 2) waits (when chunk_id >= 2), so hart 1's FCC 0
        // carries +min(chunk_id,2) credits from hart 0's matching posts
        // that hart 1 never consumed.
        uint32_t drain = (chunk_id < 2) ? chunk_id : 2;
        for (uint32_t d = 0; d < drain; d++) {
            et_sem_wait(ET_BARRIER_MINION);
        }

        // FENCE;
        // et_barrier(ET_BARRIER_GLOBAL);
        return 0;
    }

    // Hart 0: tensor engine compute
    // #ifndef UBERKERNEL_SUPPRESS_SCP_SETUP
    //     setup_cache_scp();
    // #endif
    // CLEAR_TENSOR_ERROR;

    // Q converted to F16 (one row at a time)
    et_fp16_t q_f16[FA_DK_MAX] __attribute__((aligned(64)));

    // Score buffer for QK^T output (16 scores per KV tile)
    float scores[TILE_KV] __attribute__((aligned(64)));

    // Small buffers for V accumulation
    et_fp16_t w_f16_buf[32] __attribute__((aligned(64)));       // 64 bytes
    et_fp16_t vpanel_buf[8 * 32] __attribute__((aligned(64)));  // 512 bytes

    float * acc = (float *) et_shire_l2scp_local(scp_base + SCP_ACC_OFF);

    uint32_t chunk_id = 0;

    // Iter-based outer loop (matches hart 1). When k_splits > 1 all teams
    // in a shire iterate the same number of times so the per-row shire
    // barriers stay balanced; iterations with row >= total_rows skip the
    // compute but still participate in the barriers.
    const int64_t hart0_row_base = (int64_t) shire_id + local_tile_idx * NUM_COMPUTE_SHIRES;
    int64_t       hart0_max_iters;
    if (k_splits > 1) {
        hart0_max_iters = (total_rows + tiles_stride - 1) / tiles_stride;
    } else {
        hart0_max_iters = (hart0_row_base >= total_rows) ? 0 : ((total_rows - hart0_row_base - 1) / tiles_stride + 1);
    }

    for (int64_t iter = 0; iter < hart0_max_iters; iter++) {
        const int64_t row = hart0_row_base + iter * tiles_stride;
        if (row >= total_rows) {
            // No-work iteration: only participate in barriers (k_splits > 1).
            if (k_splits > 1) {
                et_barrier(ET_BARRIER_SHIRE);  // A
                et_barrier(ET_BARRIER_SHIRE);  // B
            }
            continue;
        }

        const int64_t iq3 = row / (nhq * nq);
        const int64_t rem = row % (nhq * nq);
        const int64_t iq2 = rem / nq;
        const int64_t iq1 = rem % nq;
        const int64_t ik2 = iq2 / gqa_ratio;

        // Read Q row (F32) and convert to F16
        const float * pq = (const float *) (q_data + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);
        convert_q_row_f32_to_f16(q_f16, pq, dk);

        // V base for this head + batch (K packing handled by hart 1)
        const char * v_head = v_data + ik2 * v->nb[2] + iq3 * v->nb[3];

        // Output pointer
        float * out = (float *) (dst_data + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        zero_acc_vec(acc, dv);
        float        M         = ET_NEG_INF_F;
        float        S         = 0.0f;
        const char * mask_base = has_mask ? get_mask_row_base(mask, iq1, iq2, iq3) : (const char *) 0;

        // Flush Q_f16 to L2 so tensor_load can see it
        FENCE;
        flush_to_l2(q_f16, (dk * 2 + 63) / 64, 64);
        WAIT_CACHEOPS;

        for (int64_t kv_base = kv_start; kv_base < kv_end; kv_base += TILE_KV) {
            const int64_t kv_count = (kv_base + TILE_KV <= nk) ? TILE_KV : (nk - kv_base);

            // Set tensor_mask for partial tiles
            if (kv_count < TILE_KV) {
                uint64_t tmask = (1ULL << kv_count) - 1;
                __asm__ __volatile__("csrw 0x805, %0" : : "r"(tmask));
            }

            // ============================================================
            // QK^T via TensorFMA16A32
            // ============================================================

            // Pipelined QK^T:
            //   - Q for the whole row is preloaded once into A_L1[0..n-1].
            //     Each FMA picks its chunk via scp_loc_a = chunk_idx.
            //   - K is double-buffered in L1: K_BUFS[0]=lines 16..31,
            //     K_BUFS[1]=lines 32..47.
            //   - In iteration i (1..N-1), the K[i] load runs concurrently
            //     with the FMA on chunk i-1: they touch disjoint L1 regions
            //     (FMA reads K_BUFS[(i-1)&1], load writes K_BUFS[i&1]; FMA
            //     reads A_L1[i-1], load doesn't touch A_L1).
            //
            // L1 footprint: max dk=512 → Q uses 16 lines (0..15), K uses 32
            // lines (16..47). Within ET-SoC-1 L1 SCP (≥128 lines per minion).
            const int64_t  n_dk_chunks = dk / TILE_K;
            const uint64_t K_BUFS[2]   = {
                (uint64_t) B_L1_START,         // 16..31
                (uint64_t) (B_L1_START + 16),  // 32..47
            };

            // Preload entire Q row into A_L1[0..n_dk_chunks-1] (one tensor_load,
            // one wait, regardless of dk).
            tensor_load(false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0, (uint64_t) q_f16, 0,
                        (uint64_t) (n_dk_chunks - 1), 64, 0);

            // Prologue: wait hart 1's K[0], issue K[0] load, wait both loads.
            {
                int buf = chunk_id & 1;
                et_sem_wait(ET_BARRIER_MINION);
                tensor_load(false, false, K_BUFS[0], TENSOR_LOAD_TRANSPOSE16, 0, (uint64_t) scp_kp[buf], 0, 15, 64, 1);
                tensor_wait(TENSOR_LOAD_WAIT_0);  // Q row complete
                tensor_wait(TENSOR_LOAD_WAIT_1);  // K[0] complete
                et_sem_post(ET_BARRIER_MINION);
                chunk_id++;
            }

            // Main loop: in iter i, issue K[i] load and FMA chunk i-1 in
            // parallel. The matrix engine is busy on FMA[i-1] while the
            // load unit fetches K[i] from L2 SCP.
            //
            // Order of waits matters: wait K[i] load first, then sem_post
            // immediately (frees scp_kp[buf] for hart 1 to refill chunk i+2),
            // then wait FMA. Putting sem_post after FMA wait would stall
            // hart 1 by a full FMA latency — defeating the producer pipeline.
            for (int64_t i = 1; i < n_dk_chunks; i++) {
                int buf         = chunk_id & 1;
                int k_slot_prev = (int) ((i - 1) & 1);
                int k_slot      = (int) (i & 1);

                et_sem_wait(ET_BARRIER_MINION);
                tensor_load(false, false, K_BUFS[k_slot], TENSOR_LOAD_TRANSPOSE16, 0, (uint64_t) scp_kp[buf], 0, 15, 64,
                            1);

                tensor_fma((kv_count < TILE_KV), 3, 0, 15, 0, false, false, false, false, K_BUFS[k_slot_prev],
                           (uint64_t) (i - 1), TENSOR_FMA_OP_FP16, (i == 1));

                tensor_wait(TENSOR_LOAD_WAIT_1);  // K[i] in L1
                et_sem_post(ET_BARRIER_MINION);   // release scp_kp[buf] EARLY
                tensor_wait(TENSOR_FMA_WAIT);     // then wait FMA[i-1]
                chunk_id++;
            }

            // Epilogue: FMA on the last chunk (no overlapping load).
            {
                int k_slot_last = (int) ((n_dk_chunks - 1) & 1);
                tensor_fma((kv_count < TILE_KV), 3, 0, 15, 0, false, false, false, false, K_BUFS[k_slot_last],
                           (uint64_t) (n_dk_chunks - 1), TENSOR_FMA_OP_FP16, (n_dk_chunks == 1));
                tensor_wait(TENSOR_FMA_WAIT);
            }

            // Prefetch V rows for this tile.
            // Only useful for the partial-tile path below
            if (kv_count < TILE_KV) {
                for (int64_t d = 0; d < dv; d += 32) {
                    prefetch_kv_to_l2(v_head, kv_base, d, kv_count, v->nb[1]);
                }
            }

            // Extract QK^T scores from vector register file
            __asm__ volatile("" ::: "f0", "f1");
            {
                unsigned long _ms;
                __asm__ volatile(
                    "mova.x.m  %[ms]                \n\t"
                    "mov.m.x   m0, x0, 0xFF         \n\t"
                    "fbc.ps    f2, 0(%[p_scale])    \n\t"
                    "fmul.ps   f0, f0, f2           \n\t"
                    "fmul.ps   f1, f1, f2           \n\t"
                    "fsw.ps    f0, 0(%[dst])        \n\t"
                    "fsw.ps    f1, 32(%[dst])       \n\t"
                    "mova.m.x  %[ms]                \n\t"
                    : [ms] "=&r"(_ms)
                    : [dst] "r"(scores), [p_scale] "r"(&scale)
                    : "f0", "f1", "f2", "memory");
            }

            // ============================================================
            // Two-phase softmax + V accumulation
            // ============================================================

            float weights[TILE_KV] __attribute__((aligned(64)));
            {
                // A1: apply mask to scores, pad unused slots
                for (int64_t j = 0; j < kv_count; ++j) {
                    if (has_mask) {
                        float mv = get_mask_val_from_base(mask, mask_base, kv_base + j);
                        if (mv == ET_NEG_INF_F || mv != mv) {
                            scores[j] = ET_NEG_INF_F;
                        } else {
                            scores[j] += mv;
                        }
                    }
                }
                for (int64_t j = kv_count; j < TILE_KV; ++j) {
                    scores[j] = ET_NEG_INF_F;
                }

                // A1b: SIMD horizontal max across all 16 scores
                float tile_max;
                {
                    unsigned long _ms;
                    __asm__ volatile(
                        "mova.x.m  %[ms]              \n\t"
                        "mov.m.x   m0, x0, 0xFF       \n\t"
                        "flw.ps    f2, 0(%[sc])       \n\t"
                        "flw.ps    f3, 32(%[sc])      \n\t"
                        "fmax.ps   f2, f2, f3         \n\t"
                        "fswizz.ps f3, f2, 0xB1       \n\t"
                        "fmax.ps   f2, f2, f3         \n\t"
                        "fswizz.ps f3, f2, 0x4E       \n\t"
                        "fmax.ps   f2, f2, f3         \n\t"
                        "fmvz.x.ps t0, f2, 4          \n\t"
                        "fbcx.ps   f3, t0             \n\t"
                        "fmax.ps   %[tm], f2, f3      \n\t"
                        "mova.m.x  %[ms]              \n\t"
                        : [ms] "=&r"(_ms), [tm] "=f"(tile_max)
                        : [sc] "r"(scores)
                        : "f2", "f3", "t0", "memory");
                }

                if (tile_max > ET_NEG_INF_F) {
                    // A2: rescale accumulator if this tile has a new global max
                    if (tile_max > M) {
                        float rescale = et_exp2f((M - tile_max) * 1.4426950408889634f);
                        scale_acc_vec(acc, dv, rescale);
                        S *= rescale;
                        M = tile_max;
                    }

                    // A3: SIMD exp2 + horizontal sum
                    // Interleaved: f2/f3 chains alternate to hide ALU latency.
                    // fexp.ps has multi-cycle latency — the two independent
                    // exp2 calls naturally pipeline.
                    {
                        const float   log2e = 1.4426950408889634f;
                        float         S_tile;
                        unsigned long _ms;
                        __asm__ volatile(
                            "mova.x.m  %[ms]              \n\t"
                            "mov.m.x   m0, x0, 0xFF       \n\t"
                            "flw.ps    f2, 0(%[sc])       \n\t"
                            "fbc.ps    f4, 0(%[pM])       \n\t"
                            "flw.ps    f3, 32(%[sc])      \n\t"
                            "fbc.ps    f5, 0(%[pL])       \n\t"
                            "fsub.ps   f2, f2, f4         \n\t"
                            "fsub.ps   f3, f3, f4         \n\t"
                            "fmul.ps   f2, f2, f5         \n\t"
                            "fmul.ps   f3, f3, f5         \n\t"
                            "fexp.ps   f2, f2             \n\t"
                            "fexp.ps   f3, f3             \n\t"
                            "fsw.ps    f2, 0(%[wt])       \n\t"
                            "fsw.ps    f3, 32(%[wt])      \n\t"
                            "fadd.ps   f2, f2, f3, rne    \n\t"
                            "fswizz.ps f3, f2, 0xB1       \n\t"
                            "fadd.ps   f2, f2, f3, rne    \n\t"
                            "fswizz.ps f3, f2, 0x4E       \n\t"
                            "fadd.ps   f2, f2, f3, rne    \n\t"
                            "fmvz.x.ps t0, f2, 4          \n\t"
                            "fbcx.ps   f3, t0             \n\t"
                            "fadd.ps   %[st], f2, f3, rne \n\t"
                            "mova.m.x  %[ms]              \n\t"
                            : [ms] "=&r"(_ms), [st] "=f"(S_tile)
                            : [pM] "r"(&M), [pL] "r"(&log2e), [sc] "r"(scores), [wt] "r"(weights)
                            : "f2", "f3", "f4", "f5", "t0", "memory");
                        S += S_tile;
                    }

                    // Phase B: weights @ V via TensorFMA16A32
                    {
                        // B1: convert weights F32 → F16
                        convert_q_row_f32_to_f16(w_f16_buf, weights, TILE_KV);

                        FENCE;
                        flush_to_l2(w_f16_buf, 1, 64);
                        WAIT_CACHEOPS;

                        // Issue weights load (wait_id=0) and the first V chunk
                        // load (wait_id=1) concurrently. Weights comes from
                        // L2 SCP (just flushed); V[0] comes from DRAM via
                        // INTERLEAVE16 — running them in parallel hides the
                        // shorter load behind the longer one. For partial
                        // tiles, V is software-packed below — we only kick
                        // off the early V load on the full-tile fast path.
                        tensor_load(false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0, (uint64_t) w_f16_buf, 0, 0, 64, 0);

                        const int       v_full_tile = (kv_count == TILE_KV);
                        const uintptr_t v_base      = (uintptr_t) v_head + kv_base * v->nb[1];
                        const uint64_t  nb1_v       = (uint64_t) v->nb[1];
                        uint64_t        b_cur       = 8;

                        if (v_full_tile) {
                            tensor_load(false, false, b_cur, TENSOR_LOAD_INTERLEAVE16, 0, (uint64_t) v_base, 0, 7,
                                        nb1_v, 1);
                        }

                        tensor_wait(TENSOR_LOAD_WAIT_0);      // weights in A_L1
                        if (v_full_tile) {
                            tensor_wait(TENSOR_LOAD_WAIT_1);  // V[0] in b_cur
                        }

                        // B2: process dv in chunks of 16
                        if (v_full_tile) {
                            for (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                const uint64_t b_nxt = b_cur ^ 24;

                                if (dv_off + 16 < dv) {
                                    tensor_load(false, false, b_nxt, TENSOR_LOAD_INTERLEAVE16, 0,
                                                (uint64_t) (v_base + (dv_off + 16) * 2), 0, 7, nb1_v, 1);
                                }

                                tensor_fma(false, 3, 0, 7, 0, false, false, false, false, b_cur, A_L1_START,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ volatile("" ::: "f0", "f1");
                                {
                                    unsigned long _ms;
                                    __asm__ volatile(
                                        "mova.x.m  %[ms]            \n\t"
                                        "mov.m.x   m0, x0, 0xFF     \n\t"
                                        "flw.ps    f2, 0(%[pa])     \n\t"
                                        "flw.ps    f3, 32(%[pa])    \n\t"
                                        "fadd.ps   f0, f0, f2       \n\t"
                                        "fadd.ps   f1, f1, f3       \n\t"
                                        "fsw.ps    f0, 0(%[pa])     \n\t"
                                        "fsw.ps    f1, 32(%[pa])    \n\t"
                                        "mova.m.x  %[ms]            \n\t"
                                        : [ms] "=&r"(_ms)
                                        : [pa] "r"(acc + dv_off)
                                        : "f0", "f1", "f2", "f3", "memory");
                                }

                                if (dv_off + 16 < dv) {
                                    tensor_wait(TENSOR_LOAD_WAIT_1);
                                    b_cur = b_nxt;
                                }
                            }
                        } else {
                            // Partial tile: software pack, no pipeline
                            for (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                pack_v_interleaved(vpanel_buf, v_head, kv_base, dv_off, kv_count, v->nb[1]);
                                FENCE;
                                flush_to_l2(vpanel_buf, 8, 64);
                                WAIT_CACHEOPS;
                                tensor_load(false, false, B_L1_START, TENSOR_LOAD_PLAIN, 0, (uint64_t) vpanel_buf, 0, 7,
                                            64, 0);
                                tensor_wait(TENSOR_LOAD_WAIT_0);

                                tensor_fma(false, 3, 0, 7, 0, false, false, false, false, B_L1_START, A_L1_START,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ volatile("" ::: "f0", "f1");
                                {
                                    unsigned long _ms;
                                    __asm__ volatile(
                                        "mova.x.m  %[ms]            \n\t"
                                        "mov.m.x   m0, x0, 0xFF     \n\t"
                                        "flw.ps    f2, 0(%[pa])     \n\t"
                                        "flw.ps    f3, 32(%[pa])    \n\t"
                                        "fadd.ps   f0, f0, f2       \n\t"
                                        "fadd.ps   f1, f1, f3       \n\t"
                                        "fsw.ps    f0, 0(%[pa])     \n\t"
                                        "fsw.ps    f1, 32(%[pa])    \n\t"
                                        "mova.m.x  %[ms]            \n\t"
                                        : [ms] "=&r"(_ms)
                                        : [pa] "r"(acc + dv_off)
                                        : "f0", "f1", "f2", "f3", "memory");
                                }
                            }
                        }
                    }
                }
            }
        }

        // Finalize row
        //
        // k_splits == 1: this minion computed the full row. Normalize in
        //                place and store to DRAM.
        //
        // k_splits  > 1: this minion computed a KV slab. Publish the
        //                partial (M, S, acc) to L2 SCP, sync with the
        //                team, and let the k_split==0 member do the
        //                softmax combine and the final store. All tensor
        //                engine ops are complete before this block, so
        //                f0..f31 are free to use.
        if (k_splits > 1) {
            // Publish our partial.
            volatile float * my_stats = (volatile float *) et_shire_l2scp_local(scp_base + SCP_STATS_OFF);
            my_stats[0]               = M;
            my_stats[1]               = S;
            FENCE;
            evict_range_to_l2(acc, (int64_t) dv * (int64_t) sizeof(float));
            evict_to_l2((const void *) my_stats, 1, 64);
            WAIT_CACHEOPS;

            // A: team members have all written their partials.
            et_barrier(ET_BARRIER_SHIRE);

            if (k_split == 0) {
                // Online softmax merge: fold peers 1..k_splits-1 into our
                // own (M_running, S_running, acc). For each peer p:
                //   M_new    = max(M_running, M_p)
                //   α_own    = exp2((M_running - M_new) * log2e)
                //   α_p      = exp2((M_p     - M_new) * log2e)
                //   acc[d]   = α_own * acc[d] + α_p * peer_acc[d]
                //   S_running = α_own * S_running + α_p * S_p
                float       M_running = M;
                float       S_running = S;
                const float log2e     = 1.4426950408889634f;

                for (int64_t p = 1; p < k_splits; p++) {
                    uint64_t         peer_scp   = (local_tile_idx * k_splits + p) * SCP_PER_MINION;
                    volatile float * peer_stats = (volatile float *) et_shire_l2scp_local(peer_scp + SCP_STATS_OFF);
                    float *          peer_acc   = (float *) et_shire_l2scp_local(peer_scp + SCP_ACC_OFF);

                    // Drop stale L1D copies before reading peer's data.
                    evict_to_l2((const void *) peer_stats, 1, 64);
                    evict_range_to_l2(peer_acc, (int64_t) dv * (int64_t) sizeof(float));
                    WAIT_CACHEOPS;

                    const float M_p = peer_stats[0];
                    const float S_p = peer_stats[1];

                    const float M_new     = (M_p > M_running) ? M_p : M_running;
                    const float alpha_own = (M_running == ET_NEG_INF_F) ? 0.0f : et_exp2f((M_running - M_new) * log2e);
                    const float alpha_p   = (M_p == ET_NEG_INF_F) ? 0.0f : et_exp2f((M_p - M_new) * log2e);

                    merge_rescale_add_asm(acc, peer_acc, dv, alpha_own, alpha_p);

                    S_running = alpha_own * S_running + alpha_p * S_p;
                    M_running = M_new;
                }

                const float S_inv = (S_running == 0.0f) ? 0.0f : et_fdiv(1.0f, S_running);
                normalize_store_vec(out, acc, dv, S_inv, use_fast_store);
            }

            // B: reducer is done, team may reuse its acc/stats slabs.
            et_barrier(ET_BARRIER_SHIRE);
        } else {
            // k_splits == 1 fast path — this minion owns the full row.
            const float S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
            normalize_store_vec(out, acc, dv, S_inv, use_fast_store);
        }
    }

    FENCE;
    return 0;
}
