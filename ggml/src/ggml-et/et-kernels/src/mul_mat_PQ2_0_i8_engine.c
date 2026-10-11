#include <etsoc/common/utils.h>
#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "tensor.h"
#include "quants.h"
#include "math_fp.h"

// PQ2_0 x Q8_K-quantised activations -> F32 MUL_MAT on the tensor engine's int8 path (prefill; PP100_PLAN.md F2).
//
// The CPU computes PQ2_0 against Q8_K activations (ggml-cpu.c vec_dot_type); quantize_q8k_f32's engine mode writes the
// same int8 values as rows of a8 and their per-256 scales as lines of sc, so the integer products here are the CPU's.
// Per 128-K group (one PQ2_0 block, one fp16 scale per weight row) the consumer runs two TenB-fed int8 FMAs (64 K each)
// into TenC, drains TenC on the second (tenc_loc), applies the rank-1 dequantisation in one tensor_quant
// (INT32_TO_FP32 -> FP32_MUL_COL token scale -> FP32_MUL_ROW weight scale) and adds the group into a running sum.
// Tiles are 8 tokens x 16 weight rows: the drain and the quant fill only f0-f15, so the running sum stays in f16-f31
// (P1, r308c: 670 cycles per 8-token group against 2,621 for a 16-token tile whose sum must round-trip memory).
//
// Hart 1 (producer) expands each group of a 16-row m-tile into a TenB panel in the L2 scratchpad: line c (code byte c
// of the block, elements 4c..4c+3 = one int8 quartet) holds the 16 rows' expanded words, one 1 KB table lookup per
// code byte (no conversion, no multiply), plus one line with the 16 fp32 weight scales. Hart 0 (consumer) streams the
// panels of a K-window through r tiles of 8 tokens. Handshake, counter reset, window double-buffering and the per-shire
// start tile follow mul_mat_PQ2_0_matrix_engine.c.

#define NUM_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32
#define MACHINE_SLOTS      (NUM_COMPUTE_SHIRES * MINIONS_PER_SHIRE)

#define TILE_M   16    // weight rows per tile (B columns)
#define TILE_T   8     // tokens per tile (A rows)
#define GK       128   // K per group = one PQ2_0 block
#define QK8      256   // K per activation scale
#define KWIN     8     // groups per window
#ifndef REUSE_MAX
#define REUSE_MAX 32
#endif

#define A0_L1  0       // A buffers (8 lines each): pass 0 / pass 1 of a group
#define A1_L1  32
#define SC_L1  16      // token-scale line (MUL_COL), weight-scale line at SC_L1 + 1 (MUL_ROW)

#define PANEL_BYTES   (32 * 64)                         // one group: 32 quartet lines x 16 words
#define WIN_PANELS    (KWIN * PANEL_BYTES)
#define WIN_BYTES     (WIN_PANELS + KWIN * 64)           // panels, then one weight-scale line per group
#define CACHE_BYTES   (2 * WIN_BYTES)
#define CSCRATCH      (REUSE_MAX * TILE_T * 64ULL)       // running sums: 8 rows x 64 B per tile
#define READY_OFF     (CACHE_BYTES + CSCRATCH)
#define CONSUMED_OFF  (READY_OFF + 64)
#define SCP_PER_MINION (CONSUMED_OFF + 64)
_Static_assert(SCP_PER_MINION % 64 == 0, "per-minion L2-scratchpad slices start on cache lines");
_Static_assert(MINIONS_PER_SHIRE * SCP_PER_MINION <= 2621440, "L2 scratchpad per shire");

struct ggml_et_pq2_i8_params {
    struct ggml_tensor src0;  // PQ2_0 weights [K, M]
    struct ggml_tensor a8;    // I8 activations [K, N], row stride nb[1], rows 64-byte aligned
    struct ggml_tensor sc;    // F32 token scales: line (t8 * K/256 + kb) = scales of tokens 8*t8 .. +7 for 256-block kb
    struct ggml_tensor dst;   // F32 [M, N]
};

// Expand group g of weight rows mb .. mb+15 into the TenB panel (32 lines x 16 words) and its weight-scale line.
// A code byte b holds elements 4c .. 4c+3 in bits 2x; the TenB int8 quartet wants element 4c + x in byte x as q - 1.
// Per half of 8 rows and per code byte: one byte gather across the rows, then in packed-integer lanes
//   y = (b | b << 6 | b << 12 | b << 18) & 0x03030303   (each 2-bit field in its own byte, 0..3)
//   w = (y + 0x7f7f7f7f) ^ 0x80808080                   (per byte q - 1 = 0xff, 0x00, 0x01, 0x02; no carry crosses a byte)
// and one 32-byte store. No table: in scratchpad mode the minion's L1 is too small to keep one resident.
static inline void __attribute__((always_inline))
i8_panel(uint32_t *panel, uint32_t *wscale, const char *src0_batch, int64_t mb, int64_t g, int64_t nb1_0,
         const int32_t *row_off) {
    unsigned long old_mask;
    __asm__ volatile("mova.x.m %0\n\tmov.m.x m0, x0, 0xFF" : "=&r"(old_mask));
    __asm__ volatile("flw.ps  f1, (%[ro])\n\t"
                     "fbcx.ps f6, %[c3]\n\t"
                     "fbcx.ps f7, %[c7f]\n\t"
                     "fbcx.ps f8, %[c80]\n"
                     :
                     : [ro] "r"(row_off), [c3] "r"(0x03030303ul), [c7f] "r"(0x7f7f7f7ful), [c80] "r"(0x80808080ul)
                     : "f1", "f6", "f7", "f8");
    for (int h = 0; h < 2; ++h) {
        const char *row0 = src0_batch + (mb + 8 * h) * nb1_0 + g * (int64_t) sizeof(block_pq2_0);
        for (int l = 0; l < 8; ++l) {
            wscale[8 * h + l] = fp16_to_fp32_bits(((const block_pq2_0 *) (row0 + l * nb1_0))->d);
        }
        const uint8_t *qs  = ((const block_pq2_0 *) row0)->qs;
        uint32_t      *col = panel + 8 * h;
        for (int c = 0; c < 32; ++c) {
            __asm__ volatile(
                "fgb.ps     f2, f1(%[q])  \n\t"   // code byte c of the 8 rows (sign-extended)
                "fandi.pi   f2, f2, 255   \n\t"
                "fslli.pi   f3, f2, 6     \n\t"
                "for.pi     f3, f3, f2    \n\t"
                "fslli.pi   f4, f2, 12    \n\t"
                "for.pi     f3, f3, f4    \n\t"
                "fslli.pi   f4, f2, 18    \n\t"
                "for.pi     f3, f3, f4    \n\t"
                "fand.pi    f3, f3, f6    \n\t"
                "fadd.pi    f3, f3, f7    \n\t"
                "fxor.pi    f3, f3, f8    \n\t"
                "fsw.ps     f3, 0(%[o])   \n\t"
                :
                : [q] "r"(qs + c), [o] "r"(col + c * 16)
                : "f2", "f3", "f4", "memory");
        }
    }
    __asm__ volatile("mova.m.x %0" : : "r"(old_mask));
}

#define RUN_ST(r, o) "fsw.ps f" #r ", " #o "*32(%[p])\n\t"
#define RUN_LD(r, o) "flw.ps f" #r ", " #o "*32(%[p])\n\t"

static inline void __attribute__((always_inline)) run_spill(char *p) {
    __asm__ volatile(RUN_ST(16, 0) RUN_ST(17, 1) RUN_ST(18, 2) RUN_ST(19, 3) RUN_ST(20, 4) RUN_ST(21, 5) RUN_ST(22, 6)
                     RUN_ST(23, 7) RUN_ST(24, 8) RUN_ST(25, 9) RUN_ST(26, 10) RUN_ST(27, 11) RUN_ST(28, 12)
                     RUN_ST(29, 13) RUN_ST(30, 14) RUN_ST(31, 15)
                     : : [p] "r"(p) : "memory");
}

static inline void __attribute__((always_inline)) run_seed(const char *p) {
    __asm__ volatile(RUN_LD(16, 0) RUN_LD(17, 1) RUN_LD(18, 2) RUN_LD(19, 3) RUN_LD(20, 4) RUN_LD(21, 5) RUN_LD(22, 6)
                     RUN_LD(23, 7) RUN_LD(24, 8) RUN_LD(25, 9) RUN_LD(26, 10) RUN_LD(27, 11) RUN_LD(28, 12)
                     RUN_LD(29, 13) RUN_LD(30, 14) RUN_LD(31, 15)
                     : : [p] "r"(p)
                     : "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28",
                       "f29", "f30", "f31", "memory");
}

static inline void __attribute__((always_inline)) run_zero(void) {
    __asm__ volatile("fbci.pi f16, 0\n\tfbci.pi f17, 0\n\tfbci.pi f18, 0\n\tfbci.pi f19, 0\n\t"
                     "fbci.pi f20, 0\n\tfbci.pi f21, 0\n\tfbci.pi f22, 0\n\tfbci.pi f23, 0\n\t"
                     "fbci.pi f24, 0\n\tfbci.pi f25, 0\n\tfbci.pi f26, 0\n\tfbci.pi f27, 0\n\t"
                     "fbci.pi f28, 0\n\tfbci.pi f29, 0\n\tfbci.pi f30, 0\n\tfbci.pi f31, 0\n"
                     ::: "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28",
                     "f29", "f30", "f31");
}

// running sum (f16-f31, rows 0-7) += this group's dequantised tile (f0-f15)
static inline void __attribute__((always_inline)) run_add(void) {
    __asm__ volatile("fadd.ps f16, f16, f0\n\tfadd.ps f17, f17, f1\n\tfadd.ps f18, f18, f2\n\t"
                     "fadd.ps f19, f19, f3\n\tfadd.ps f20, f20, f4\n\tfadd.ps f21, f21, f5\n\t"
                     "fadd.ps f22, f22, f6\n\tfadd.ps f23, f23, f7\n\tfadd.ps f24, f24, f8\n\t"
                     "fadd.ps f25, f25, f9\n\tfadd.ps f26, f26, f10\n\tfadd.ps f27, f27, f11\n\t"
                     "fadd.ps f28, f28, f12\n\tfadd.ps f29, f29, f13\n\tfadd.ps f30, f30, f14\n\t"
                     "fadd.ps f31, f31, f15\n"
                     ::: "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13",
                     "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26",
                     "f27", "f28", "f29", "f30", "f31", "memory");
}

int entry_point(struct ggml_et_pq2_i8_params *params, void *env) {
    (void) env;
    const uint64_t hart_id  = get_hart_id();
    const uint64_t shire_id = get_shire_id();
    if (shire_id >= NUM_COMPUTE_SHIRES) return 0;

    const int      is_hart1     = hart_id & 1;
    const uint64_t local_minion = (hart_id >> 1) & 0x1F;

    const int64_t K = params->src0.ne[0];
    const int64_t M = params->src0.ne[1];
    const int64_t N = params->a8.ne[1];
    // Uniform checks only (every hart returns together): the host routes exactly these shapes here.
    if (params->src0.type != GGML_TYPE_PQ2_0 || params->dst.type != GGML_TYPE_F32 || (M % TILE_M) != 0 ||
        (K % QK8) != 0 || params->a8.ne[0] != K || params->dst.ne[0] != M || params->dst.ne[1] != N ||
        (params->a8.nb[1] % 64) != 0 || ((uint64_t) params->a8.data % 64) != 0 || ((uint64_t) params->sc.data % 64) != 0) {
        return -1;
    }

    const int64_t nb1_0 = params->src0.nb[1];
    const int64_t nb1_a = params->a8.nb[1];
    const int64_t nb1_d = params->dst.nb[1];
    const char   *src0  = (const char *) params->src0.data;
    const char   *a8    = (const char *) params->a8.data;
    const char   *sc    = (const char *) params->sc.data;
    char         *dst   = (char *) params->dst.data;

    const int64_t m_tiles   = M / TILE_M;
    const int64_t t_tiles   = (N + TILE_T - 1) / TILE_T;
    const int64_t n_groups  = K / GK;
    const int64_t kb_per_t  = K / QK8;
    const int64_t n_windows = (n_groups + KWIN - 1) / KWIN;

    const int64_t tiles_stride = MACHINE_SLOTS;
    const int64_t my_start     = (int64_t) shire_id + (int64_t) local_minion * NUM_COMPUTE_SHIRES;

    const uint64_t     scp_base     = local_minion * SCP_PER_MINION;
    volatile uint32_t *ready_ctr    = (volatile uint32_t *) et_shire_l2scp_local(scp_base + READY_OFF);
    volatile uint32_t *consumed_ctr = (volatile uint32_t *) et_shire_l2scp_local(scp_base + CONSUMED_OFF);
    char *win[2] = { (char *) et_shire_l2scp_local(scp_base), (char *) et_shire_l2scp_local(scp_base + WIN_BYTES) };
    char *cscratch = (char *) et_shire_l2scp_local(scp_base + CACHE_BYTES);

    // Tiles per unit: the panels of a window are reused over r tiles of 8 tokens. Keep >= 16 tiles (128 tokens) per
    // panel where the machine stays busy; fewer waves first, then the smaller r.
    int64_t ru = 1, best = INT64_MAX;
    const int64_t max_r = t_tiles < REUSE_MAX ? t_tiles : REUSE_MAX;
    for (int64_t r = 1; r <= max_r; ++r) {
        const int64_t units = m_tiles * ((t_tiles + r - 1) / r);
        const int64_t waves = (units + MACHINE_SLOTS - 1) / MACHINE_SLOTS;
        const int64_t score = waves * (r > 16 ? r : 16);
        if (score < best) {
            best = score;
            ru   = r;
        }
    }
    const int64_t t_groups = (t_tiles + ru - 1) / ru;
    const int64_t units    = m_tiles * t_groups;

    // ----- hart 1: producer -----
    if (is_hart1) {
        scp_signal(ready_ctr, 0);
        scp_signal(consumed_ctr, 0);
        et_barrier(ET_BARRIER_MINION);  // W24: the consumer reads the counters only after this reset
        int32_t __attribute__((aligned(32))) row_off[8];
        for (int l = 0; l < 8; ++l) row_off[l] = (int32_t) (l * nb1_0);
        uint32_t wid = 0;
        for (int64_t unit = my_start; unit < units; unit += tiles_stride) {
            const int64_t mb = (unit % m_tiles) * TILE_M;
            for (int64_t w = 0; w < n_windows; ++w) {
                const int buf = wid & 1;
                if (wid >= 2) scp_wait(consumed_ctr, wid - 1);
                const int64_t g0  = w * KWIN;
                const int64_t kgn = (g0 + KWIN <= n_groups) ? KWIN : (n_groups - g0);
                for (int64_t i = 0; i < kgn; ++i) {
                    i8_panel((uint32_t *) (win[buf] + i * PANEL_BYTES), (uint32_t *) (win[buf] + WIN_PANELS + i * 64),
                             src0, mb, g0 + i, nb1_0, row_off);
                }
                FENCE;
                flush_to_l2_multi(win[buf], kgn * 32, 64);
                flush_to_l2_multi(win[buf] + WIN_PANELS, kgn, 64);
                WAIT_CACHEOPS;
                wid++;
                scp_signal(ready_ctr, wid);
            }
        }
        FENCE;
        return 0;
    }

    // ----- hart 0: consumer -----
    setup_cache_scp();
    CLEAR_TENSOR_ERROR;
    et_barrier(ET_BARRIER_MINION);
    evict_to_l2((const void *) ready_ctr, 1, 64);
    WAIT_CACHEOPS;
    evict_to_l2((const void *) consumed_ctr, 1, 64);
    WAIT_CACHEOPS;

    static const uint8_t __attribute__((aligned(64))) zero_line[64] = { 0 };
    const int64_t tile_rot = (int64_t) shire_id;
    uint32_t      wid      = 0;
    for (int64_t unit = my_start; unit < units; unit += tiles_stride) {
        const int64_t mb      = (unit % m_tiles) * TILE_M;
        const int64_t tg      = unit / m_tiles;
        const int64_t t_first = tg * ru;
        int64_t       r_count = t_tiles - t_first;
        if (r_count > ru) r_count = ru;

        for (int64_t w = 0; w < n_windows; ++w) {
            const int buf = wid & 1;
            wid++;
            scp_wait(ready_ctr, wid);
            const int64_t g0      = w * KWIN;
            const int64_t kgn     = (g0 + KWIN <= n_groups) ? KWIN : (n_groups - g0);
            const int     is_last = (w == n_windows - 1);
            const char   *pw      = win[buf];

            for (int64_t rr = 0; rr < r_count; ++rr) {
                const int64_t r     = (rr + tile_rot) % r_count;
                const int64_t t     = t_first + r;
                const int64_t tok0  = t * TILE_T;
                const int64_t n_cur = (tok0 + TILE_T <= N) ? TILE_T : (N - tok0);
                // four rows (AROWS == 3) are avoided as in the fp32 engines: a zeroed fifth row
                const int64_t arows = (n_cur == 4) ? 4 : (n_cur - 1);
                char         *cs    = cscratch + r * (TILE_T * 64);

                if (w > 0) {
                    run_seed(cs);
                } else {
                    run_zero();
                }
                if (n_cur == 4) {
                    tensor_load(false, false, A0_L1 + 4, TENSOR_LOAD_PLAIN, 0, (uint64_t) zero_line, 0, 0, 64, 0);
                    tensor_wait(TENSOR_LOAD_WAIT_0);
                    tensor_load(false, false, A1_L1 + 4, TENSOR_LOAD_PLAIN, 0, (uint64_t) zero_line, 0, 0, 64, 0);
                    tensor_wait(TENSOR_LOAD_WAIT_0);
                }

                const char *arow = a8 + tok0 * nb1_a;
                tensor_load(false, false, A0_L1, TENSOR_LOAD_PLAIN, 0, (uint64_t) (arow + g0 * GK), 0, n_cur - 1,
                            (uint64_t) nb1_a, 0);
                for (int64_t i = 0; i < kgn; ++i) {
                    const int64_t g = g0 + i;
                    tensor_wait(TENSOR_LOAD_WAIT_0);
                    tensor_load(false, false, A1_L1, TENSOR_LOAD_PLAIN, 0, (uint64_t) (arow + g * GK + 64), 0, n_cur - 1,
                                (uint64_t) nb1_a, 0);
                    tensor_load(false, false, SC_L1, TENSOR_LOAD_PLAIN, 0,
                                (uint64_t) (sc + (t * kb_per_t + g / 2) * 64), 0, 0, 64, 1);
                    tensor_load(false, false, SC_L1 + 1, TENSOR_LOAD_PLAIN, 0, (uint64_t) (pw + WIN_PANELS + i * 64), 0,
                                0, 64, 1);
                    tensor_load_setup_b(false, (uint64_t) (pw + i * PANEL_BYTES), 15, 64, 1);
                    tensor_fma(false, 3, arows, 15, 0, false, false, false, true, 16, A0_L1, 3, true);
                    tensor_wait(TENSOR_FMA_WAIT);
                    tensor_wait(TENSOR_LOAD_WAIT_0);
                    tensor_load_setup_b(false, (uint64_t) (pw + i * PANEL_BYTES + 16 * 64), 15, 64, 1);
                    tensor_fma(false, 3, arows, 15, 0, true, false, false, true, 16, A1_L1, 3, false);
                    tensor_wait(TENSOR_FMA_WAIT);
                    if (i + 1 < kgn) {
                        tensor_load(false, false, A0_L1, TENSOR_LOAD_PLAIN, 0, (uint64_t) (arow + (g + 1) * GK), 0,
                                    n_cur - 1, (uint64_t) nb1_a, 0);
                    }
                    tensor_wait(TENSOR_LOAD_WAIT_1);
                    tensor_quant(0, 3, arows, SC_L1, QUANT_LAST_TRANS, QUANT_LAST_TRANS, QUANT_LAST_TRANS,
                                 QUANT_LAST_TRANS, QUANT_LAST_TRANS, QUANT_LAST_TRANS, QUANT_LAST_TRANS,
                                 QUANT_FP32_MUL_ROW, QUANT_FP32_MUL_COL, QUANT_INT32_TO_FP32);
                    tensor_wait(TENSOR_QUANT_WAIT);
                    run_add();
                }

                if (is_last) {
                    tensor_store(0, 16, 3, n_cur - 1, (uint64_t) (dst + tok0 * nb1_d + mb * (int64_t) sizeof(float)), 0,
                                 (uint64_t) nb1_d);
                    tensor_wait(TENSOR_STORE_WAIT);
                } else {
                    run_spill(cs);
                }
            }
            scp_signal(consumed_ctr, wid);
        }
    }
    FENCE;
    return 0;
}
