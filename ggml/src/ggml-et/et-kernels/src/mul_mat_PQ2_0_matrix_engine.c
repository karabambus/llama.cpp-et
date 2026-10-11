#include <etsoc/common/utils.h>
#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "tensor.h"
#include "quants.h"
#include "math_fp.h"

// PQ2_0 x F32 -> F32 MUL_MAT on the tensor (matrix) engine, TensorFMA32 (prefill: N >= 53).
// Same producer/consumer structure, tiling, reuse path and synchronisation as mul_mat_Q4_0_matrix_engine.c;
// only the panel dequantisation differs.
// Hart 1: dequantize PQ2_0 weights to FP32 into double-buffered L2 SCP.
// Hart 0: tensor engine compute (FMA, store).
//
// A PQ2_0 block holds 128 elements ({ fp16 d; uint8 qs[32] }, element j in byte j/4 at bit 2*(j%4), weight
// d*(q-1)); a panel covers BLOCK_K = 32 elements, i.e. one quarter of a block (8 code bytes).
//
// Two execution paths (selected at runtime by the ru_n wave-score search below,
// i.e. reuse_ok = ru_n >= 2 — NOT by N % TILE_N; both paths handle partial N-tiles):
//   * REUSE path: dequantize each weight K-window ONCE and reuse it across ru_n consecutive N-tiles;
//     partial C is round-tripped through an L2-SCP scratch between K-windows.
//   * ORIGINAL path: one output tile at a time, no reuse.

#define NUM_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32

#define TILE_M  16
#define TILE_N  16
#define BLOCK_K 32      // elements per dequant panel (a quarter of a 128-element PQ2_0 block)
#define FMA_K   16      // tensor FMA k-width for FP32 (a_num_cols = FMA_K-1)

// --- Reuse knobs ----------------------------------------------------------
// REUSE_MAX caps the L2-SCP C-scratch footprint; the actual reuse factor is
// chosen at runtime (see ru_n) as the largest value that still keeps the whole
// machine busy. KWIN is the dequant-cache depth (K-blocks per window).
#ifndef REUSE_MAX
#define REUSE_MAX 15
#endif
#ifndef KWIN
#define KWIN    16      // K-blocks per dequant window (cache depth)
#endif

#define MACHINE_SLOTS (NUM_COMPUTE_SHIRES * MINIONS_PER_SHIRE)  // 1024

#define CACHEOP_MAX 0
#define REP_RATE    0

#define A_L1_START 0    // L1 SCP lines  0..15 for A (activations)
#define B_L1_START 16   // L1 SCP lines 16..31 for B (dequantized weights)
#define A1_L1_START 32  // L1 SCP lines 32..47: second A buffer (REUSE path double-buffers the activation loads)

// Single dequant panel: BLOCK_K k-lines x TILE_M m (FP32) = 32*64 = 2048 bytes,
// [k][m] order: panel[k*TILE_M + m].
#define SCP_PANEL_SIZE   (BLOCK_K * TILE_M * (uint64_t)sizeof(float))  // 2048

// L2 SCP layout per minion. The REUSE path needs the larger footprint, so the
// per-minion stride uses it for both paths (mutually exclusive at runtime).
//   [0 .. RU_BUF_BYTES)            cache buffer 0 (KWIN panels)
//   [RU_BUF_BYTES .. 2*..)         cache buffer 1 (KWIN panels)
//   [RU_CACHE_BYTES .. +R*1024)    REUSE_MAX C-scratch tiles (16 rows*64B each)
//   ready_ctr, consumed_ctr        sync counters
// The ORIGINAL path reuses [0,2048) and [2048,4096) as its two panels and the
// same ready/consumed counters (which sit above the cache region).
#define RU_BUF_BYTES     (KWIN * SCP_PANEL_SIZE)
#define RU_CACHE_BYTES   (2 * RU_BUF_BYTES)
#define RU_CSCRATCH_BYTES (REUSE_MAX * 16 * 64ULL)
#define SCP_READY_OFF    (RU_CACHE_BYTES + RU_CSCRATCH_BYTES)
#define SCP_CONSUMED_OFF (SCP_READY_OFF + 64)
#define SCP_PER_MINION   (SCP_CONSUMED_OFF + 64)

// Dequantize 32 elements (panel kb_panel = quarter kb_panel % 4 of block kb_panel / 4) of TILE_M weight rows into
// the FP32 panel, written directly in TenB [k][m] order: panel[k*TILE_M + m].
//
// Per half-tile of 8 rows and per code byte b (elements 4b..4b+3): one byte gather across the 8 rows (offsets
// l * nb1_0), then for each 2-bit field a shift, a mask of 3, (q - 1) * d with the rows' own scales in the lanes, and one
// contiguous 32-byte store into panel line 4b + j, columns 8h..8h+7. Every panel line is written by two whole-half
// stores; a column-wise scatter down 8 lines per 8 values cost far more (the engine step is producer-bound, r276).
// The fp16 scale is converted in integer code (fcvt.ps.f16 flushes subnormal halves, KB 02 section 5.1).
static inline void __attribute__((always_inline))
dequant_pq2_0_panel(float *panel, const char *src0_batch,
                    int64_t mb, int64_t kb_panel, int64_t nb1_0) {
    int32_t  __attribute__((aligned(32))) row_off[8];
    uint32_t __attribute__((aligned(32))) scale_bits[8];
    for (int l = 0; l < 8; ++l) {
        row_off[l] = (int32_t) (l * nb1_0);
    }

    unsigned long old_mask;
    __asm__ volatile(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"   // all 8 lanes active
        : [ms] "=&r"(old_mask)
    );

    for (int h = 0; h < 2; ++h) {
        const block_pq2_0 *blk0 = (const block_pq2_0 *)(src0_batch + (mb + 8 * h) * nb1_0) + (kb_panel >> 2);
        for (int l = 0; l < 8; ++l) {
            scale_bits[l] = fp16_to_fp32_bits(((const block_pq2_0 *)((const char *) blk0 + l * nb1_0))->d);
        }
        const uint8_t *qs  = blk0->qs + (kb_panel & 3) * 8;
        char          *col = (char *) panel + h * 32;
        for (int b = 0; b < 8; ++b) {
            __asm__ volatile(
                "flw.ps      f1, (%[ro])    \n\t"   // f1 = row offsets l * nb1_0
                "flw.ps      f3, (%[sc])    \n\t"   // f3 = the 8 rows' scales
                "fgb.ps      f4, f1(%[q])   \n\t"   // code byte b of each row (4 elements)

                "fandi.pi    f5, f4, 3      \n\t"   // element 4b
                "faddi.pi    f5, f5, -1     \n\t"
                "fcvt.ps.pw  f5, f5, rne    \n\t"
                "fmul.ps     f5, f5, f3     \n\t"
                "fsw.ps      f5, 0(%[c])    \n\t"

                "fsrli.pi    f5, f4, 2      \n\t"   // element 4b+1
                "fandi.pi    f5, f5, 3      \n\t"
                "faddi.pi    f5, f5, -1     \n\t"
                "fcvt.ps.pw  f5, f5, rne    \n\t"
                "fmul.ps     f5, f5, f3     \n\t"
                "fsw.ps      f5, 64(%[c])   \n\t"

                "fsrli.pi    f5, f4, 4      \n\t"   // element 4b+2
                "fandi.pi    f5, f5, 3      \n\t"
                "faddi.pi    f5, f5, -1     \n\t"
                "fcvt.ps.pw  f5, f5, rne    \n\t"
                "fmul.ps     f5, f5, f3     \n\t"
                "fsw.ps      f5, 128(%[c])  \n\t"

                "fsrli.pi    f5, f4, 6      \n\t"   // element 4b+3
                "fandi.pi    f5, f5, 3      \n\t"
                "faddi.pi    f5, f5, -1     \n\t"
                "fcvt.ps.pw  f5, f5, rne    \n\t"
                "fmul.ps     f5, f5, f3     \n\t"
                "fsw.ps      f5, 192(%[c])  \n\t"
                :
                : [ro] "r"(row_off), [sc] "r"(scale_bits), [q] "r"(qs + b), [c] "r"(col + b * 4 * 64)
                : "f1", "f3", "f4", "f5", "memory"
            );
        }
    }

    __asm__ volatile("mova.m.x %0" :: "r"(old_mask));
}

// Spill / seed the FP32 C accumulator (16x16 tile in the vector register file,
// row n -> f2n[cols 0..7], f2n+1[cols 8..15]) to/from a 1 KB L2-SCP scratch.
// scratch layout: row n at byte offset n*64. Always moves all 16 rows; rows
// beyond a partial n_cur carry harmless garbage (never stored / recomputed).
#define C_ROW_PAIR_ST(n0, n1, base)                                              \
    __asm__ volatile("fsw.ps f" #n0 ", (%0)\n\t fsw.ps f" #n1 ", (%1)\n\t"       \
                     :: "r"((base)), "r"((base) + 32) : "memory")
#define C_ROW_PAIR_LD(n0, n1, base)                                              \
    __asm__ volatile("flw.ps f" #n0 ", (%0)\n\t flw.ps f" #n1 ", (%1)\n\t"       \
                     :: "r"((base)), "r"((base) + 32) : "f" #n0, "f" #n1)

static inline void __attribute__((always_inline))
c_spill(char *s) {
    C_ROW_PAIR_ST(0,  1,  s + 0  * 64); C_ROW_PAIR_ST(2,  3,  s + 1  * 64);
    C_ROW_PAIR_ST(4,  5,  s + 2  * 64); C_ROW_PAIR_ST(6,  7,  s + 3  * 64);
    C_ROW_PAIR_ST(8,  9,  s + 4  * 64); C_ROW_PAIR_ST(10, 11, s + 5  * 64);
    C_ROW_PAIR_ST(12, 13, s + 6  * 64); C_ROW_PAIR_ST(14, 15, s + 7  * 64);
    C_ROW_PAIR_ST(16, 17, s + 8  * 64); C_ROW_PAIR_ST(18, 19, s + 9  * 64);
    C_ROW_PAIR_ST(20, 21, s + 10 * 64); C_ROW_PAIR_ST(22, 23, s + 11 * 64);
    C_ROW_PAIR_ST(24, 25, s + 12 * 64); C_ROW_PAIR_ST(26, 27, s + 13 * 64);
    C_ROW_PAIR_ST(28, 29, s + 14 * 64); C_ROW_PAIR_ST(30, 31, s + 15 * 64);
}

static inline void __attribute__((always_inline))
c_seed(char *s) {
    C_ROW_PAIR_LD(0,  1,  s + 0  * 64); C_ROW_PAIR_LD(2,  3,  s + 1  * 64);
    C_ROW_PAIR_LD(4,  5,  s + 2  * 64); C_ROW_PAIR_LD(6,  7,  s + 3  * 64);
    C_ROW_PAIR_LD(8,  9,  s + 4  * 64); C_ROW_PAIR_LD(10, 11, s + 5  * 64);
    C_ROW_PAIR_LD(12, 13, s + 6  * 64); C_ROW_PAIR_LD(14, 15, s + 7  * 64);
    C_ROW_PAIR_LD(16, 17, s + 8  * 64); C_ROW_PAIR_LD(18, 19, s + 9  * 64);
    C_ROW_PAIR_LD(20, 21, s + 10 * 64); C_ROW_PAIR_LD(22, 23, s + 11 * 64);
    C_ROW_PAIR_LD(24, 25, s + 12 * 64); C_ROW_PAIR_LD(26, 27, s + 13 * 64);
    C_ROW_PAIR_LD(28, 29, s + 14 * 64); C_ROW_PAIR_LD(30, 31, s + 15 * 64);
}

int entry_point(struct ggml_et_binary_params *params, void *env) {
    (void) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    if (shire_id >= NUM_COMPUTE_SHIRES) return 0;

    const int is_hart1 = hart_id & 1;
    uint64_t local_minion = (hart_id >> 1) & 0x1F;

    // Dimensions (both harts need these for tile assignment)
    const int64_t K = params->src0.ne[0];
    const int64_t M = params->src0.ne[1];
    const int64_t N = params->src1.ne[1];

    if ((M % TILE_M) != 0)   return 0;
    if ((K % QK_PQ2_0) != 0) return 0;

    const int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    const int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    const int64_t nb1_0 = params->src0.nb[1];
    const int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];

    const int64_t nb1_1 = params->src1.nb[1];
    const int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];

    const int64_t nb1_d = params->dst.nb[1];
    const int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    const char *src0_base = (const char *) params->src0.data;
    const char *src1_base = (const char *) params->src1.data;
    char       *dst_base  = (char *) params->dst.data;

    const int64_t m_tiles = M / TILE_M;
    const int64_t n_tiles = (N + TILE_N - 1) / TILE_N;
    const int64_t batch_count = ne2_1 * ne3_1;

    const int64_t r2 = ne2_1 / ne2_0;
    const int64_t r3 = ne3_1 / ne3_0;

    const int64_t k_steps = K / BLOCK_K;        // number of 32-element panels

    const int64_t tiles_per_shire = MINIONS_PER_SHIRE;
    const int64_t local_tile_idx  = local_minion;
    const int64_t tiles_stride    = (int64_t) NUM_COMPUTE_SHIRES * tiles_per_shire;
    const int64_t my_start        = (int64_t) shire_id + local_tile_idx * NUM_COMPUTE_SHIRES;

    // L2 SCP pointers for this minion.
    const uint64_t scp_base = local_minion * SCP_PER_MINION;
    volatile uint32_t *ready_ctr =
        (volatile uint32_t *) et_shire_l2scp_local(scp_base + SCP_READY_OFF);
    volatile uint32_t *consumed_ctr =
        (volatile uint32_t *) et_shire_l2scp_local(scp_base + SCP_CONSUMED_OFF);

    // Calculate ru_n to perfectly minimize hardware waves while avoiding Consumer bottleneck.
    // The pipeline is perfectly balanced at r=8. Score = waves * max(8, r).
    // We find the r that minimizes Score.
    int64_t best_r = 1;
    int64_t min_score = INT64_MAX;
    int64_t max_search_r = REUSE_MAX;
    if (max_search_r > n_tiles) max_search_r = n_tiles;

    for (int64_t r = 1; r <= max_search_r; r++) {
        int64_t n_groups = (n_tiles + r - 1) / r;
        int64_t base_units = m_tiles * n_groups * batch_count;
        int64_t waves = (base_units + MACHINE_SLOTS - 1) / MACHINE_SLOTS;
        
        int64_t penalty = (r > 8) ? r : 8;
        int64_t score = waves * penalty;
        
        if (score < min_score) {
            min_score = score;
            best_r = r;
        }
    }
    int64_t ru_n = best_r;

    // Reuse pays only when it groups >=2 N-tiles; otherwise the windowing /
    // C round-trip is pure overhead, so use the one-tile-at-a-time path.
    const int reuse_ok = (ru_n >= 2);

    // =====================================================================
    // REUSE path: dequant each K-window once, reuse across ru_n N-tiles.
    // =====================================================================
    if (reuse_ok) {
        char *cache_buf[2] = {
            (char *) et_shire_l2scp_local(scp_base),
            (char *) et_shire_l2scp_local(scp_base + RU_BUF_BYTES),
        };
        char *cscratch = (char *) et_shire_l2scp_local(scp_base + RU_CACHE_BYTES);

        const int64_t n_groups   = (n_tiles + ru_n - 1) / ru_n;
        const int64_t units_pb   = m_tiles * n_groups;
        const int64_t base_units = units_pb * batch_count;
        const int64_t n_windows  = (k_steps + KWIN - 1) / KWIN;

        // ----- Hart 1: producer -----
        if (is_hart1) {
            scp_signal(ready_ctr, 0);
            scp_signal(consumed_ctr, 0);
            // W24: hart 0 must not read the counters before this reset (stale values from an earlier
            // launch would let it consume unfilled panels) — same barrier as mul_mat_Q8_0_matrix_engine.
            et_barrier(ET_BARRIER_MINION);
            uint32_t wid = 0;

            for (int64_t unit = my_start; unit < base_units; unit += tiles_stride) {
                const int64_t batch_idx    = unit / units_pb;
                const int64_t unit_in_b    = unit % units_pb;
                const int64_t mb_idx       = unit_in_b % m_tiles;

                const int64_t i3   = batch_idx / ne2_1;
                const int64_t i2   = batch_idx % ne2_1;
                const int64_t i2_0 = i2 / r2;
                const int64_t i3_0 = i3 / r3;

                const char *src0_batch = src0_base + i3_0 * nb3_0 + i2_0 * nb2_0;
                const int64_t mb = mb_idx * TILE_M;

                for (int64_t kw = 0; kw < n_windows; ++kw) {
                    const int buf = wid & 1;
                    if (wid >= 2) scp_wait(consumed_ctr, wid - 1);

                    const int64_t kb0 = kw * KWIN;
                    const int64_t kbn = (kb0 + KWIN <= k_steps) ? KWIN : (k_steps - kb0);

                    float *cf = (float *) cache_buf[buf];
                    for (int64_t i = 0; i < kbn; ++i) {
                        dequant_pq2_0_panel(cf + i * (SCP_PANEL_SIZE / 4),
                                           src0_batch, mb, kb0 + i, nb1_0);
                    }
                    FENCE;
                    flush_to_l2_multi(cache_buf[buf], kbn * BLOCK_K, 64);   // W24: FlushVA caps at 16 lines/op; a window is up to KWIN*32 lines
                    WAIT_CACHEOPS;

                    wid++;
                    scp_signal(ready_ctr, wid);
                }
            }
            FENCE;
            return 0;
        }

        // ----- Hart 0: consumer -----
        setup_cache_scp();
#if CACHEOP_MAX > 0 || REP_RATE > 0
        ucache_control(1, REP_RATE, CACHEOP_MAX);
#endif
        CLEAR_TENSOR_ERROR;
        et_barrier(ET_BARRIER_MINION);   // W24: pair with hart 1's counter reset
        evict_to_l2((const void *) ready_ctr, 1, 64);    WAIT_CACHEOPS;
        evict_to_l2((const void *) consumed_ctr, 1, 64); WAIT_CACHEOPS;

        uint32_t wid = 0;
        for (int64_t unit = my_start; unit < base_units; unit += tiles_stride) {
            const int64_t batch_idx = unit / units_pb;
            const int64_t unit_in_b = unit % units_pb;
            const int64_t g_idx     = unit_in_b / m_tiles;
            const int64_t mb_idx    = unit_in_b % m_tiles;

            const int64_t i3 = batch_idx / ne2_1;
            const int64_t i2 = batch_idx % ne2_1;

            const char *src1_batch = src1_base + i3 * nb3_1 + i2 * nb2_1;
            char       *dst_batch  = dst_base  + i3 * nb3_d + i2 * nb2_d;

            const int64_t mb        = mb_idx * TILE_M;
            const int64_t nb_base_t = g_idx * ru_n;                    // first N-tile
            int64_t r_count = n_tiles - nb_base_t;
            if (r_count > ru_n) r_count = ru_n;

            for (int64_t kw = 0; kw < n_windows; ++kw) {
                const int buf = wid & 1;
                wid++;
                scp_wait(ready_ctr, wid);

                const int64_t kb0 = kw * KWIN;
                const int64_t kbn = (kb0 + KWIN <= k_steps) ? KWIN : (k_steps - kb0);
                const int is_last = (kw == n_windows - 1);
                float *cf = (float *) cache_buf[buf];

                for (int64_t r = 0; r < r_count; ++r) {
                    const int64_t nb = (nb_base_t + r) * TILE_N;
                    const int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);
                    const int64_t arows_fma = (n_cur == 4) ? 4 : (n_cur - 1);

                    char *cs = cscratch + r * (16 * 64);

                    if (kw > 0) c_seed(cs);
                    int first = (kw == 0) ? 1 : 0;

                    if (n_cur == 4) {
                        static const float __attribute__((aligned(64))) zero_line[16] = {0};
                        tensor_load(false, false, A_L1_START + 4, TENSOR_LOAD_PLAIN, 0,
                                    (uint64_t) zero_line, 0, 0, 64, 0);
                        tensor_wait(TENSOR_LOAD_WAIT_0);
                        tensor_load(false, false, A1_L1_START + 4, TENSOR_LOAD_PLAIN, 0,
                                    (uint64_t) zero_line, 0, 0, 64, 0);
                        tensor_wait(TENSOR_LOAD_WAIT_0);
                    }

                    // Activation loads are double-buffered: step s (K-block kb0 + s/2, half s%2) reads A from buffer
                    // s%2 while the load for step s+1 fills the other one. All A loads use id 0 (one in flight; the
                    // TenB weight load uses id 1), and a buffer is refilled only after the FMA that read it has
                    // completed. FMA order is unchanged, so results are bit-identical to the single-buffered loop.
                    const char *a_col = src1_batch + nb * nb1_1;
                    const int64_t n_steps = 2 * kbn;
                    tensor_load(false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0,
                                (uint64_t)(a_col + kb0 * BLOCK_K * (int64_t) sizeof(float)),
                                0, n_cur - 1, (uint64_t) nb1_1, 0);
                    for (int64_t st = 0; st < n_steps; ++st) {
                        tensor_wait(TENSOR_LOAD_WAIT_0);
                        if (st + 1 < n_steps) {
                            const int64_t k_next = kb0 * BLOCK_K + (st + 1) * FMA_K;
                            tensor_load(false, false, ((st + 1) & 1) ? A1_L1_START : A_L1_START,
                                        TENSOR_LOAD_PLAIN, 0,
                                        (uint64_t)(a_col + k_next * (int64_t) sizeof(float)),
                                        0, n_cur - 1, (uint64_t) nb1_1, 0);
                        }

                        tensor_load_setup_b(
                            false,
                            (uint64_t)(cf + (st >> 1) * (SCP_PANEL_SIZE / 4) + (st & 1) * FMA_K * TILE_M),
                            FMA_K - 1, 64, 1);

                        tensor_fma(
                            false, 3, arows_fma, FMA_K - 1, 0,
                            false, false, false, true,
                            B_L1_START, (st & 1) ? A1_L1_START : A_L1_START, TENSOR_FMA_OP_FP32, first);
                        tensor_wait(TENSOR_FMA_WAIT);
                        first = 0;
                    }

                    if (is_last) {
                        tensor_store(
                            0, 0, 3, n_cur - 1,
                            (uint64_t)(dst_batch + nb * nb1_d + mb * (int64_t) sizeof(float)),
                            0, (uint64_t) nb1_d);
                        tensor_wait(TENSOR_STORE_WAIT);
                    } else {
                        c_spill(cs);
                    }
                }
                scp_signal(consumed_ctr, wid);
            }
        }
        FENCE;
        return 0;
    }

    // =====================================================================
    // ORIGINAL path (ru_n < 2): one output tile at a time, no reuse.
    // =====================================================================
    const int64_t base_tiles = m_tiles * n_tiles * batch_count;
    float *scp_panel[2] = {
        (float *) et_shire_l2scp_local(scp_base),
        (float *) et_shire_l2scp_local(scp_base + SCP_PANEL_SIZE),
    };

    if (is_hart1) {
        scp_signal(ready_ctr, 0);
        scp_signal(consumed_ctr, 0);
        et_barrier(ET_BARRIER_MINION);   // W24: see REUSE path
        uint32_t chunk_id = 0;

        for (int64_t tile = my_start; tile < base_tiles; tile += tiles_stride) {
            const int64_t tiles_per_batch = m_tiles * n_tiles;
            const int64_t batch_idx       = tile / tiles_per_batch;
            const int64_t tile_in_batch   = tile % tiles_per_batch;
            const int64_t mb_idx          = tile_in_batch % m_tiles;

            const int64_t i3   = batch_idx / ne2_1;
            const int64_t i2   = batch_idx % ne2_1;
            const int64_t i2_0 = i2 / r2;
            const int64_t i3_0 = i3 / r3;

            const char *src0_batch = src0_base + i3_0 * nb3_0 + i2_0 * nb2_0;
            const int64_t mb = mb_idx * TILE_M;

            for (int64_t kb = 0; kb < k_steps; ++kb) {
                int buf = chunk_id & 1;
                if (chunk_id >= 2) scp_wait(consumed_ctr, chunk_id - 1);

                dequant_pq2_0_panel(scp_panel[buf], src0_batch, mb, kb, nb1_0);

                FENCE;
                flush_to_l2_multi(scp_panel[buf], BLOCK_K, 64);         // W24: 32 lines > the 16-line FlushVA cap
                WAIT_CACHEOPS;

                chunk_id++;
                scp_signal(ready_ctr, chunk_id);
            }
        }
        FENCE;
        return 0;
    }

    setup_cache_scp();
#if CACHEOP_MAX > 0 || REP_RATE > 0
    ucache_control(1, REP_RATE, CACHEOP_MAX);
#endif
    CLEAR_TENSOR_ERROR;
    et_barrier(ET_BARRIER_MINION);   // W24: pair with hart 1's counter reset
    evict_to_l2((const void *) ready_ctr, 1, 64);    WAIT_CACHEOPS;
    evict_to_l2((const void *) consumed_ctr, 1, 64); WAIT_CACHEOPS;

    uint32_t chunk_id = 0;
    for (int64_t tile = my_start; tile < base_tiles; tile += tiles_stride) {
        const int64_t tiles_per_batch = m_tiles * n_tiles;
        const int64_t batch_idx       = tile / tiles_per_batch;
        const int64_t tile_in_batch   = tile % tiles_per_batch;
        const int64_t nb_idx          = tile_in_batch / m_tiles;
        const int64_t mb_idx          = tile_in_batch % m_tiles;

        const int64_t i3 = batch_idx / ne2_1;
        const int64_t i2 = batch_idx % ne2_1;

        const char *src1_batch = src1_base + i3 * nb3_1 + i2 * nb2_1;
        char       *dst_batch  = dst_base  + i3 * nb3_d + i2 * nb2_d;

        const int64_t mb = mb_idx * TILE_M;
        const int64_t nb = nb_idx * TILE_N;
        const int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);
        const int64_t arows_fma = (n_cur == 4) ? 4 : (n_cur - 1);

        if (n_cur == 4) {
            static const float __attribute__((aligned(64))) zero_line[16] = {0};
            tensor_load(false, false, A_L1_START + 4, TENSOR_LOAD_PLAIN, 0,
                        (uint64_t) zero_line, 0, 0, 64, 0);
            tensor_wait(TENSOR_LOAD_WAIT_0);
        }

        int first = 1;
        for (int64_t kb = 0; kb < k_steps; ++kb) {
            int buf = chunk_id & 1;
            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            for (int half = 0; half < 2; ++half) {
                const int64_t k_elem = kb * BLOCK_K + half * FMA_K;
                tensor_load(
                    false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0,
                    (uint64_t)(src1_batch + nb * nb1_1 + k_elem * (int64_t) sizeof(float)),
                    0, n_cur - 1, (uint64_t) nb1_1, 0);
                tensor_wait(TENSOR_LOAD_WAIT_0);

                tensor_load_setup_b(
                    false,
                    (uint64_t)(scp_panel[buf] + (int64_t) half * FMA_K * TILE_M),
                    FMA_K - 1, 64, 1);

                tensor_fma(
                    false, 3, arows_fma, FMA_K - 1, 0,
                    false, false, false, true,
                    B_L1_START, A_L1_START, TENSOR_FMA_OP_FP32, first);
                tensor_wait(TENSOR_FMA_WAIT);
                first = 0;
            }

            scp_signal(consumed_ctr, chunk_id);
        }

        tensor_store(
            0, 0, 3, n_cur - 1,
            (uint64_t)(dst_batch + nb * nb1_d + mb * (int64_t) sizeof(float)),
            0, (uint64_t) nb1_d);
        tensor_wait(TENSOR_STORE_WAIT);
    }

    FENCE;
    return 0;
}
