//******************************************************************************
// MUL_MAT_ID kernel for Q4_0 experts (Mixture of Experts), expert-grouped (backlog 039)
//
// C[m, s, t] = sum_k A[k, m, ids[s, t]] * B[k, s % ne11, t]
//   A: Q4_0 [K, M, n_expert]   B: F32 [K, ne11, N]   ids: I32 [S, N]   C: F32 [M, S, N]
//
// v0 (mul_mat_id_Q4_0_v0.c) walked the (token, slot) pairs and re-streamed / re-unpacked
// the whole expert for every pair: at N=32 every expert is read n_e ~ 2 times, at N=128
// ~8 times. Here every hart first groups the S*N pairs by expert (counting sort in its
// private L2-SCP slice), then the work unit is (active expert, block of 16 rows): each
// 32-element weight block of a row pair is gathered, unpacked and pre-scaled ONCE and
// FMA'd against up to 4 tokens of that expert (2 rows x 4 tokens = 8 accumulators), so
// weight traffic and unpack work are per expert, not per pair. Outputs for a 16-row block
// are staged per token and written as one owned 64-byte line (fsq2) when the dst column
// is line-aligned, otherwise with per-element global atomics.
//
// Numerics: prod = (scale * q) * b is exact for the prescale (q in [-8,7], scale f16 -> f32),
// so the result differs from v0 only in the f32 summation order.
// Limits (else v0 runs): n_expert <= 512, S*N <= 4096 pairs. K % 32 == 0 always required.
//
// 039b: SCP-staged double buffer (overlap the DRAM->SCP expert-weight stream with compute; W25 found the stream
// itself DRAM-bound at 20-33 GB/s regardless of fetch instruction, so the only lever left is not serializing it
// with the ~1 ms/unit of FMA work). Hart 0 of each minion issues `tensor_load_l2scp` for both its own and its
// partner's next unit's rows (id 0 only - id 1's wait never returns on this firmware), waits, evicts the buffer's
// lines from its own L1 (the DMA bypasses L1 on write, so a stale cached copy from 2 iterations ago would
// otherwise survive undetected), then signals ready; both harts compute their own unit from their own SCP buffer
// while hart 0 issues the next pair's load. Falls back to the original direct-DRAM-read path (unchanged) when the
// double-buffered rows don't fit the per-hart SCP budget (large K) or a unit has no partner (odd tail).
//******************************************************************************

#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"
#include "quants.h"
#include "tensor.h"

#include <stdint.h>

#ifdef ET_UBERKERNEL
#    define ET_MMID_ENTRY_NAME mul_mat_id_Q4_0_entry
#else
#    define ET_MMID_ENTRY_NAME entry_point
#endif
#undef entry_point
#define entry_point mmid_q4_0_v0
#include "mul_mat_id_Q4_0_v0.c"
#undef entry_point

#define MMID_MAX_E     1024
#define MMID_MAX_PAIRS 4096
#define MMID_ROWS      16   // rows per work unit = one 64-byte dst line per token
#define MMID_T         4    // tokens per pass
// Per-hart SCP budget: 2.5 MiB/shire / 64 harts-per-shire = 40960 B, used exclusively by this hart (grouping
// scratch below, then - 039b - the SCP row-staging double buffer). 40960 * 64 == the whole shire scratchpad.
#define MMID_SCRATCH_B  40960
#define MMID_SCP_CTR_B  128  // 039b: 2 lines reserved for this hart's ready_ctr/consumed_ctr (1 line each)
#ifndef MMID_DIAG
#    define MMID_DIAG 0   // 0: production v1 (SCP staging only if MMID_SCP_ENABLE); 9/10/11 = 039b bring-up variants
                           // 1: no token loads/FMAs  2: no gathers/unpack  3: no B loads  4: grouping only  5: no K loop
                           // (diagnostics only)  9: 039b SCP staging, no overlap  10: 039b SCP staging, double-buffered
                           // 11: = 9 plus a per-hart trace record [tid, 2.0, wid, wait_fails, units, scp==dram matches,
                           //     kernel_env->shire_mask, is_hart0, num_threads, offs[E], pairs[0], active[0], n_active, rows_u,
                           //     params->shire_mask, hart_id] published into src1 (16 floats at src1[tid*16]) at the end -
                           //     mmid_check prints them; src1 is clobbered so values are garbage, the dst write map is intact
#endif
// 039b DISABLED 2026-08-26: the hart0/hart1 SCP handshake below deadlocked on first board test (mmid_check, gate
// shape, N=1) and the deadlock propagated into a wedged card that the normal DM_CMD_RESET_ETSOC recovery could not
// clear ("State not recoverable" from dev_mngt_service - needed root-level intervention). Root cause NOT found by
// static review (barrier arity, counter sequencing, and buffer indexing all check out on paper; the individual
// primitives - id-0 tensor_load_l2scp, chip-wide concurrent issuance - were each independently verified safe in
// Exp 039's one-shot bisection probes). Suspect: combining chip-wide concurrent DMA issuance with a *repeated,
// iterative* cross-hart software handshake (not the one-shot probes 039 ran) exposes a timing/hardware interaction
// only visible with board access. Do not flip this to 1 without a single-minion or single-shire reproduction first
// (narrow the blast radius before the next chip-wide run) - see experiments/039_expert_grouped_mmid/NOTES.md.
#ifndef MMID_SCP_ENABLE
#    define MMID_SCP_ENABLE 0  // 039b SCP-staged path: verified correct + hang-free chip-wide 2026-08-27 but slower than the direct path (NOTES); off
#endif

// ---- the K loop for (2 rows) x (4 tokens): ONE asm block, so no value lives in an f-register across C code
// (only f8/f9/f18-f27 survive compiler-generated code; accumulators and weights here span f0-f31 freely).
// Per 32-element block: both rows' scales (block_q4_0.d sits 2 bytes before .qs), 4 gathers, unpack + prescale into
// f20-f23 (row0: elems 0-7, 8-15, 16-23, 24-31) / f25-f28 (row1) - the nibble->element mapping of q4_dot_tile -
// then per token 4 activation loads + 8 fmadd into f0..f7 (token i: row0 -> f(2i), row1 -> f(2i+1)). After the loop
// each accumulator is reduced and stored as a scalar: token i row0 -> p_r + 64*i, row1 -> p_r1 + 64*i.
#define MMID_UNPACK_ROW(RAW0, RAW8, LO0, LO1, HI0, HI1, S)          \
    "fandi.pi    " LO0 ", " RAW0 ", 15\n"                          \
    "faddi.pi    " LO0 ", " LO0 ", -8\n"                           \
    "fcvt.ps.pw  " LO0 ", " LO0 ", rne\n"                          \
    "fmul.ps     " LO0 ", " LO0 ", " S "\n"                         \
    "fandi.pi    " LO1 ", " RAW8 ", 15\n"                          \
    "faddi.pi    " LO1 ", " LO1 ", -8\n"                           \
    "fcvt.ps.pw  " LO1 ", " LO1 ", rne\n"                          \
    "fmul.ps     " LO1 ", " LO1 ", " S "\n"                         \
    "fsrli.pi    " HI0 ", " RAW0 ", 4\n"                           \
    "fandi.pi    " HI0 ", " HI0 ", 15\n"                           \
    "faddi.pi    " HI0 ", " HI0 ", -8\n"                           \
    "fcvt.ps.pw  " HI0 ", " HI0 ", rne\n"                          \
    "fmul.ps     " HI0 ", " HI0 ", " S "\n"                         \
    "fsrli.pi    " HI1 ", " RAW8 ", 4\n"                           \
    "fandi.pi    " HI1 ", " HI1 ", 15\n"                           \
    "faddi.pi    " HI1 ", " HI1 ", -8\n"                           \
    "fcvt.ps.pw  " HI1 ", " HI1 ", rne\n"                          \
    "fmul.ps     " HI1 ", " HI1 ", " S "\n"

#if MMID_DIAG == 1
#    define MMID_TOKEN(ACC0, ACC1, BP) ""
#else
#if MMID_DIAG == 3
#    define MMID_TOKEN_LOADS(BP) ""
#else
#    define MMID_TOKEN_LOADS(BP)                                    \
    "flw.ps   f10, 0(" BP ")\n"                                    \
    "flw.ps   f13, 32(" BP ")\n"                                   \
    "flw.ps   f14, 64(" BP ")\n"                                   \
    "flw.ps   f15, 96(" BP ")\n"
#endif
#define MMID_TOKEN(ACC0, ACC1, BP)                                  \
    MMID_TOKEN_LOADS(BP)                                            \
    "fmadd.ps " ACC0 ", f20, f10, " ACC0 ", rne\n"                 \
    "fmadd.ps " ACC1 ", f25, f10, " ACC1 ", rne\n"                 \
    "fmadd.ps " ACC0 ", f21, f13, " ACC0 ", rne\n"                 \
    "fmadd.ps " ACC1 ", f26, f13, " ACC1 ", rne\n"                 \
    "fmadd.ps " ACC0 ", f22, f14, " ACC0 ", rne\n"                 \
    "fmadd.ps " ACC1 ", f27, f14, " ACC1 ", rne\n"                 \
    "fmadd.ps " ACC0 ", f23, f15, " ACC0 ", rne\n"                 \
    "fmadd.ps " ACC1 ", f28, f15, " ACC1 ", rne\n"
#endif

#define MMID_REDUCE_STORE(ACC, OFF, PTR)                            \
    "fswizz.ps f11, " ACC ", 0xB1\n"                               \
    "fadd.ps   f12, " ACC ", f11, rne\n"                           \
    "fswizz.ps f13, f12, 0x4E\n"                                   \
    "fadd.ps   f14, f12, f13, rne\n"                               \
    "fmvz.x.ps a4, f14, 4\n"                                       \
    "fbcx.ps   f15, a4\n"                                          \
    "fadd.ps   f14, f14, f15, rne\n"                               \
    "fsw       f14, " OFF "(" PTR ")\n"

static const int32_t mmid_gather[8] __attribute__((aligned(32))) = { 0, 1, 2, 3, 4, 5, 6, 7 };

// The K loop for (2 rows) x (T tokens), T in {1, 2, 4}: generated from one template so the per-token asm is only
// emitted for the tokens that exist (no dummy work for n_e < 4).
// row0/row1: first block's qs; b0..b3: token activation columns (32 floats per block); p_r/p_r1: out slots
#if MMID_DIAG == 2
#    define MMID_GATHER_UNPACK ""
#else
#    define MMID_GATHER_UNPACK                                                          \
        "fgb.ps      f11, f31(t0)\n"                                                   \
        "addi        a6, t0, 8\n"                                                      \
        "fgb.ps      f12, f31(a6)\n"                                                   \
        "fgb.ps      f16, f31(t1)\n"                                                   \
        "addi        a7, t1, 8\n"                                                      \
        "fgb.ps      f17, f31(a7)\n"                                                   \
        MMID_UNPACK_ROW("f11", "f12", "f20", "f21", "f22", "f23", "f24")                \
        MMID_UNPACK_ROW("f16", "f17", "f25", "f26", "f27", "f28", "f29")
#endif

#define MMID_KLOOP_FN(NAME, TOKENS, REDUCES)                                                                    \
    static inline void __attribute__((always_inline)) NAME(const uint8_t * row0, const uint8_t * row1,             \
                                                           const float * b0, const float * b1, const float * b2, \
                                                           const float * b3, int64_t k_blocks, float * p_r,      \
                                                           float * p_r1) {                                       \
        __asm__ volatile(                                                                                       \
            "mv   t0, %[q0]\n mv   t1, %[q1]\n"                                                                 \
            "mv   t2, %[b0]\n mv   t3, %[b1]\n mv   t4, %[b2]\n mv   t5, %[b3]\n"                                \
            "mv   t6, %[kb]\n"                                                                                  \
            "flw.ps  f31, %[g]\n"                                                                               \
            "fbci.pi f0, 0\n fbci.pi f1, 0\n fbci.pi f2, 0\n fbci.pi f3, 0\n"                                    \
            "fbci.pi f4, 0\n fbci.pi f5, 0\n fbci.pi f6, 0\n fbci.pi f7, 0\n"                                    \
            "1:\n"                                                                                              \
            "lhu         a4, -2(t0)\n"                                                                          \
            "fbcx.ps     f24, a4\n"                                                                             \
            "fcvt.ps.f16 f24, f24\n"                                                                            \
            "lhu         a5, -2(t1)\n"                                                                          \
            "fbcx.ps     f29, a5\n"                                                                             \
            "fcvt.ps.f16 f29, f29\n"                                                                            \
            MMID_GATHER_UNPACK                                                                                  \
            TOKENS                                                                                              \
            "addi t0, t0, 18\n addi t1, t1, 18\n"                                                               \
            "addi t2, t2, 128\n addi t3, t3, 128\n addi t4, t4, 128\n addi t5, t5, 128\n"                        \
            "addi t6, t6, -1\n"                                                                                 \
            "bnez t6, 1b\n"                                                                                     \
            REDUCES                                                                                             \
            :                                                                                                   \
            : [q0] "r"(row0), [q1] "r"(row1), [b0] "r"(b0), [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3),           \
              [kb] "r"(k_blocks), [pr] "r"(p_r), [pr1] "r"(p_r1), [g] "m"(*(const int32_t (*)[8]) mmid_gather)  \
            : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a4", "a5", "a6", "a7", "memory", "f0", "f1", "f2",     \
              "f3", "f4", "f5", "f6", "f7", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f20",       \
              "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f31");                             \
    }

MMID_KLOOP_FN(mmid_rows2_tok1, MMID_TOKEN("f0", "f1", "t2"),
              MMID_REDUCE_STORE("f0", "0", "%[pr]") MMID_REDUCE_STORE("f1", "0", "%[pr1]"))
MMID_KLOOP_FN(mmid_rows2_tok2, MMID_TOKEN("f0", "f1", "t2") MMID_TOKEN("f2", "f3", "t3"),
              MMID_REDUCE_STORE("f0", "0", "%[pr]") MMID_REDUCE_STORE("f1", "0", "%[pr1]")
              MMID_REDUCE_STORE("f2", "64", "%[pr]") MMID_REDUCE_STORE("f3", "64", "%[pr1]"))
MMID_KLOOP_FN(mmid_rows2_tok4,
              MMID_TOKEN("f0", "f1", "t2") MMID_TOKEN("f2", "f3", "t3") MMID_TOKEN("f4", "f5", "t4") MMID_TOKEN("f6", "f7", "t5"),
              MMID_REDUCE_STORE("f0", "0", "%[pr]") MMID_REDUCE_STORE("f1", "0", "%[pr1]")
              MMID_REDUCE_STORE("f2", "64", "%[pr]") MMID_REDUCE_STORE("f3", "64", "%[pr1]")
              MMID_REDUCE_STORE("f4", "128", "%[pr]") MMID_REDUCE_STORE("f5", "128", "%[pr1]")
              MMID_REDUCE_STORE("f6", "192", "%[pr]") MMID_REDUCE_STORE("f7", "192", "%[pr1]"))

// L2 prefetch of a whole Q4_0 row (nlines <= 16 per CSR write)
static inline void __attribute__((always_inline)) mmid_prefetch_row(const void * row, int64_t bytes) {
    int64_t lines = (bytes + 63) / 64;
    const char * p = (const char *) row;
    while (lines > 0) {
        const int64_t n = lines > 16 ? 16 : lines;
        l2_prefetch(p, (uint64_t) n, 64);
        p += n * 64;
        lines -= n;
    }
}

static inline int64_t m0_of_unit(int thread_id, int64_t row_blocks, int64_t rows_u, const int32_t * active, size_t nb01, size_t nb02) {
    const int64_t u = thread_id;
    return (int64_t) active[u / row_blocks] * (int64_t) nb02 + (u % row_blocks) * rows_u * (int64_t) nb01;
}

static inline void mmid_store_line(float * dst, const float * src) {
    __asm__ volatile(
        "flq2 f0, 0(%[src]) \n\t"
        "flq2 f1, 32(%[src]) \n\t"
        "fsq2 f0, 0(%[dst]) \n\t"
        "fsq2 f1, 32(%[dst]) \n\t"
        :
        : [src] "r"(src), [dst] "r"(dst)
        : "f0", "f1", "memory");
}

// 039b: line index (within THIS shire's 2.5 MiB SCP) for a byte offset inside hart `his`'s MMID_SCRATCH_B slice.
static inline uint64_t mmid_scp_line(uint64_t his, uint64_t byte_off) {
    return (his * (uint64_t) MMID_SCRATCH_B + byte_off) / 64;
}

// 039b: stage `bytes` (multiple of 64) from DRAM `src` into this shire's SCP starting at line `dst_line0`, <=16
// lines per tensor_load_l2scp (id 0 only: id 1's wait never returns on this firmware; never a second load in
// flight before the first is retired). Retirement is either the hardware wait (MMID_SCP_HW_WAIT=1, the
// single-outstanding-load pattern verified in 039 NOTES) or - bring-up default - a bounded evict-poll of every
// staged line against a sentinel (the hang-proof MMID_DIAG==7 probe technique): a stalled load then costs
// MMID_SCP_WAIT_SPINS polls and a wrong answer instead of an unbounded hardware wait that can wedge the card
// (the chip-wide 039b deadlock of 2026-08-26 is unexplained; this is the only unbounded wait left in the path).
// Returns the number of lines whose poll gave up (0 = everything landed).
#ifndef MMID_SCP_HW_WAIT
#    define MMID_SCP_HW_WAIT 0
#endif
#ifndef MMID_SCP_WAIT_SPINS
#    define MMID_SCP_WAIT_SPINS 2000000  // bound for every software poll in this kernel (scp_wait + line polls)
#endif
#define MMID_SCP_SENT0  0xC0FFEE00u
#define MMID_SCP_SENT15 0xC0FFEE0Fu
static inline int mmid_load_rows_scp(uint64_t dst_line0, const void * src, int64_t bytes) {
    const char * p    = (const char *) src;
    uint64_t     dl    = dst_line0;
    int64_t      left  = (bytes + 63) / 64;
    int          fails = 0;
    while (left > 0) {
        const int64_t n = left > 16 ? 16 : left;
#if !MMID_SCP_HW_WAIT
        // sentinel word 0 and 15 of every destination line, pushed into the L2 SCP line itself (L1 copy dropped)
        // BEFORE the load is issued, so the write-back cannot race the tensor unit's write of the same line
        volatile uint32_t * l0 = (volatile uint32_t *) et_shire_l2scp_local(dl * 64);
        for (int64_t i = 0; i < n; ++i) {
            l0[i * 16 + 0]  = MMID_SCP_SENT0;
            l0[i * 16 + 15] = MMID_SCP_SENT15;
        }
        FENCE;
        evict_to_l2((const void *) l0, (uint64_t) n, 64);
        WAIT_CACHEOPS;
#endif
        et_tensor_load_l2scp_conf_t c = { false, dl, (uint64_t) p, (uint64_t) (n - 1), 64, 0 };
        et_tensor_load_l2scp(&c);
#if MMID_SCP_HW_WAIT
        WAIT_TENSOR_LOAD_L2_0;
#else
        for (int64_t i = 0; i < n; ++i) {  // tensor writes bypass L1: evict before every poll read
            volatile uint32_t * l = l0 + i * 16;
            int32_t spin = 0;
            for (; spin < MMID_SCP_WAIT_SPINS; ++spin) {
                evict_to_l2((const void *) l, 1, 64);
                WAIT_CACHEOPS;
                if (l[0] != MMID_SCP_SENT0 || l[15] != MMID_SCP_SENT15) { break; }
            }
            fails += (spin == MMID_SCP_WAIT_SPINS);
        }
#endif
        dl += n;
        p += n * 64;
        left -= n;
    }
    return fails;
}

// 039b: invalidate `bytes` (multiple of 64) worth of this hart's own L1 lines at `addr` so a subsequent plain load
// re-fetches from L2 SCP - required because tensor_load_l2scp writes bypass L1, and a hart's own earlier read of
// the same (reused) buffer slot would otherwise survive as a stale L1 copy.
static inline void mmid_evict_rows(const void * addr, int64_t bytes) {
    FENCE;
    const char * p    = (const char *) addr;
    int64_t      left = (bytes + 63) / 64;
    while (left > 0) {
        const int64_t n = left > 16 ? 16 : left;
        evict_to_l2(p, (uint64_t) n, 64);
        p += n * 64;
        left -= n;
    }
    WAIT_CACHEOPS;
}

// 039b safety net: bounded version of platform.h's scp_wait. The unbounded original deadlocked this kernel on its
// first board test and the hang propagated into a wedged card (see 039b incident writeup, NOTES.md) - whatever the
// exact bug is, this hart must never park in a spin loop that can only be broken by the host's launch-level
// timeout (which failed to save the card last time). Giving up after MMID_SCP_WAIT_SPINS turns a stuck handshake
// into a wrong-but-bounded answer (caught by the correctness check) instead of an unbounded device hang.
static inline int mmid_scp_wait_bounded(volatile uint32_t * flag, uint32_t expected) {
    for (int32_t spin = 0; spin < MMID_SCP_WAIT_SPINS; ++spin) {
        evict_to_l2((const void *) flag, 1, 64);
        WAIT_CACHEOPS;
        if (*flag >= expected) {
            return 1;
        }
    }
    return 0;  // gave up: caller proceeds anyway rather than hang
}

// 039b: compute one unit's row-pairs from SCP-resident, tightly-packed rows (buf + r*row_bytes = row r's
// block_q4_0 data). Same arithmetic as the direct-DRAM-read path's inner loop, just sourced from the SCP buffer
// the caller already staged and evicted from L1.
static inline void mmid_compute_unit_scp(const char * buf, int64_t row_bytes, int64_t K_blocks, int64_t nrows,
                                          int64_t m0, int32_t p0, int32_t p1, const int32_t * pairs,
                                          const char * b_base, size_t nb11, size_t nb12, char * d_base, size_t nbd1,
                                          size_t nbd2, int64_t ne11, int line_ok, float out_blk[MMID_T][MMID_ROWS]) {
    for (int32_t pg = p0; pg < p1; pg += MMID_T) {
        const int32_t T = (p1 - pg < MMID_T) ? (p1 - pg) : MMID_T;
        const float * bcol[MMID_T];
        float *       dcol[MMID_T];
        for (int32_t i = 0; i < MMID_T; ++i) {
            const int32_t pi = pg + (i < T ? i : 0);
            const int64_t t  = pairs[pi] >> 8, s = pairs[pi] & 255;
            bcol[i] = (const float *) (b_base + (s % ne11) * nb11 + t * nb12);
            dcol[i] = (float *) (d_base + s * nbd1 + t * nbd2) + m0;
        }
        for (int64_t r = 0; r < nrows; r += 2) {
            const int64_t      r1   = (r + 1 < nrows) ? r + 1 : r;  // odd tail: row1 duplicates row0
            const block_q4_0 * row0 = (const block_q4_0 *) (buf + r * row_bytes);
            const block_q4_0 * row1 = (const block_q4_0 *) (buf + r1 * row_bytes);
            if (T >= 4) {
                mmid_rows2_tok4(row0->qs, row1->qs, bcol[0], bcol[1], bcol[2], bcol[3], K_blocks, &out_blk[0][r], &out_blk[0][r1]);
            } else if (T >= 2) {
                mmid_rows2_tok2(row0->qs, row1->qs, bcol[0], bcol[1], bcol[1], bcol[1], K_blocks, &out_blk[0][r], &out_blk[0][r1]);
                if (T == 3) {
                    mmid_rows2_tok1(row0->qs, row1->qs, bcol[2], bcol[2], bcol[2], bcol[2], K_blocks, &out_blk[2][r], &out_blk[2][r1]);
                }
            } else {
                mmid_rows2_tok1(row0->qs, row1->qs, bcol[0], bcol[0], bcol[0], bcol[0], K_blocks, &out_blk[0][r], &out_blk[0][r1]);
            }
        }
        for (int32_t i = 0; i < T; ++i) {
            if (line_ok && nrows == MMID_ROWS) {
                mmid_store_line(dcol[i], out_blk[i]);
            } else {
                for (int64_t r = 0; r < nrows; ++r) {
                    atomic_store_f32((volatile float *) (dcol[i] + r), out_blk[i][r]);
                }
            }
        }
    }
}

int ET_MMID_ENTRY_NAME(struct ggml_et_mul_mat_id_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    if (!kernel_env || !params) {
        return -1;
    }
    // partition by the launch mask the host actually used when it narrowed it (params->shire_mask, set only via
    // GGML_ET_MMID_SHIRE_MASK): platform.h's helpers are chip-wide stubs, see ggml_et_mask_threads()
    int       thread_id;
    const int num_threads = params->shire_mask ? ggml_et_mask_threads(params->shire_mask, get_hart_id(), &thread_id)
                                               : (thread_id = get_relative_thread_id(kernel_env->shire_mask),
                                                  get_num_threads(kernel_env->shire_mask));
    if (thread_id < 0) {
        return 0;
    }
    const struct ggml_tensor * src0 = &params->src0;
    const struct ggml_tensor * src1 = &params->src1;
    const struct ggml_tensor * src2 = &params->src2;
    const struct ggml_tensor * dst  = &params->dst;
    if (src0->type != GGML_TYPE_Q4_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        return -1;
    }
    const int64_t K = src0->ne[0], M = src0->ne[1], E = src0->ne[2];
    const int64_t S = src2->ne[0], N = src2->ne[1], ne11 = src1->ne[1];
    if (E > MMID_MAX_E || S * N > MMID_MAX_PAIRS || M < 2) {
        return mmid_q4_0_v0(params, env);
    }
    const char *    a_base   = (const char *) src0->data;
    const char *    b_base   = (const char *) src1->data;
    const int32_t * ids_base = (const int32_t *) src2->data;
    char *          d_base   = (char *) dst->data;
    if (!a_base || !b_base || !ids_base || !d_base || K % QK4_0 != 0) {
        return -1;
    }
    const size_t nb01 = src0->nb[1], nb02 = src0->nb[2];
    const size_t nb11 = src1->nb[1], nb12 = src1->nb[2];
    const size_t nb20 = src2->nb[0], nb21 = src2->nb[1];
    const size_t nbd1 = dst->nb[1], nbd2 = dst->nb[2];
    if (src0->nb[0] != sizeof(block_q4_0) || src1->nb[0] != sizeof(float) || nb20 != sizeof(int32_t) ||
        dst->nb[0] != sizeof(float)) {
        return -1;
    }
    const int64_t K_blocks = K / QK4_0;
    const int64_t row_bytes = K_blocks * (int64_t) sizeof(block_q4_0);
    const int     line_ok  = (((uintptr_t) d_base & 63) == 0) && (nbd1 % 64 == 0) && (nbd2 % 64 == 0);
    const int     rows_tight_packed = (nb01 == (size_t) row_bytes);  // 039b needs a straight contiguous row copy

    // ---- group the pairs by expert (every hart, private L2-SCP slice; pseudo-expert E = invalid ids) ----
    const uint64_t hart_in_shire = get_hart_id() & 63;
#if MMID_DIAG == 7
    if (hart_in_shire & 1) { return 0; }  // probes: the even hart owns the minion's whole 64 KiB SCP window
#endif
    int32_t * scratch = (int32_t *) et_shire_l2scp_local(hart_in_shire * MMID_SCRATCH_B);
    int32_t * counts  = scratch;                    // E+1
    int32_t * offs    = counts + (E + 1);           // E+2
    int32_t * pairs   = offs + (E + 2);             // S*N, (tok << 8 | slot)
    int32_t * active  = pairs + S * N;              // <= E+1
    for (int64_t e = 0; e <= E; ++e) {
        counts[e] = 0;
    }
    for (int64_t t = 0; t < N; ++t) {
        for (int64_t s = 0; s < S; ++s) {
            int32_t id = *(const int32_t *) ((const char *) ids_base + s * nb20 + t * nb21);
            if (id < 0 || id >= E) {
                id = (int32_t) E;
            }
            counts[id]++;
        }
    }
    offs[0] = 0;
    for (int64_t e = 0; e <= E; ++e) {
        offs[e + 1] = offs[e] + counts[e];
    }
    int32_t n_active = 0;
    for (int64_t e = 0; e <= E; ++e) {
        if (counts[e] > 0) {
            active[n_active++] = (int32_t) e;
        }
        counts[e] = offs[e];  // reuse as the scatter cursor
    }
    for (int64_t t = 0; t < N; ++t) {
        for (int64_t s = 0; s < S; ++s) {
            int32_t id = *(const int32_t *) ((const char *) ids_base + s * nb20 + t * nb21);
            if (id < 0 || id >= E) {
                id = (int32_t) E;
            }
            pairs[counts[id]++] = (int32_t) ((t << 8) | s);
        }
    }

    // rows per unit: 16 (one dst line per token) unless that leaves harts idle (small N), then 8/4/2
    int64_t rows_u = MMID_ROWS;
    while (rows_u > 2 && (int64_t) n_active * ((M + rows_u - 1) / rows_u) < num_threads) {
        rows_u >>= 1;
    }
    // 039b: shrink rows_u further so a double-buffered pair of (rows_u x row_bytes) fits this hart's SCP budget
    // left over after the grouping scratch above (counts/offs/pairs/active, sized for the worst case E/S*N this
    // hart could have grouped, not just n_active/S*N actually seen). scp_ok (set after row_blocks/total_units
    // below) is what actually gates the staged path - this loop only shrinks rows_u so staging CAN fit.
    const int64_t grouping_ints  = (E + 1) + (E + 2) + S * N + (E + 1);
    const int64_t grouping_bytes = ((grouping_ints * (int64_t) sizeof(int32_t) + 63) / 64) * 64;
    const int64_t scp_buf_avail  = MMID_SCRATCH_B - grouping_bytes - MMID_SCP_CTR_B;
    while (rows_u > 2 && 2 * rows_u * row_bytes > scp_buf_avail) {
        rows_u >>= 1;
    }
    const int64_t row_blocks  = (M + rows_u - 1) / rows_u;
    const int64_t total_units = (int64_t) n_active * row_blocks;
    // One SCP row-buffer slot for this hart's unit, rounded up to a whole number of 64B lines so buffer 1 always
    // starts line-aligned after buffer 0 (row_bytes = K_blocks*18 is not itself always a multiple of 64).
    const int64_t buf_bytes = ((rows_u * row_bytes + 63) / 64) * 64;
    const int     has_invalid = (n_active > 0 && active[n_active - 1] == (int32_t) E);
    const int64_t valid_total_units = total_units - (has_invalid ? row_blocks : 0);
    // tensor_load_l2scp copies whole 64B lines from a line-aligned DRAM address: every unit must start on a line
    // (a_base, e*nb02 and m0*nb01 all 64B multiples) and span whole lines (no short last unit) - otherwise
    // test-backend-ops shapes like K=256/rows_u=2 (2 x 144 B) staged garbage/NaN (2026-08-27). Fall back to the
    // direct path for such shapes; the production shapes (K=2048 M=768, K=768 M=2048, rows_u=16) qualify.
    const int scp_aligned = (((uintptr_t) a_base & 63) == 0) && ((rows_u * row_bytes) % 64 == 0) && (M % rows_u == 0);
    const int scp_ok = MMID_SCP_ENABLE && (MMID_DIAG == 0 || MMID_DIAG == 9 || MMID_DIAG == 10 || MMID_DIAG == 11) &&
                        rows_tight_packed && scp_aligned && rows_u >= 2 && 2 * buf_bytes <= scp_buf_avail &&
                        (num_threads % 2) == 0 && valid_total_units > 0;

    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    float __attribute__((aligned(32))) out_blk[MMID_T][MMID_ROWS];

#if MMID_DIAG == 7
    {   // tensor_load_l2scp probe, HANG-PROOF: never tensor_wait; sentinel the destination line, issue the load, poll
        // with a bounded spin. bits (variant = n_expert - 128): 1 setup_cache_scp on even harts, 2 CLEAR_TENSOR_ERROR
        // even, 4 CLEAR odd, 8 issue on thread 0, 32 issue on thread 1 (odd hart), 128 16 lines instead of 1,
        // 256 every even hart streams its unit's first row (18 lines, poll the last line). Reports per issuing hart:
        // d[0]=thread d[1]=issued d[2]=tensor_error d[3]=line changed d[4]=spins (x1000)
        const int variant = (int) (M - 768);  // variant channel: M = 768 + bits (keeps the host setup small)
        const int odd     = (int) (hart_in_shire & 1);
        if ((variant & 1) && !odd) { setup_cache_scp(); }
        if ((variant & 2) && !odd) { CLEAR_TENSOR_ERROR; }
        if ((variant & 4) && odd)  { CLEAR_TENSOR_ERROR; }
        if ((variant & 512) && !odd && (!(variant & 4096) || thread_id < num_threads / 2)) {  // 4096: shires 0-15 only
            // BANDWIDTH PROBE: this even hart streams the rows of ITS unit and its sibling's unit (u, u+1: 2 x rows_u
            // rows) into the minion's 64 KiB L2-SCP window (both harts' slices) with 16-line loads, ids alternating,
            // then polls the last line (bounded). Time vs the 33 GB/s cache path (DIAG 6).
            const uint64_t win0  = (hart_in_shire >> 1) * (2 * MMID_SCRATCH_B / 64) + 512;  // past the grouping scratch
            const int64_t  lpr   = (row_bytes + 63) / 64;
            uint64_t       dl    = win0;
            uint32_t       id    = 0;
            int            issued_n = 0;
            volatile uint32_t * last = 0;
            const char *   last_src = 0;
            const int           depth = (variant & 128) ? 16 : ((variant & 256) ? 4 : 1);
            volatile uint32_t * ring_p[2] = { 0, 0 };  // outstanding loads: last line + expected first word
            uint32_t            ring_w[2] = { 0, 0 };
            int32_t             spins_total = 0;
            if (variant & 2048) {
                // GLOBAL-SEQUENTIAL pattern: even hart h fetches 1 KiB chunks h, h+1024, h+2048, ... of the contiguous
                // weight tensor (32 chunks = 32 KiB per hart, 33 MB chip-wide, DRAM pages touched in order)
                const int64_t n_even = num_threads / 2;
                const int64_t my     = thread_id / 2;
                for (int64_t k = 0; k < 32; ++k) {
                    const char * src = a_base + (k * n_even + my) * 1024;
                    if ((variant & 1024) && issued_n > 0 && (issued_n % depth) == 0) { WAIT_TENSOR_LOAD_L2_0; }
                    last = (volatile uint32_t *) et_shire_l2scp_local((dl + 15) * 64);
                    last_src = src + 15 * 64;
                    last[0] = ~*(const uint32_t *) last_src;
                    FENCE; evict_to_l2((const void *) last, 1, 64); WAIT_CACHEOPS;
                    et_tensor_load_l2scp_conf_t c = { false, dl, (uint64_t) src, 15, 64, 0 };
                    et_tensor_load_l2scp(&c);
                    issued_n++; dl += 16;
                }
            } else
            for (int k = 0; k < 2; ++k) {
                const int64_t uu = (int64_t) thread_id + k;
                if (uu >= total_units) { break; }
                const char * ub = a_base + (int64_t) active[uu / row_blocks] * (int64_t) nb02 + (uu % row_blocks) * rows_u * (int64_t) nb01;
                const int64_t nr = (M - (uu % row_blocks) * rows_u < rows_u) ? (M - (uu % row_blocks) * rows_u) : rows_u;
                for (int64_t r = 0; r < nr; ++r) {
                    const char * row = ub + r * nb01;
                    int64_t left = lpr, off = 0;
                    while (left > 0) {
                        const int64_t n = left > 16 ? 16 : left;
                        if (dl + n > win0 + 2 * MMID_SCRATCH_B / 64 - 512) { break; }  // window full
                        // canonical double buffer: at most one load outstanding per id (the queue is shallow)
                        // bit 1024: hardware wait (id 0) every D loads (D = 16 if bit 128, 4 if bit 256, else 1);
                        // otherwise at most 2 loads in flight, the oldest retired by evict-polling its last line
                        if ((variant & 1024) && issued_n > 0 && (issued_n % depth) == 0) { WAIT_TENSOR_LOAD_L2_0; }
                        if (!(variant & 1024) && issued_n >= 2) {
                            volatile uint32_t * rp = ring_p[issued_n & 1]; const uint32_t rw = ring_w[issued_n & 1];
                            for (int32_t sp = 0; sp < 200000; ++sp, ++spins_total) {
                                evict_to_l2((const void *) rp, 1, 64); WAIT_CACHEOPS;
                                if (rp[0] == rw) { break; }
                            }
                        }
                        last = (volatile uint32_t *) et_shire_l2scp_local((dl + n - 1) * 64);
                        last_src = row + off + (n - 1) * 64;
                        last[0] = ~*(const uint32_t *) last_src;  // sentinel != expected word, pushed to the SCP line
                        FENCE; evict_to_l2((const void *) last, 1, 64); WAIT_CACHEOPS;
                        ring_p[issued_n & 1] = last; ring_w[issued_n & 1] = *(const uint32_t *) last_src;
                        et_tensor_load_l2scp_conf_t c = { false, dl, (uint64_t) (row + off), (uint64_t) (n - 1), 64, id };
                        et_tensor_load_l2scp(&c);
                        issued_n++;
                        dl += n; off += n * 64; left -= n;
                    }
                }
            }
            if ((variant & 1024) && issued_n > 0) { WAIT_TENSOR_LOAD_L2_0; }
            int32_t spins = 0, changed = 0;
            if (last) {
                const uint32_t want = *(const uint32_t *) last_src;  // DRAM value that must appear in the SCP line
                for (; spins < 200000; ++spins) {
                    evict_to_l2((const void *) last, 1, 64);
                    WAIT_CACHEOPS;
                    if (last[0] == want) { changed = 1; break; }
                }
            }
            const uint64_t terr = get_tensor_error();
            const int64_t t = pairs[0] >> 8, sl = pairs[0] & 255;
            float * d = (float *) (d_base + sl * nbd1 + t * nbd2) + (thread_id % (M / 8)) * 5;
            atomic_store_f32((volatile float *) (d + 0), (float) thread_id);
            atomic_store_f32((volatile float *) (d + 1), 1.0f);
            atomic_store_f32((volatile float *) (d + 2), (float) (int32_t) (terr & 0xFFFF));
            atomic_store_f32((volatile float *) (d + 3), (float) changed);
            atomic_store_f32((volatile float *) (d + 4), (float) ((spins + spins_total) / 1000));
            __asm__ volatile("mova.m.x %0" ::"r"(saved_mask)); FENCE; return 0;
        }
        const int issue = ((variant & 8) && thread_id == 0) || ((variant & 32) && thread_id == 1) || ((variant & 256) && !odd);
        if (issue) {
            const uint64_t line0  = hart_in_shire * (MMID_SCRATCH_B / 64) + 256;  // past the grouping scratch
            const int      nlines = (variant & 256) ? 18 : ((variant & 128) ? 16 : 1);
            volatile uint32_t * last = (volatile uint32_t *) et_shire_l2scp_local((line0 + nlines - 1) * 64);
            for (int w = 0; w < 16; ++w) { last[w] = 0xC0FFEE00u + (uint32_t) w; }
            FENCE;
            evict_to_l2((const void *) last, 1, 64);  // sentinel into the L2 SCP line itself, L1 copy dropped
            WAIT_CACHEOPS;
            const char * src = a_base + (variant & 256 ? m0_of_unit(thread_id, row_blocks, rows_u, active, nb01, nb02) : 0);
            int64_t left = nlines, off = 0; uint64_t dl = line0;
            while (left > 0) {
                const int64_t n = left > 16 ? 16 : left;
                et_tensor_load_l2scp_conf_t c = { false, dl, (uint64_t) (src + off), (uint64_t) (n - 1), 64, 0 };
                et_tensor_load_l2scp(&c);
                dl += n; off += n * 64; left -= n;
            }
            if (variant & 16) { WAIT_TENSOR_LOAD_L2_0; }  // isolate the hardware wait itself
            int32_t spins = 0, changed = 0;
            for (; spins < 200000; ++spins) {  // every read re-fetches the line from L2 SCP (tensor writes bypass L1)
                evict_to_l2((const void *) last, 1, 64);
                WAIT_CACHEOPS;
                if (last[0] != 0xC0FFEE00u || last[15] != 0xC0FFEE0Fu) { changed = 1; break; }
            }
            const uint64_t terr = get_tensor_error();
            const int64_t t = pairs[0] >> 8, sl = pairs[0] & 255;
            float * d = (float *) (d_base + sl * nbd1 + t * nbd2) + (thread_id % (M / 8)) * 5;
            atomic_store_f32((volatile float *) (d + 0), (float) thread_id);
            atomic_store_f32((volatile float *) (d + 1), 1.0f);
            atomic_store_f32((volatile float *) (d + 2), (float) (int32_t) (terr & 0xFFFF));
            atomic_store_f32((volatile float *) (d + 3), (float) changed);
            atomic_store_f32((volatile float *) (d + 4), (float) (spins / 1000));
        }
        __asm__ volatile("mova.m.x %0" ::"r"(saved_mask)); FENCE; return 0;
    }
#endif
#if MMID_DIAG == 4
    if (total_units > 0) { __asm__ volatile("mova.m.x %0" ::"r"(saved_mask)); FENCE; return 0; }  // grouping only
#endif
#if MMID_DIAG == 11
#    define MMID_TRACE(...) __VA_ARGS__
#else
#    define MMID_TRACE(...)
#endif
#if MMID_DIAG == 0 || MMID_DIAG == 9 || MMID_DIAG == 10 || MMID_DIAG == 11
    if (scp_ok) {
        // ---- invalid-expert units: no weights to load, same zeroing as the direct path below, independent per hart ----
        if (has_invalid) {
            for (int64_t u = thread_id; u < total_units; u += num_threads) {
                if (u < valid_total_units) { continue; }
                const int64_t rb    = u % row_blocks;
                const int64_t m0    = rb * rows_u;
                const int64_t nrows = (M - m0 < rows_u) ? (M - m0) : rows_u;
                const int32_t p0 = offs[E], p1 = offs[E + 1];
                for (int32_t p = p0; p < p1; ++p) {
                    const int64_t t = pairs[p] >> 8, s = pairs[p] & 255;
                    float *       d = (float *) (d_base + s * nbd1 + t * nbd2) + m0;
                    for (int64_t r = 0; r < nrows; ++r) {
                        atomic_store_f32((volatile float *) (d + r), 0.0f);
                    }
                }
            }
        }

        // ---- 039b: SCP-staged double buffer. Hart 0 of each minion is the dedicated streamer (only even harts can
        // issue tensor_load_l2scp): it stages both units of a pair (its own "a" unit and its partner's "b" unit)
        // one or two pairs ahead of compute (nbuf slots). Hart 1 is the dedicated computer: it computes both units
        // of the pair from SCP once staged. The overlap comes from hart 0 and hart 1 being separate physical cores
        // running concurrently (hart 0 streams pair p+1 while hart 1 computes pair p), not from interleaving
        // load/compute on a single hart's instruction stream (tensor_load_l2scp allows only one outstanding load
        // per id, so a single hart cannot itself issue a load and do unrelated compute before waiting on it without
        // this same restructuring). MMID_DIAG==9 forces nbuf=1 (strict alternation, no overlap - correctness/bring-up
        // check); default and MMID_DIAG==10 use nbuf=2 (double-buffered, the actual overlap).
        const int      is_hart0    = !(hart_in_shire & 1);
        const uint64_t his0        = hart_in_shire & ~1ULL;  // this minion's hart-0 index (unit "a"'s SCP region)
        const uint64_t his1        = his0 | 1ULL;            // this minion's hart-1 index (unit "b"'s SCP region)
        char * const   base0       = (char *) et_shire_l2scp_local(his0 * (uint64_t) MMID_SCRATCH_B);
        char * const   base1       = (char *) et_shire_l2scp_local(his1 * (uint64_t) MMID_SCRATCH_B);
        char * const   buf_a[2]    = { base0 + grouping_bytes, base0 + grouping_bytes + buf_bytes };
        char * const   buf_b[2]    = { base1 + grouping_bytes, base1 + grouping_bytes + buf_bytes };
        volatile uint32_t * const ready_ctr    = (volatile uint32_t *) (base0 + grouping_bytes + 2 * buf_bytes);
        volatile uint32_t * const consumed_ctr = (volatile uint32_t *) (base0 + grouping_bytes + 2 * buf_bytes + 64);
        const uint64_t his0_line0 = mmid_scp_line(his0, (uint64_t) grouping_bytes);
        const uint64_t his1_line0 = mmid_scp_line(his1, (uint64_t) grouping_bytes);
#if MMID_DIAG == 9 || MMID_DIAG == 11
        const int nbuf = 1;
#else
        const int nbuf = 2;
#endif

        if (is_hart0) {
            CLEAR_TENSOR_ERROR;
            *ready_ctr = 0; *consumed_ctr = 0;
            FENCE;
            evict_to_l2((const void *) ready_ctr, 1, 64);
            evict_to_l2((const void *) consumed_ctr, 1, 64);
            WAIT_CACHEOPS;
        }
        et_barrier(ET_BARRIER_MINION);  // hart 1 must not scp_wait before hart 0's reset above is visible

        const int64_t num_minions = num_threads / 2;
        const int64_t minion_id   = thread_id / 2;
        const int64_t total_pairs = (valid_total_units + 1) / 2;
        MMID_TRACE(uint32_t dg_wid = 0, dg_wfail = 0, dg_units = 0, dg_match = 0;)

        if (is_hart0) {
            uint32_t wid = 0;
            for (int64_t p = minion_id; p < total_pairs; p += num_minions) {
                const int64_t ua    = 2 * p, ub = 2 * p + 1;
                const int     has_b = ub < valid_total_units;
                const int     buf   = (int) (wid % (uint32_t) nbuf);
                if (wid >= (uint32_t) nbuf) {
                    const int ok = mmid_scp_wait_bounded(consumed_ctr, wid - (uint32_t) nbuf + 1);
                    (void) ok;
                    MMID_TRACE(dg_wfail += (ok == 0);)
                }

                const int32_t e_a  = active[ua / row_blocks];
                const int64_t m0_a = (ua % row_blocks) * rows_u;
                const int64_t nr_a = (M - m0_a < rows_u) ? (M - m0_a) : rows_u;
                const char *  src_a = a_base + (int64_t) e_a * nb02 + m0_a * (int64_t) nb01;
                {
                    const int lf = mmid_load_rows_scp(his0_line0 + (uint64_t) buf * (buf_bytes / 64), src_a, nr_a * row_bytes);
                    (void) lf;
                    MMID_TRACE(dg_wfail += lf;)
                }
                MMID_TRACE({   // did the staged data land? (first word of the unit's first row, re-read from L2 SCP)
                    evict_to_l2((const void *) buf_a[buf], 1, 64);
                    WAIT_CACHEOPS;
                    dg_match += (*(volatile const uint32_t *) buf_a[buf] == *(const uint32_t *) src_a);
                    dg_units += 1 + has_b;
                })
                if (has_b) {
                    const int32_t e_b  = active[ub / row_blocks];
                    const int64_t m0_b = (ub % row_blocks) * rows_u;
                    const int64_t nr_b = (M - m0_b < rows_u) ? (M - m0_b) : rows_u;
                    const int lf = mmid_load_rows_scp(his1_line0 + (uint64_t) buf * (buf_bytes / 64),
                                                      a_base + (int64_t) e_b * nb02 + m0_b * (int64_t) nb01, nr_b * row_bytes);
                    (void) lf;
                    MMID_TRACE(dg_wfail += lf;)
                }
                wid++;
                scp_signal(ready_ctr, wid);
            }
            MMID_TRACE(dg_wid = wid;)
        } else {
            uint32_t wid = 0;
            for (int64_t p = minion_id; p < total_pairs; p += num_minions) {
                const int64_t ua    = 2 * p, ub = 2 * p + 1;
                const int     has_b = ub < valid_total_units;
                const int     buf   = (int) (wid % (uint32_t) nbuf);
                wid++;
                {
                    const int ok = mmid_scp_wait_bounded(ready_ctr, wid);
                    (void) ok;
                    MMID_TRACE(dg_wfail += (ok == 0);)
                }

                const int32_t e_a  = active[ua / row_blocks];
                const int64_t m0_a = (ua % row_blocks) * rows_u;
                const int64_t nr_a = (M - m0_a < rows_u) ? (M - m0_a) : rows_u;
                const int32_t pa0 = offs[e_a], pa1 = offs[e_a + 1];
                mmid_evict_rows(buf_a[buf], nr_a * row_bytes);
                MMID_TRACE({
                    dg_match += (*(volatile const uint32_t *) buf_a[buf] ==
                                 *(const uint32_t *) (a_base + (int64_t) e_a * nb02 + m0_a * (int64_t) nb01));
                    dg_units += 1 + has_b;
                })
                mmid_compute_unit_scp(buf_a[buf], row_bytes, K_blocks, nr_a, m0_a, pa0, pa1, pairs, b_base, nb11,
                                       nb12, d_base, nbd1, nbd2, ne11, line_ok, out_blk);
                if (has_b) {
                    const int32_t e_b  = active[ub / row_blocks];
                    const int64_t m0_b = (ub % row_blocks) * rows_u;
                    const int64_t nr_b = (M - m0_b < rows_u) ? (M - m0_b) : rows_u;
                    const int32_t pb0 = offs[e_b], pb1 = offs[e_b + 1];
                    mmid_evict_rows(buf_b[buf], nr_b * row_bytes);
                    mmid_compute_unit_scp(buf_b[buf], row_bytes, K_blocks, nr_b, m0_b, pb0, pb1, pairs, b_base, nb11,
                                           nb12, d_base, nbd1, nbd2, ne11, line_ok, out_blk);
                }
                scp_signal(consumed_ctr, wid);
            }
            MMID_TRACE(dg_wid = wid;)
        }
        MMID_TRACE({   // publish this hart's trace record into src1 (clobbers activations - diagnostics only)
            const int64_t  rec_cap = (K * ne11 * N) / 16;
            if (rec_cap > 0) {
                float * rec = (float *) (uintptr_t) b_base + (int64_t) (thread_id % rec_cap) * 16;
                atomic_store_f32((volatile float *) (rec + 0), (float) thread_id);
                atomic_store_f32((volatile float *) (rec + 1), 2.0f);
                atomic_store_f32((volatile float *) (rec + 2), (float) (int32_t) dg_wid);
                atomic_store_f32((volatile float *) (rec + 3), (float) (int32_t) dg_wfail);
                atomic_store_f32((volatile float *) (rec + 4), (float) (int32_t) dg_units);
                atomic_store_f32((volatile float *) (rec + 5), (float) (int32_t) dg_match);
                atomic_store_f32((volatile float *) (rec + 6), (float) (int32_t) (kernel_env->shire_mask & 0xFFFF));  // (terr was always 0)
                atomic_store_f32((volatile float *) (rec + 7), (float) is_hart0);
                atomic_store_f32((volatile float *) (rec + 8), (float) num_threads);
                atomic_store_f32((volatile float *) (rec + 9), (float) offs[E]);
                atomic_store_f32((volatile float *) (rec + 10), (float) pairs[0]);
                atomic_store_f32((volatile float *) (rec + 11), (float) active[0]);
                atomic_store_f32((volatile float *) (rec + 12), (float) n_active);
                atomic_store_f32((volatile float *) (rec + 13), (float) (int32_t) rows_u);
                atomic_store_f32((volatile float *) (rec + 14), (float) (int32_t) (params->shire_mask & 0xFFFF));  // was dg_np (always 1)
                atomic_store_f32((volatile float *) (rec + 15), (float) (int32_t) (get_hart_id() & 0xFFFF));
            }
        })
    }
#endif
    if (!scp_ok)
    for (int64_t u = thread_id; u < total_units; u += num_threads) {
        const int32_t e     = active[u / row_blocks];
        const int64_t rb    = u % row_blocks;
        const int64_t m0    = rb * rows_u;
        const int64_t nrows = (M - m0 < rows_u) ? (M - m0) : rows_u;
        const int32_t p0 = offs[e], p1 = offs[e + 1];

        if (e == (int32_t) E) {  // invalid expert ids: zero the outputs
            for (int32_t p = p0; p < p1; ++p) {
                const int64_t t = pairs[p] >> 8, s = pairs[p] & 255;
                float *       d = (float *) (d_base + s * nbd1 + t * nbd2) + m0;
                for (int64_t r = 0; r < nrows; ++r) {
                    atomic_store_f32((volatile float *) (d + r), 0.0f);
                }
            }
            continue;
        }
        const char * exp_base = a_base + (int64_t) e * nb02;
#if MMID_DIAG >= 6
        {   // streaming probes: read the unit's rows with one method, no compute
            const int64_t lines_per_row = (row_bytes + 63) / 64;
#if MMID_DIAG == 6   // prefetch_va all rows, then wait
            for (int64_t r = 0; r < nrows; ++r) { mmid_prefetch_row(exp_base + (m0 + r) * nb01, row_bytes); }
            WAIT_PREFETCH_0;
#elif MMID_DIAG == 7 // tensor_load_l2scp bisect: variant = n_expert - 128 (see 039 NOTES)
            //  0: nothing (prologue only)   1: thread 0: one 1-line load, no wait   2: 1 + WAIT_TENSOR_LOAD_L2_0
            //  3: hart 0 of every minion: 16 lines + wait   4: hart 1 of every minion: 16 lines + wait
            //  5: full unit streaming on even harts (as before)   6: full streaming on odd harts
            const int variant = (int) (M - 768);  // variant channel: M = 768 + bits (keeps the host setup small)
            const int odd     = (int) (hart_in_shire & 1);
            uint64_t  terr    = 0xEEEE;
            int       issued  = 0;
            if (variant == 1 || variant == 2) {
                if (thread_id == 0) {
                    et_tensor_load_l2scp_conf_t c = { false, 0, (uint64_t) exp_base, 0, 64, 0 };
                    et_tensor_load_l2scp(&c);
                    if (variant == 2) { WAIT_TENSOR_LOAD_L2_0; }
                    issued = 1;
                }
            } else if (variant == 3 || variant == 4) {
                if (odd == (variant == 4)) {
                    et_tensor_load_l2scp_conf_t c = { false, hart_in_shire * (MMID_SCRATCH_B / 64), (uint64_t) exp_base, 15, 64, 0 };
                    et_tensor_load_l2scp(&c);
                    WAIT_TENSOR_LOAD_L2_0;
                    issued = 1;
                }
            } else if (variant == 5 || variant == 6) {
                if (odd == (variant == 6)) {
                    uint64_t dst_line = hart_in_shire * (MMID_SCRATCH_B / 64);
                    for (int64_t r = 0; r < nrows; ++r) {
                        const char * row = exp_base + (m0 + r) * nb01;
                        int64_t left = lines_per_row, off = 0;
                        while (left > 0) {
                            const int64_t n = left > 16 ? 16 : left;
                            et_tensor_load_l2scp_conf_t c = { false, dst_line, (uint64_t) (row + off), (uint64_t) (n - 1), 64, 0 };
                            et_tensor_load_l2scp(&c);
                            dst_line += n; off += n * 64; left -= n;
                        }
                    }
                    WAIT_TENSOR_LOAD_L2_0;
                    issued = 1;
                }
            }
            if (issued) {
                terr = get_tensor_error();
            }
            if (u == thread_id) {  // first unit of every hart: publish (hart id, issued, tensor error) into its dst column
                const int64_t t = pairs[p0] >> 8, sl = pairs[p0] & 255;
                float * d = (float *) (d_base + sl * nbd1 + t * nbd2) + m0;
                atomic_store_f32((volatile float *) (d + 0), (float) thread_id);
                atomic_store_f32((volatile float *) (d + 1), (float) issued);
                atomic_store_f32((volatile float *) (d + 2), (float) (int32_t) (terr & 0xFFFF));
            }
            continue;
#else                // demand loads: one vector load per line, accumulated so they stay live
            __asm__ volatile("fbci.pi f20, 0" ::: "f20");
            for (int64_t r = 0; r < nrows; ++r) {
                const char * row = exp_base + (m0 + r) * nb01;
                for (int64_t l = 0; l < lines_per_row; ++l) {
                    __asm__ volatile("flw.ps f10, 0(%[p])\n fadd.ps f20, f20, f10\n" : : [p] "r"(row + l * 64) : "f10", "f20", "memory");
                }
            }
            uint32_t sink; __asm__ volatile("fmv.x.s %0, f20" : "=r"(sink)); out_blk[0][0] = (float) (int32_t) sink;
#endif
            continue;
        }
#endif
        // stage the unit's rows in L2 ahead of the gathers (two row pairs ahead of the compute)
        mmid_prefetch_row(exp_base + m0 * nb01, row_bytes);
        if (nrows > 1) { mmid_prefetch_row(exp_base + (m0 + 1) * nb01, row_bytes); }
        if (nrows > 2) { mmid_prefetch_row(exp_base + (m0 + 2) * nb01, row_bytes); }
        if (nrows > 3) { mmid_prefetch_row(exp_base + (m0 + 3) * nb01, row_bytes); }

        for (int32_t pg = p0; pg < p1; pg += MMID_T) {
            const int32_t T = (p1 - pg < MMID_T) ? (p1 - pg) : MMID_T;
            const float * bcol[MMID_T];
            float *       dcol[MMID_T];
            for (int32_t i = 0; i < MMID_T; ++i) {
                const int32_t pi = pg + (i < T ? i : 0);
                const int64_t t = pairs[pi] >> 8, s = pairs[pi] & 255;
                bcol[i] = (const float *) (b_base + (s % ne11) * nb11 + t * nb12);
                dcol[i] = (float *) (d_base + s * nbd1 + t * nbd2) + m0;
            }
            for (int64_t r = 0; r < nrows; r += 2) {
                const int64_t      r1   = (r + 1 < nrows) ? r + 1 : r;  // odd tail: row1 duplicates row0
                const block_q4_0 * row0 = (const block_q4_0 *) (exp_base + (m0 + r) * nb01);
                const block_q4_0 * row1 = (const block_q4_0 *) (exp_base + (m0 + r1) * nb01);
                if (pg == p0 && r + 4 < nrows) {  // next-but-one row pair (first token pass only: rows stay in L2)
                    mmid_prefetch_row(exp_base + (m0 + r + 4) * nb01, row_bytes);
                    if (r + 5 < nrows) { mmid_prefetch_row(exp_base + (m0 + r + 5) * nb01, row_bytes); }
                }
#if MMID_DIAG == 5
                if (K_blocks > 0) { continue; }  // unit loop without the K-loop asm (prefetch + stores only)
#endif
                if (T >= 4) {
                    mmid_rows2_tok4(row0->qs, row1->qs, bcol[0], bcol[1], bcol[2], bcol[3], K_blocks, &out_blk[0][r], &out_blk[0][r1]);
                } else if (T >= 2) {
                    mmid_rows2_tok2(row0->qs, row1->qs, bcol[0], bcol[1], bcol[1], bcol[1], K_blocks, &out_blk[0][r], &out_blk[0][r1]);
                    if (T == 3) {
                        mmid_rows2_tok1(row0->qs, row1->qs, bcol[2], bcol[2], bcol[2], bcol[2], K_blocks, &out_blk[2][r], &out_blk[2][r1]);
                    }
                } else {
                    mmid_rows2_tok1(row0->qs, row1->qs, bcol[0], bcol[0], bcol[0], bcol[0], K_blocks, &out_blk[0][r], &out_blk[0][r1]);
                }
            }
            for (int32_t i = 0; i < T; ++i) {
                if (line_ok && nrows == MMID_ROWS) {
                    mmid_store_line(dcol[i], out_blk[i]);
                } else {
                    for (int64_t r = 0; r < nrows; ++r) {
                        atomic_store_f32((volatile float *) (dcol[i] + r), out_blk[i][r]);
                    }
                }
            }
        }
    }

    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
    FENCE;
    return 0;
}
