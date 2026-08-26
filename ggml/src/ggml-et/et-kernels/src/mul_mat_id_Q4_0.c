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
#define MMID_SCRATCH_B 32768
#ifndef MMID_DIAG
#    define MMID_DIAG 0   // 1: no token loads/FMAs  2: no gathers/unpack  3: no B loads  4: grouping only  5: no K loop (diagnostics only)
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

int ET_MMID_ENTRY_NAME(struct ggml_et_mul_mat_id_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    if (!kernel_env || !params) {
        return -1;
    }
    const int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    const int num_threads = get_num_threads(kernel_env->shire_mask);
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
    const int     line_ok  = (((uintptr_t) d_base & 63) == 0) && (nbd1 % 64 == 0) && (nbd2 % 64 == 0);

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
    const int64_t row_blocks  = (M + rows_u - 1) / rows_u;
    const int64_t total_units = (int64_t) n_active * row_blocks;
    const int64_t row_bytes   = K_blocks * (int64_t) sizeof(block_q4_0);

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
