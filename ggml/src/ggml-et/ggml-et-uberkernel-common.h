#pragma once

#include <stdint.h>

struct ggml_et_uberkernel_inst {
    uint16_t kernel_id;
    uint16_t flags;
    uint32_t params_offset;
    uint32_t params_size;
};

struct ggml_et_uberkernel_params {
    uint32_t num_insts;
    uint32_t inst_stride;
    uint64_t insts;
    uint64_t params_blob;
    // 052: bit 0 = eviction mode 1 (per hart: whole-L1 evict syscall + the L2->L3 region walk split across the 64
    // harts of the shire) instead of mode 0 (every hart walks every input region). Host: GGML_ET_UK_EVICT_MODE.
    uint32_t flags;
    uint32_t reserved;
    // flags bit 3 (host GGML_ET_UK_PROFILE): hart 0 writes hpmcounter3 (600 MHz) after every per-instruction
    // barrier into prof_buf[0..num_insts] (num_insts + 1 entries); the host attributes prof_buf[i+1] - prof_buf[i]
    // to instruction i (its body on the slowest hart plus the next instruction's evicts and barrier; a final
    // chip-wide barrier precedes the last stamp). The host poisons the buffer and rejects non-monotonic stamps.
    uint64_t prof_buf;
};

// Reserved instruction (outside the generated kernel ids): copy small graph inputs into place at the start of a
// segment. The bytes travel inside the segment's parameter upload instead of one host->device command each (each
// such command is a ~1.2 ms round trip through the master minion, seven per decode graph). Layout at the
// instruction's params: header, n_jobs jobs, then the data; job.src_off is relative to the params start, words are
// 32-bit. The device writes with global atomic stores, which later plain loads of any shire see.
#define GGML_ET_UBERKERNEL_INPUT_COPY 0xFFF0u
struct ggml_et_uk_input_copy_hdr {
    uint32_t n_jobs;
    uint32_t reserved;
};
struct ggml_et_uk_input_copy_job {
    uint64_t dst;
    uint32_t src_off;
    uint32_t words;
};
#ifdef __cplusplus
static_assert(sizeof(struct ggml_et_uk_input_copy_hdr) == 8, "input copy header layout");
static_assert(sizeof(struct ggml_et_uk_input_copy_job) == 16, "input copy job layout");
#else
_Static_assert(sizeof(struct ggml_et_uk_input_copy_hdr) == 8, "input copy header layout");
_Static_assert(sizeof(struct ggml_et_uk_input_copy_job) == 16, "input copy job layout");
#endif
