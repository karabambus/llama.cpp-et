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
    // to instruction i (its body on the slowest hart plus the next instruction's evicts and barrier).
    uint64_t prof_buf;
};
