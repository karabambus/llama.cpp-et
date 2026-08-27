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
};
