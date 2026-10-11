#pragma once

#include "ggml-backend-impl.h"
#include "ggml-et-uberkernel-common.h"

#include <device-layer/IDeviceLayer.h>
#include <runtime/IProfiler.h>
#include <runtime/IRuntime.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

std::shared_ptr<rt::IRuntime> ggml_et_runtime();

struct ggml_backend_et_buffer_type_context {
    int         devidx;
    std::string name;
};

struct ggml_backend_et_buffer_context {
    int          devidx;
    void *       data;  // Device memory pointer
    size_t       size;
    rt::DeviceId rtid;
};

struct ggml_backend_et_context {
    int devidx;
};

struct ggml_backend_et_device_context;

// One slot in the uberkernel ring. The host vectors back the H2D copy and
// must outlive the upload; the device buffers feed the kernel that consumes
// them. pending_event lets us know when both have drained so the slot can
// be recycled.
struct ggml_backend_et_uberkernel_slot {
    std::vector<ggml_et_uberkernel_inst> insts;
    std::vector<std::byte>               params_blob;

    std::byte * device_insts           = nullptr;
    std::byte * device_params          = nullptr;
    size_t      device_insts_capacity  = 0;
    size_t      device_params_capacity = 0;
    std::byte * device_prof            = nullptr;  // GGML_ET_UK_PROFILE stamps, (num_insts + 1) * 8 bytes
    size_t      device_prof_capacity   = 0;
    std::vector<std::string> inst_names;  // graph node of each instruction (GGML_ET_UK_PROFILE_NODES only)

    rt::EventId pending_event{};
    bool        has_pending = false;
};

struct ggml_backend_et_uberkernel_context {
    bool     failed     = false;
    uint64_t shire_mask = 0;

    // Ring of slots. We accumulate into slots[current_slot]; on segment
    // commit we fire the H2D + launch and rotate to the next slot,
    // waiting on its previous launch only if it hasn't drained yet.
    static constexpr size_t         SLOT_COUNT = 4;
    ggml_backend_et_uberkernel_slot slots[SLOT_COUNT];
    size_t                          current_slot = 0;
};

// One small host->device upload staged by set_tensor and not yet sent.
struct ggml_et_input_copy_src {
    const std::byte * src;   // private copy owned by the device context (staged_h2d)
    std::byte *       dst;
    size_t            size;
};

struct ggml_backend_et_device_context {
    int                        devidx;
    rt::DeviceId               rtid;
    std::string                name;
    std::string                desc;
    size_t                     total_mem;
    ggml_backend_buffer_type_t buftype;

    // Kernel management - default stream for ordered execution on this device
    rt::StreamId                                  default_stream;
    std::unordered_map<std::string, rt::KernelId> loaded_kernels;

    // trace buffer - for printing support
    std::byte * trace_buffer;

    // private copy of moe_sum routing weights when the graph allocator placed the fused output over them
    std::byte * moe_w_scratch = nullptr;
    std::byte * pq2_tail_scratch = nullptr;  // PQ2_0 GEMV K-split tail records (zeroed once), see mul_mat_PQ2_0.c
    uint32_t    pq2_tail_gen     = 0;        // generation of the last launch that used it
    std::byte * f32_ksplit_scratch = nullptr;  // mul_mat_f32 K-split row records (zeroed once), see mul_mat_f32.c
    std::byte * pq2_i8_a8 = nullptr;  // int8 activations of the PQ2_0 int8 engine (quantize_q8k_f32 engine mode)
    size_t      pq2_i8_a8_bytes = 0;
    std::byte * pq2_i8_sc = nullptr;  // their per-(8-token, 256-K) scale lines
    size_t      pq2_i8_sc_bytes = 0;
    uint64_t    graph_seq = 0;            // incremented by every graph_compute
    // what the int8 scratch holds: the activation tensor, its data and shape, and the graph it was quantised in
    const ggml_tensor * pq2_i8_src = nullptr;
    const void *        pq2_i8_src_data = nullptr;
    int64_t             pq2_i8_src_ne[2] = { 0, 0 };
    size_t              pq2_i8_src_nb1 = 0;
    uint64_t            pq2_i8_src_seq = 0;

    bool                               uberkernel_enabled = false;
    // Set per graph by graph_compute: batching is enabled AND the graph is decode-sized (token count at or under
    // GGML_ET_UK_MAX_TOKENS). Every instruction pays a chip-wide barrier plus the consumer-side evicts, which on long
    // prefill instructions outweigh the launch savings: on Bonsai-2-27B batching is +37 % at one token and already
    // -1 % at 16 tokens, -7 % at 128 (home box r116-r118, r132-r133), so the default bound is 8.
    bool                               uberkernel_active  = false;
    int                                uberkernel_max_tokens = 0;
    ggml_backend_et_uberkernel_context uberkernel;

    // Small set_tensor uploads (graph inputs) staged instead of sent at once. graph_compute folds them into the
    // segment's first instruction when the graph is batched; every other path sends them, in order, before it
    // queues anything (ggml_et_flush_staged_h2d). The private copies live until the stream has drained, because
    // the runtime reads a host source from a worker thread after memcpyHostToDevice returns.
    std::vector<std::vector<std::byte>> staged_h2d;
    size_t                              staged_h2d_bytes = 0;
    std::vector<ggml_et_input_copy_src> pending_h2d;
    // True once anything has been queued on default_stream without waiting for it (kernel launch, segment, async copy,
    // staged flush); cleared when synchronize drains the stream. llama's scheduler synchronizes before copying each
    // user input (12 times per decode graph): with nothing in flight there is nothing to wait for, and the staged
    // inputs already own private copies, so synchronize returns at once and the graph can fold them in.
    bool                                stream_busy = true;
};

struct ggml_backend_et_reg_ctx {
    std::vector<ggml_backend_dev_t> devices;
};
