#include <map>
#include "ggml-et-ops.h"
#include <algorithm>
#include "ggml-et-kernels.h"
#include <chrono>

#include "ggml-et-kernels-embed.hpp"
#include "ggml-et-uberkernel-kernel-map.h"
#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>
#include <atomic>
#include <vector>
#include <string>
#include <fstream>

#define ET_TRACE_DECODER_IMPL
#include <et-trace/decoder.h>
#include <et-trace/layout.h>

static constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

static size_t ggml_et_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

static size_t ggml_et_next_capacity(size_t current_capacity, size_t required_capacity) {
    if (current_capacity == 0) {
        return required_capacity;
    }

    size_t next_capacity = current_capacity;
    while (next_capacity < required_capacity) {
        next_capacity *= 2;
    }

    return next_capacity;
}

// 052: GGML_ET_UBERKERNEL_STATS=1 prints per graph_compute how the launches were batched (segments / batched
// instructions / standalone launches) - the number that decides whether the uberkernel pays.
struct ggml_et_uk_stats { uint64_t segments = 0, insts = 0, standalone = 0, graphs = 0; };
static ggml_et_uk_stats g_uk_stats;
static std::map<std::string, std::pair<uint64_t, uint64_t>> g_uk_prof;  // kernel [shape] -> (n, cycles)
static uint64_t g_uk_prof_bad = 0;  // segments whose stamps were unwritten or not monotonic
static_assert(sizeof(ggml_et_uberkernel_params) == 40, "host/device params layout");
static const bool g_uk_stats_on = [] { const char * e = getenv("GGML_ET_UBERKERNEL_STATS"); return e && *e && strcmp(e, "0") != 0; }();

static ggml_backend_et_uberkernel_slot & ggml_et_uberkernel_current_slot(ggml_backend_et_uberkernel_context * uk_ctx) {
    return uk_ctx->slots[uk_ctx->current_slot];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
static void ggml_et_uberkernel_slot_wait(ggml_backend_et_uberkernel_slot &     slot,
                                         const std::shared_ptr<rt::IRuntime> & runtime) {
    if (!slot.has_pending || !runtime) {
        return;
    }
    runtime->waitForEvent(slot.pending_event);
    slot.has_pending = false;
}

static void ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    if (!uk_ctx) {
        return;
    }

    uk_ctx->shire_mask = 0;
    auto & slot        = ggml_et_uberkernel_current_slot(uk_ctx);
    // Drain any prior launch on this slot before clearing its host buffers.
    // begin_graph and abort_graph both come through here; in either case we
    // must not yank the source memory out from under an in-flight DMA.
    ggml_et_uberkernel_slot_wait(slot, ggml_et_runtime());
    slot.insts.clear();
    slot.params_blob.clear();
}

static bool ggml_et_uberkernel_ensure_slot_capacity(ggml_backend_et_uberkernel_slot & slot,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_size,
                                                    size_t                            params_size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!dev_ctx || !runtime) {
        return false;
    }

    try {
        if (slot.device_insts == nullptr || insts_size > slot.device_insts_capacity) {
            const size_t new_capacity = ggml_et_next_capacity(slot.device_insts_capacity, insts_size);
            if (slot.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_insts);
            }
            slot.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_insts_capacity = slot.device_insts ? new_capacity : 0;
        }

        if (slot.device_params == nullptr || params_size > slot.device_params_capacity) {
            const size_t new_capacity = ggml_et_next_capacity(slot.device_params_capacity, params_size);
            if (slot.device_params) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_params);
            }
            slot.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_params_capacity = slot.device_params ? new_capacity : 0;
        }
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        return false;
    }

    return slot.device_insts != nullptr && slot.device_params != nullptr;
}

#define ET_TRACE_DECODER_IMPL
#include <et-trace/decoder.h>
#include <et-trace/layout.h>

// Get embedded kernel data by name
static std::vector<std::byte> ggml_et_get_embedded_kernel(const std::string & kernel_name) {
    auto it = ggml_et_embedded_kernels.find(kernel_name);
    if (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        return {};
    }

    const unsigned char * data = it->second.first;
    uint64_t              size = it->second.second;

    std::vector<std::byte> buffer(size);
    std::memcpy(buffer.data(), data, size);

    return buffer;
}

// Read kernel from file (for development/override)
static std::vector<std::byte> ggml_et_read_kernel_file(const std::string & kernel_path) {
    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> buffer(size);
    file.read(reinterpret_cast<char *>(buffer.data()), size);

    return buffer;
}

// Load kernel from file or embedded data
bool ggml_et_load_kernel(ggml_backend_et_device_context * dev_ctx, const std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        return false;
    }

    // Check if kernel already loaded
    if (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        return true;
    }

    std::vector<std::byte> kernel_data;
    const char *           kernels_path = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    if (kernels_path) {
        std::string kernel_file = std::string(kernels_path) + "/" + kernel_name + ".elf";
        kernel_data             = ggml_et_read_kernel_file(kernel_file);

        if (!kernel_data.empty()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } else {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    if (kernel_data.empty()) {
        kernel_data = ggml_et_get_embedded_kernel(kernel_name);
        if (kernel_data.empty()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            return false;
        }
    }

    try {
        // Load kernel code using device's default stream
        auto load_result = runtime->loadCode(dev_ctx->default_stream, kernel_data.data(), kernel_data.size());
        runtime->waitForEvent(load_result.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = load_result.kernel_;
        return true;

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        return false;
    }
}

// W23 mitigation (backlog 035): env-gated launch retry + counter, reported at exit when non-zero.
static const bool g_et_launch_retry = [] {
    const char * e = getenv("GGML_ET_LAUNCH_RETRY");
    return e && *e && strcmp(e, "0") != 0;
}();
static std::atomic<int> g_et_launch_retries{0};
// GGML_ET_LAUNCH_SYNC_EVERY=D: wait for the stream after every D-th launch (caps the host run-ahead that exhausts the
// firmware's dispatch slots — W23) without the per-kernel cost of GGML_ET_LAUNCH_RETRY. 0 = off.
static const int g_et_launch_sync_every = [] {
    const char * e = getenv("GGML_ET_LAUNCH_SYNC_EVERY");
    return e && *e ? atoi(e) : 0;
}();
static std::atomic<long> g_et_launch_count{0};
static const int g_et_launch_retry_report = [] {
    atexit([] {
        if (g_et_launch_retries.load() > 0) {
            fprintf(stderr, "ET-RETRY: %d kernel dispatch(es) were dropped by the firmware and re-issued in this process\n",
                    g_et_launch_retries.load());
        }
    });
    return 0;
}();

static bool ggml_et_launch_kernel_internal(ggml_backend_et_device_context * dev_ctx,
                                           const std::string &              kernel_name,
                                           void *                           params,
                                           size_t                           params_size,
                                           uint64_t                         shire_mask,
                                           bool                             enable_print,
                                           bool                             sync_error_check,
                                           rt::EventId *                    out_event = nullptr) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        return false;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    if (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        if (!ggml_et_load_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            return false;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        if (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            return false;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    try {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.setShireMask(shire_mask);  // Default: all shires (0xFFFFFFFF)
        k_opts.setBarrier(true);          // Wait for completion
        k_opts.setFlushL3(false);         // No L3 flush needed
        if (enable_print) {
            k_opts.setUserTracing(reinterpret_cast<uint64_t>(dev_ctx->trace_buffer),
                                  static_cast<uint32_t>(ET_TRACE_BUFFER_SIZE),
                                  0,                      // threshold
                                  shire_mask,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        if (sync_error_check) {
            ggml_et_wait_stream(dev_ctx, "launch");
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            if (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                for (const auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (int) error.errorCode_);
                    fprintf(stderr, "ET-SYNC-CHECK: error BEFORE kernel \"%s\": code=%d\n", kernel_name.c_str(),
                            (int) error.errorCode_);
                }
                fflush(stderr);
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_stream, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_size, k_opts);
        if (out_event) {
            *out_event = launch_event;
        }

        // GGML_ET_LAUNCH_RETRY=1 (W23 / backlog 035): the firmware intermittently drops a dispatch
        // (KernelLaunchUnexpectedError, no error context, kernel never ran — Exp 005). Wait for this launch and,
        // if the only stream errors are dropped dispatches, re-issue it once. Safe because the stream is drained
        // (nothing else in flight) and a dropped launch touched no memory. Costs one host round trip per kernel.
        if (g_et_launch_sync_every > 0 && !g_et_launch_retry && !sync_error_check &&
            (++g_et_launch_count % g_et_launch_sync_every) == 0) {
            ggml_et_wait_stream(dev_ctx, "launch");
        }
        if (g_et_launch_retry && !sync_error_check) {
            ggml_et_wait_stream(dev_ctx, "launch");
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            if (!errors.empty()) {
                bool only_dropped = true;
                for (const auto & error : errors) {
                    if (error.errorCode_ != rt::DeviceErrorCode::KernelLaunchUnexpectedError) {
                        only_dropped = false;
                    }
                }
                if (!only_dropped) {
                    for (const auto & error : errors) {
                        GGML_LOG_ERROR("ET: stream error after kernel \"%s\". Code: %d\n", kernel_name.c_str(),
                                       (int) error.errorCode_);
                        fprintf(stderr, "ET: stream error after kernel \"%s\": %s\n", kernel_name.c_str(),
                                error.getString().c_str());
                    }
                    fflush(stderr);
                    abort();
                }
                g_et_launch_retries++;
                fprintf(stderr, "ET-RETRY: dispatch of kernel \"%s\" dropped by the firmware (KernelLaunchUnexpectedError x%zu); "
                                "re-issuing once (total retries so far: %d)\n", kernel_name.c_str(), errors.size(),
                        g_et_launch_retries.load());
                fflush(stderr);
                launch_event = runtime->kernelLaunch(dev_ctx->default_stream, kernel_id,
                                                     reinterpret_cast<std::byte *>(params), params_size, k_opts);
                if (out_event) {
                    *out_event = launch_event;
                }
                ggml_et_wait_stream(dev_ctx, "launch");
                auto errors2 = runtime->retrieveStreamErrors(dev_ctx->default_stream);
                if (!errors2.empty()) {
                    for (const auto & error : errors2) {
                        fprintf(stderr, "ET-RETRY: kernel \"%s\" failed again after re-issue: code=%d %s\n",
                                kernel_name.c_str(), (int) error.errorCode_, error.getString().c_str());
                    }
                    fflush(stderr);
                    abort();
                }
            }
        }

        if (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_BUFFER_SIZE);
            runtime->memcpyDeviceToHost(dev_ctx->default_stream, dev_ctx->trace_buffer, host_trace_buf.data(),
                                        ET_TRACE_BUFFER_SIZE);
            ggml_et_wait_stream(dev_ctx, "launch");
            const auto * trace_header = reinterpret_cast<const trace_buffer_std_header_t *>(host_trace_buf.data());
            const trace_entry_header_t * entry = nullptr;
            while ((entry = Trace_Decode(trace_header, entry))) {
                if (entry->type != TRACE_TYPE_STRING) {
                    continue;
                }
                const auto * str_entry = reinterpret_cast<const trace_string_t *>(entry);
                printf("[hart %d] %s", entry->hart_id, str_entry->string);
            }
        }

        if (sync_error_check) {
            // Already triggered. No need to retrigger
            if (!enable_print) {
                ggml_et_wait_stream(dev_ctx, "launch");
            }
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            if (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                for (const auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (int) error.errorCode_);
                    int ctx_type = -1;
                    if (error.errorContext_.has_value() && !error.errorContext_.value().empty()) {
                        ctx_type = (int) error.errorContext_.value()[0].type_;
                    }
                    fprintf(stderr, "ET-SYNC-CHECK: error DURING kernel \"%s\": code=%d ctx_type=%d\n  runtime: %s\n",
                            kernel_name.c_str(), (int) error.errorCode_, ctx_type, error.getString().c_str());
                }
                fflush(stderr);
                abort();
            }
        }

        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        return false;
    }
}

void ggml_et_uberkernel_begin_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    if (!uk_ctx) {
        return;
    }

    uk_ctx->failed = false;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

static bool ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context *     dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    if (!uk_ctx || !dev_ctx) {
        return false;
    }

    auto & slot = ggml_et_uberkernel_current_slot(uk_ctx);
    if (slot.insts.empty()) {
        return true;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = true;
        return false;
    }

    const size_t   insts_size  = slot.insts.size() * sizeof(ggml_et_uberkernel_inst);
    const size_t   params_size = slot.params_blob.size();
    const uint64_t shire_mask  = uk_ctx->shire_mask;
    bool           ok          = false;

    try {
        if (!ggml_et_uberkernel_ensure_slot_capacity(slot, dev_ctx, insts_size, params_size)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = true;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            slot.insts.clear();
            slot.params_blob.clear();
            uk_ctx->shire_mask = 0;
            return false;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpyHostToDevice(dev_ctx->default_stream, reinterpret_cast<const std::byte *>(slot.insts.data()),
                                    slot.device_insts, insts_size, true);
        runtime->memcpyHostToDevice(dev_ctx->default_stream, slot.params_blob.data(), slot.device_params, params_size,
                                    true);

        static const bool uk_prof_on = [] {
            const char * e = getenv("GGML_ET_UK_PROFILE");
            return e && *e && strcmp(e, "0") != 0;
        }();
        const size_t prof_size = (slot.insts.size() + 1) * sizeof(uint64_t);
        if (uk_prof_on && (slot.device_prof == nullptr || prof_size > slot.device_prof_capacity)) {
            const size_t new_capacity = ggml_et_next_capacity(slot.device_prof_capacity, prof_size);
            if (slot.device_prof) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_prof);
            }
            slot.device_prof          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_prof_capacity = slot.device_prof ? new_capacity : 0;
        }
        static const uint32_t uk_flags = [] {
            // Default mode 1 (whole-L1 evict, then the L2->L3 walk split across the shire): with every consumer case
            // evicting its inputs, mode 0 has every hart walk every input span and is slower than no uberkernel.
            const char * e = getenv("GGML_ET_UK_EVICT_MODE");
            const char * t = getenv("GGML_ET_UK_TIMING_FLAGS");  // bits 1-2, attribution runs only (uberkernel.c)
            return ((uint32_t) ((e && *e) ? strtoul(e, nullptr, 0) : 1u) & 1u) |
                   ((uint32_t) ((t && *t) ? strtoul(t, nullptr, 0) : 0u) & 6u);
        }();
        const bool prof_this = uk_prof_on && slot.device_prof != nullptr;
        static const std::vector<uint64_t> poison(1 << 16, ~0ull);
        if (prof_this && prof_size <= poison.size() * sizeof(uint64_t)) {
            runtime->memcpyHostToDevice(dev_ctx->default_stream, reinterpret_cast<const std::byte *>(poison.data()),
                                        slot.device_prof, prof_size, true);
        }
        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(slot.insts.size()),
            static_cast<uint32_t>(sizeof(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(slot.device_insts),
            reinterpret_cast<uint64_t>(slot.device_params),
            uk_flags | (prof_this ? 8u : 0u),
            0u,
            prof_this ? reinterpret_cast<uint64_t>(slot.device_prof) : 0ull,
        };

        rt::EventId launch_event{};
        g_uk_stats.segments++;
        ok = ggml_et_launch_kernel_internal(dev_ctx, "uberkernel", &params, sizeof(params), shire_mask, false, false,
                                            &launch_event);
        if (ok && prof_this) {
            // Serialises the segment (profiling only): read the stamps back in stream order and attribute them.
            std::vector<uint64_t> stamps(slot.insts.size() + 1);
            rt::EventId ev = runtime->memcpyDeviceToHost(dev_ctx->default_stream, slot.device_prof,
                                                         reinterpret_cast<std::byte *>(stamps.data()), prof_size, true);
            runtime->waitForEvent(ev);
            bool sane = true;
            for (size_t i = 0; i < stamps.size() && sane; ++i) {
                sane = stamps[i] != ~0ull && (i == 0 || stamps[i] > stamps[i - 1]);
            }
            if (!sane) {
                g_uk_prof_bad++;
            } else {
                for (size_t i = 0; i < slot.insts.size(); ++i) {
                    std::string key = ggml_et_uberkernel_kernel_name_from_id(slot.insts[i].kernel_id);
                    if (key.rfind("mul_mat_", 0) == 0 && key.find("_id_") == std::string::npos &&
                        slot.insts[i].params_size >= sizeof(ggml_et_mm_q8_params)) {
                        // matmul kernels share the mm_q8 params layout (src0 first): key the row by [M,K,N]
                        const auto * mp = reinterpret_cast<const ggml_et_mm_q8_params *>(slot.params_blob.data() +
                                                                                           slot.insts[i].params_offset);
                        key += "[" + std::to_string(mp->src0.ne[1]) + "," + std::to_string(mp->src0.ne[0]) + "," +
                               std::to_string(mp->src1.ne[1]) + "]";
                    } else if (key == "get_rows_f32" && slot.insts[i].params_size >= sizeof(ggml_et_get_rows_params)) {
                        // key the row by [table rows x row width, rows gathered]
                        const auto * gp = reinterpret_cast<const ggml_et_get_rows_params *>(slot.params_blob.data() +
                                                                                             slot.insts[i].params_offset);
                        key += "[" + std::to_string(gp->src0.ne[1]) + "x" + std::to_string(gp->src0.ne[0]) + "," +
                               std::to_string(gp->src1.ne[0] * gp->src1.ne[1] * gp->src1.ne[2]) + "]";
                    }
                    auto & acc = g_uk_prof[key];
                    acc.first++;
                    acc.second += stamps[i + 1] - stamps[i];
                }
            }
        }
        if (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            slot.pending_event = launch_event;
            slot.has_pending   = true;
        }
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    if (ok) {
        uk_ctx->current_slot = (uk_ctx->current_slot + 1) % ggml_backend_et_uberkernel_context::SLOT_COUNT;
        auto & next          = ggml_et_uberkernel_current_slot(uk_ctx);
        ggml_et_uberkernel_slot_wait(next, runtime);
        next.insts.clear();
        next.params_blob.clear();
    } else {
        slot.insts.clear();
        slot.params_blob.clear();
    }
    uk_ctx->shire_mask = 0;
    return ok;
}

void ggml_et_uberkernel_abort_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    if (!uk_ctx) {
        return;
    }

    uk_ctx->failed = false;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

bool ggml_et_uberkernel_failed(const ggml_backend_et_uberkernel_context * uk_ctx) {
    return uk_ctx && uk_ctx->failed;
}

// Prepend the staged graph inputs to the (empty) current segment as one GGML_ET_UBERKERNEL_INPUT_COPY instruction.
bool ggml_et_uberkernel_add_input_copy(ggml_backend_et_device_context * dev_ctx, const ggml_et_input_copy_src * ops,
                                       size_t n_ops) {
    if (!dev_ctx || !dev_ctx->uberkernel_active || n_ops == 0) {
        return false;
    }
    ggml_backend_et_uberkernel_context * uk_ctx = &dev_ctx->uberkernel;
    auto &                               slot   = ggml_et_uberkernel_current_slot(uk_ctx);
    if (!slot.insts.empty()) {
        return false;  // the copy must be the segment's first instruction
    }
    const size_t head = sizeof(ggml_et_uk_input_copy_hdr) + n_ops * sizeof(ggml_et_uk_input_copy_job);
    size_t       data = ggml_et_align_up(head, 64);
    std::vector<std::byte> blob(data);
    auto * hdr   = reinterpret_cast<ggml_et_uk_input_copy_hdr *>(blob.data());
    hdr->n_jobs  = (uint32_t) n_ops;
    hdr->reserved = 0;
    for (size_t i = 0; i < n_ops; ++i) {
        if (ops[i].size % 4 != 0 || ((uintptr_t) ops[i].dst % 4) != 0) {
            return false;
        }
        const size_t off = blob.size();
        blob.insert(blob.end(), ops[i].src, ops[i].src + ops[i].size);
        blob.resize(ggml_et_align_up(blob.size(), 64));
        auto * job    = reinterpret_cast<ggml_et_uk_input_copy_job *>(blob.data() + sizeof(ggml_et_uk_input_copy_hdr)) + i;
        job->dst      = (uint64_t) (uintptr_t) ops[i].dst;
        job->src_off  = (uint32_t) off;
        job->words    = (uint32_t) (ops[i].size / 4);
    }
    const size_t params_offset = ggml_et_align_up(slot.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    slot.params_blob.resize(params_offset);
    slot.params_blob.insert(slot.params_blob.end(), blob.begin(), blob.end());
    ggml_et_uberkernel_inst inst = { (uint16_t) GGML_ET_UBERKERNEL_INPUT_COPY, 0, (uint32_t) params_offset,
                                     (uint32_t) blob.size() };
    slot.insts.push_back(inst);
    uk_ctx->shire_mask = 0xFFFFFFFFULL;
    g_uk_stats.insts++;
    return true;
}

static bool ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      const std::string &              kernel_name,
                                      void *                           params,
                                      size_t                           params_size,
                                      uint64_t                         shire_mask,
                                      bool                             enable_print,
                                      bool                             sync_error_check) {
    if (!dev_ctx) {
        return false;
    }

    ggml_backend_et_uberkernel_context * uk_ctx        = &dev_ctx->uberkernel;
    uint16_t                             uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    // GGML_ET_UBERKERNEL_EXCLUDE="softmax_f32,scale_f32": launch the listed kernels standalone (flushing the
    // pending segment first) instead of batching them. Bisect aid for inter-instruction visibility bugs.
    {
        static const std::vector<std::string> excluded = [] {
            std::vector<std::string> v;
            const char * e = getenv("GGML_ET_UBERKERNEL_EXCLUDE");
            if (e) {
                std::string cur;
                for (const char * c = e;; ++c) {
                    if (*c == ',' || *c == '\0') {
                        if (!cur.empty()) { v.push_back(cur); }
                        cur.clear();
                        if (*c == '\0') { break; }
                    } else if (*c != ' ') {
                        cur.push_back(*c);
                    }
                }
            }
            return v;
        }();
        for (const auto & x : excluded) {
            if (x == kernel_name) {
                uberkernel_id = GGML_ET_UBERKERNEL_KERNEL_INVALID;
                break;
            }
        }
        // 052: GGML_ET_UBERKERNEL_ONLY="moe_router_f32,moe_sum_f32": batch only the listed kernels (bisect aid;
        // the complement of EXCLUDE). Empty/unset = no restriction.
        static const std::vector<std::string> only = [] {
            std::vector<std::string> v;
            const char * e = getenv("GGML_ET_UBERKERNEL_ONLY");
            if (e) {
                std::string cur;
                for (const char * c = e;; ++c) {
                    if (*c == ',' || *c == '\0') {
                        if (!cur.empty()) { v.push_back(cur); }
                        cur.clear();
                        if (*c == '\0') { break; }
                    } else if (*c != ' ') {
                        cur.push_back(*c);
                    }
                }
            }
            return v;
        }();
        if (!only.empty() && uberkernel_id != GGML_ET_UBERKERNEL_KERNEL_INVALID) {
            bool listed = false;
            for (const auto & x : only) { if (x == kernel_name) { listed = true; break; } }
            if (!listed) { uberkernel_id = GGML_ET_UBERKERNEL_KERNEL_INVALID; }
        }
    }
    // The dispatcher's per-instruction barrier is chip-wide (32 shires): a narrowed launch (GGML_ET_MMID_SHIRE_MASK)
    // must never enter a segment.
    if (shire_mask != 0xFFFFFFFFULL) {
        uberkernel_id = GGML_ET_UBERKERNEL_KERNEL_INVALID;
    }
    if (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        if (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            return false;
        }
        g_uk_stats.standalone++;
        return ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }
    g_uk_stats.insts++;

    auto &       slot          = ggml_et_uberkernel_current_slot(uk_ctx);
    const size_t params_offset = ggml_et_align_up(slot.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    if (params_offset > slot.params_blob.size()) {
        slot.params_blob.resize(params_offset);
    }

    const std::byte * params_bytes = reinterpret_cast<const std::byte *>(params);
    slot.params_blob.insert(slot.params_blob.end(), params_bytes, params_bytes + params_size);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_offset),
        static_cast<uint32_t>(params_size),
    };
    slot.insts.push_back(inst);

    if (slot.insts.size() == 1) {
        uk_ctx->shire_mask = shire_mask;
    }

        if (sync_error_check) {
        // GGML_ET_SYNC_CHECK: commit the segment now (this op is its last instruction) and attribute any
        // device error to this kernel, mirroring the standalone-launch path.
        if (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            return false;
        }
        std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
        if (runtime) {
            ggml_et_wait_stream(dev_ctx, "launch");
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            if (!errors.empty()) {
                for (const auto & error : errors) {
                    int ctx_type = -1;
                    if (error.errorContext_.has_value() && !error.errorContext_.value().empty()) {
                        ctx_type = (int) error.errorContext_.value()[0].type_;
                    }
                    fprintf(stderr, "ET-SYNC-CHECK: error DURING uberkernel op \"%s\": code=%d ctx_type=%d\n  runtime: %s\n",
                            kernel_name.c_str(), (int) error.errorCode_, ctx_type, error.getString().c_str());
                }
                fflush(stderr);
                abort();
            }
        }
    }
    return true;
}

bool ggml_et_uberkernel_end_graph(ggml_backend_et_device_context * dev_ctx) {
    if (!dev_ctx || !dev_ctx->uberkernel_active) {
        return true;
    }

    const bool ok = ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
    if (g_uk_stats_on) {
        g_uk_stats.graphs++;
        fprintf(stderr, "ET-UK-STATS graph=%llu segments=%llu batched_insts=%llu standalone=%llu\n",
                (unsigned long long) g_uk_stats.graphs, (unsigned long long) g_uk_stats.segments,
                (unsigned long long) g_uk_stats.insts, (unsigned long long) g_uk_stats.standalone);
        g_uk_stats.segments = g_uk_stats.insts = g_uk_stats.standalone = 0;
    }
    if (!g_uk_prof.empty()) {
        // One table per graph (GGML_ET_UK_PROFILE): instruction i = its body on the slowest hart + the next
        // instruction's evicts and chip-wide barrier; cycles are hpmcounter3 at 600 MHz.
        std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> rows(g_uk_prof.begin(), g_uk_prof.end());
        std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) { return a.second.second > b.second.second; });
        uint64_t total = 0;
        for (const auto & r : rows) { total += r.second.second; }
        fprintf(stderr, "ET-UK-PROF total_cycles=%llu total_ms@600MHz=%.2f bad_segments=%llu\n", (unsigned long long) total,
                (double) total / 600e3, (unsigned long long) g_uk_prof_bad);
        for (const auto & r : rows) {
            fprintf(stderr, "ET-UK-PROF %-40s n=%6llu mean_cycles=%10llu mean_us=%8.1f total_ms=%8.2f\n",
                    r.first.c_str(), (unsigned long long) r.second.first,
                    (unsigned long long) (r.second.second / r.second.first),
                    (double) r.second.second / 600.0 / (double) r.second.first, (double) r.second.second / 600e3);
        }
        g_uk_prof.clear();
        g_uk_prof_bad = 0;
    }
    return ok;
}

bool ggml_et_wait_stream(ggml_backend_et_device_context * dev_ctx, const char * where) {
    static const int timeout_s = [] { const char * e = getenv("GGML_ET_WAIT_TIMEOUT"); return e ? atoi(e) : 0; }();
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        return false;
    }
    if (timeout_s <= 0) {
        runtime->waitForStream(dev_ctx->default_stream);
        return true;
    }
    if (runtime->waitForStream(dev_ctx->default_stream, std::chrono::seconds(timeout_s))) {
        return true;
    }
    fprintf(stderr, "ET: waitForStream timed out after %d s at %s — hung kernel; aborting the stream\n", timeout_s, where);
    fflush(stderr);
    try {
        rt::EventId ev = runtime->abortStream(dev_ctx->default_stream);
        const bool  ok = runtime->waitForEvent(ev, std::chrono::seconds(10));
        fprintf(stderr, "ET: abortStream %s\n", ok ? "completed" : "did not complete within 10 s");
        auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
        for (const auto & err : errors) {
            fprintf(stderr, "ET: stream error after abort: %s\n", err.getString().c_str());
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "ET: abortStream threw: %s\n", e.what());
    }
    fflush(stderr);
    abort();
}

bool ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           const std::string &              kernel_name,
                           void *                           params,
                           size_t                           params_size,
                           uint64_t                         shire_mask,
                           bool                             enable_print,
                           bool                             sync_error_check) {
    if (!dev_ctx) {
        return false;
    }

    // GGML_ET_SYNC_CHECK=1: wait after every kernel and attribute device stream errors to the kernel that
    // raised them (debug aid; serialises launches). Without it a device exception surfaces only at the next
    // synchronize() as an anonymous "stream error" abort.
    {
        static const bool force_sync = [] {
            const char * e = getenv("GGML_ET_SYNC_CHECK");
            return e && *e && strcmp(e, "0") != 0;
        }();
        sync_error_check = sync_error_check || force_sync;
    }

    if (!dev_ctx->uberkernel_active) {
        return ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    return ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                     sync_error_check);
}

void ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, const std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        return;
    }

    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    if (kernel_it != dev_ctx->loaded_kernels.end()) {
        try {
            runtime->unloadCode(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

void ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx) {
    if (!dev_ctx) {
        return;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_names;
    kernel_names.reserve(dev_ctx->loaded_kernels.size());
    for (const auto & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_names.push_back(kernel_pair.first);
    }

    for (const auto & kernel_name : kernel_names) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    for (const auto & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    return loaded_kernels;
}
