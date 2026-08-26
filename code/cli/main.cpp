#include "lm/lm.h"

#include <cstdio>
#include <cstring>

static void print_help() {
    std::puts("tiny-lm control-plane slice");
    std::puts("usage: tiny-lm [options]");
    std::puts("  --backend auto|cpu|vulkan|rocr|rocm|cuda|openvino|directml");
    std::puts("  --model PATH --context N --threads N --device N");
    std::puts("  --load eager|mmap|lazy|stream --kv-dtype f16|bf16|q8|q6|q4");
    std::puts("  --kv-page-tokens N --trace --deterministic --no-prefetch");
    std::puts("  --dump-config --dry-run --list-devices --help");
}

static void text_probe(void *, const lm_probe *probe) {
    std::printf("probe trace=%llu stage=%u kind=%u bytes=%u\n",
                static_cast<unsigned long long>(probe->trace_id), probe->stage,
                probe->kind, probe->bytes);
}

int main(int argc, char **argv) {
    lm_config config;
    lm_config_init(&config);
    const char *bad = nullptr;
    const lm_status parsed = lm_config_parse_argv(&config, argc, argv, &bad);
    if (parsed != LM_OK) {
        std::fprintf(stderr, "configuration error: %s (%s)\n",
                     lm_status_name(parsed), bad ? bad : "unknown");
        return 2;
    }
    bool list_devices = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_help();
            return 0;
        }
        if (std::strcmp(argv[i], "--list-devices") == 0) list_devices = true;
    }
    if (list_devices) {
        uint32_t count = 0u;
        const lm_status status = lm_vulkan_device_count(&count);
        if (status != LM_OK) {
            std::fprintf(stderr, "device discovery failed: %s\n", lm_status_name(status));
            return 4;
        }
        for (uint32_t i = 0u; i < count; ++i) {
            lm_vulkan_device_info info{};
            if (lm_vulkan_device_info_get(i, &info) != LM_OK) return 4;
            std::printf("device=%u name=%s api=%u vendor=0x%08x shader_int_dot=%u subgroup=%u cpu=%u\n",
                        i, info.name, info.api_version, info.vendor_id,
                        info.shader_int_dot, info.subgroup, info.is_cpu);
        }
        if (config.backend == LM_BACKEND_AUTO && !config.model_path[0]) return 0;
    }

    lm_runtime *runtime = nullptr;
    const lm_status created = lm_runtime_create(&config, &runtime);
    if (created != LM_OK) {
        std::fprintf(stderr, "backend selection failed: %s; requested backend is not built in this slice\n",
                     lm_status_name(created));
        return 3;
    }
    if (config.trace) lm_runtime_set_probe_sink(runtime, text_probe, nullptr);
    (void)lm_runtime_dump_config(runtime, stdout);
    if (config.trace) {
        lm_probe probe{};
        probe.trace_id = 1u;
        probe.kind = 1u;
        probe.stage = 1u;
        probe.bytes = static_cast<uint32_t>(std::strlen(config.model_path));
        (void)lm_runtime_emit_probe(runtime, &probe);
    }
    std::puts("status=control-plane-ready");
    std::puts("inference=not-enabled-in-this-vertical-slice");
    lm_runtime_destroy(runtime);
    return 0;
}
