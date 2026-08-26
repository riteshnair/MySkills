#include "lm/lm.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>

struct ProbeState { unsigned count; uint32_t last_stage; };

static void capture_probe(void *user, const lm_probe *probe) {
    ProbeState *state = static_cast<ProbeState *>(user);
    ++state->count;
    state->last_stage = probe->stage;
}

static void test_defaults() {
    lm_config config;
    lm_config_init(&config);
    assert(config.backend == LM_BACKEND_AUTO);
    assert(config.load_mode == LM_LOAD_MMAP);
    assert(config.context_tokens == 4096u);
    assert(config.kv_page_tokens == 32u);
    assert(config.threads == 1u);
}

static void test_cli() {
    char a0[] = "tiny-lm";
    char a1[] = "--backend"; char a2[] = "cpu";
    char a3[] = "--load"; char a4[] = "lazy";
    char a5[] = "--kv-dtype"; char a6[] = "q8";
    char a7[] = "--context"; char a8[] = "2048";
    char a9[] = "--threads"; char a10[] = "4";
    char a11[] = "--trace"; char a12[] = "--model"; char a13[] = "model.gguf";
    char *argv[] = {a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13};
    lm_config config;
    lm_config_init(&config);
    const char *bad = nullptr;
    assert(lm_config_parse_argv(&config, 14, argv, &bad) == LM_OK);
    assert(config.backend == LM_BACKEND_CPU);
    assert(config.load_mode == LM_LOAD_LAZY);
    assert(config.kv_dtype == LM_KV_Q8);
    assert(config.context_tokens == 2048u);
    assert(config.threads == 4u);
    assert(config.trace == 1u);
    assert(std::strcmp(config.model_path, "model.gguf") == 0);
}

static void test_invalid_backend() {
    char a0[] = "tiny-lm"; char a1[] = "--backend"; char a2[] = "vulkan";
    char *argv[] = {a0,a1,a2};
    lm_config config;
    lm_config_init(&config);
    const char *bad = nullptr;
    assert(lm_config_parse_argv(&config, 3, argv, &bad) == LM_OK);
    lm_runtime *runtime = nullptr;
    assert(lm_runtime_create(&config, &runtime) == LM_ERR_UNSUPPORTED);
    assert(runtime == nullptr);
}

static void test_model_and_cpu_math() {
    const char *gguf = "test-fixture.gguf";
    {
        std::ofstream file(gguf, std::ios::binary);
        const unsigned char header[] = {
            'G','G','U','F', 3,0,0,0,
            1,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0
        };
        file.write(reinterpret_cast<const char *>(header), sizeof(header));
    }
    lm_model_info info;
    char error[128] = {};
    assert(lm_model_inspect(gguf, &info, error, sizeof(error)) == LM_OK);
    assert(info.format == LM_MODEL_GGUF && info.version == 3u && info.tensor_count == 1u);
    std::remove(gguf);

    const float a[] = {1.0f, 2.0f, 3.0f};
    const float b[] = {4.0f, 5.0f, 6.0f};
    float dot = 0.0f;
    assert(lm_cpu_dot_f32(a, b, 3u, &dot) == LM_OK);
    assert(dot == 32.0f);
    const float logits[] = {0.0f, 1.0f, 2.0f};
    float probabilities[3] = {};
    assert(lm_cpu_softmax_f32(logits, probabilities, 3u) == LM_OK);
    assert(probabilities[2] > probabilities[1] && probabilities[1] > probabilities[0]);
    assert(probabilities[0] + probabilities[1] + probabilities[2] > 0.999f);
}

static void test_kv_pages() {
    lm_kv_cache *cache = nullptr;
    assert(lm_kv_cache_create(2u, 4u, &cache) == LM_OK);
    uint32_t first = UINT32_MAX;
    assert(lm_kv_cache_append(cache, &first, 3u) == LM_OK);
    uint32_t second = UINT32_MAX;
    assert(lm_kv_cache_fork(cache, first, &second) == LM_OK);
    assert(lm_kv_cache_append(cache, &second, 1u) == LM_OK);
    assert(lm_kv_cache_append(cache, &second, 1u) == LM_ERR_CAPACITY);
    lm_kv_stats stats;
    assert(lm_kv_cache_get_stats(cache, &stats) == LM_OK);
    assert(stats.used_pages == 2u && stats.free_pages == 0u && stats.shared_pages == 0u);
    assert(lm_kv_cache_rollback(cache, second, 1u) == LM_OK);
    assert(lm_kv_cache_release(cache, first) == LM_OK);
    assert(lm_kv_cache_release(cache, second) == LM_OK);
    assert(lm_kv_cache_get_stats(cache, &stats) == LM_OK && stats.free_pages == 2u);
    lm_kv_cache_destroy(cache);
}

static void test_probe_and_runtime() {
    lm_config config;
    lm_config_init(&config);
    config.trace = 1u;
    lm_runtime *runtime = nullptr;
    assert(lm_runtime_create(&config, &runtime) == LM_OK);
    ProbeState state{0u, 0u};
    lm_runtime_set_probe_sink(runtime, capture_probe, &state);
    lm_probe probe{};
    probe.trace_id = 17u;
    probe.stage = 9u;
    assert(lm_runtime_emit_probe(runtime, &probe) == LM_OK);
    assert(state.count == 1u && state.last_stage == 9u);
    assert(lm_runtime_dump_config(runtime, stdout) == LM_OK);
    lm_runtime_destroy(runtime);
}

int main() {
    test_defaults();
    test_cli();
    test_invalid_backend();
    test_model_and_cpu_math();
    test_kv_pages();
    test_probe_and_runtime();
    std::puts("core_tests=PASS");
    return 0;
}
