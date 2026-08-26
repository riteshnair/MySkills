#include "lm/lm.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

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
    char a5[] = "--weights"; char a6[] = "preserve";
    char a7[] = "--kv-dtype"; char a8[] = "q8";
    char a9[] = "--context"; char a10[] = "2048";
    char a11[] = "--threads"; char a12[] = "4";
    char a13[] = "--trace"; char a14[] = "--model"; char a15[] = "model.gguf";
    char *argv[] = {a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15};
    lm_config config;
    lm_config_init(&config);
    const char *bad = nullptr;
    assert(lm_config_parse_argv(&config, 16, argv, &bad) == LM_OK);
    assert(config.backend == LM_BACKEND_CPU);
    assert(config.load_mode == LM_LOAD_LAZY);
    assert(config.weight_policy == LM_WEIGHT_PRESERVE);
    assert(config.kv_dtype == LM_KV_Q8);
    assert(config.context_tokens == 2048u);
    assert(config.threads == 4u);
    assert(config.trace == 1u);
    assert(std::strcmp(config.model_path, "model.gguf") == 0);
}

static void test_quantize_policy_gate() {
    lm_config config;
    lm_config_init(&config);
    config.weight_policy = LM_WEIGHT_QUANTIZE_CACHE;
    lm_runtime *runtime = nullptr;
    assert(lm_runtime_create(&config, &runtime) == LM_ERR_UNSUPPORTED);
    assert(runtime == nullptr);
}

static void test_vulkan_backend_resolution() {
    char a0[] = "tiny-lm"; char a1[] = "--backend"; char a2[] = "vulkan";
    char *argv[] = {a0,a1,a2};
    lm_config config;
    lm_config_init(&config);
    const char *bad = nullptr;
    assert(lm_config_parse_argv(&config, 3, argv, &bad) == LM_OK);
    lm_runtime *runtime = nullptr;
    uint32_t device_count = 0u;
    const lm_status discovered = lm_vulkan_device_count(&device_count);
    const lm_status created = lm_runtime_create(&config, &runtime);
    if (discovered == LM_OK && device_count > 0u) {
        assert(created == LM_OK && runtime != nullptr);
        lm_runtime_destroy(runtime);
    } else {
        assert(created == LM_ERR_UNSUPPORTED && runtime == nullptr);
    }
}

static void test_tensor_and_buffer() {
    float values[6] = {};
    const uint32_t dims[2] = {2u, 3u};
    lm_tensor tensor{};
    assert(lm_tensor_make_view(values, sizeof(values), LM_DTYPE_F32, 2u, dims, &tensor) == LM_OK);
    assert(tensor.strides[0] == 3u && tensor.strides[1] == 1u);
    lm_tensor bad = tensor;
    bad.bytes = 4u;
    assert(lm_tensor_validate(&bad) == LM_ERR_CAPACITY);
    lm_buffer *buffer = nullptr;
    assert(lm_buffer_alloc(64u, &buffer) == LM_OK);
    lm_tensor buffer_view{};
    assert(lm_buffer_view(buffer, &buffer_view) == LM_OK);
    assert(buffer_view.bytes == 64u && buffer_view.dtype == LM_DTYPE_U8);
    unsigned char quantized_bytes[20] = {};
    const uint32_t quant_dims[1] = {32u};
    lm_tensor quantized{};
    assert(lm_tensor_make_quant_view(quantized_bytes, sizeof(quantized_bytes), 1u, quant_dims, 32u, 20u, &quantized) == LM_OK);
    assert(quantized.quant_format == LM_QUANT_BLOCK_STORAGE && quantized.dtype == LM_DTYPE_U8);
    assert(lm_tensor_make_quant_view(quantized_bytes, 19u, 1u, quant_dims, 32u, 20u, &quantized) == LM_ERR_CAPACITY);
    unsigned char q8_bytes[34] = {};
    q8_bytes[0] = 0x00u;
    q8_bytes[1] = 0x3cu;
    for (unsigned i = 0u; i < 32u; ++i) q8_bytes[2u + i] = static_cast<unsigned char>(i + 1u);
    lm_tensor q8_weights{};
    assert(lm_tensor_make_q8_0_view(q8_bytes, sizeof(q8_bytes), 1u, quant_dims, &q8_weights) == LM_OK);
    const float q8_input[32] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float q8_result = 0.0f;
    assert(lm_cpu_dot_q8_0(&q8_weights, q8_input, 32u, &q8_result) == LM_OK);
    assert(q8_result == 528.0f);
    assert(lm_tensor_make_q8_0_view(q8_bytes, 33u, 1u, quant_dims, &q8_weights) == LM_ERR_CAPACITY);
    unsigned char q4_bytes[18] = {};
    q4_bytes[0] = 0x00u;
    q4_bytes[1] = 0x3cu;
    for (unsigned i = 0u; i < 16u; ++i) q4_bytes[2u + i] = 0x99u;
    lm_tensor q4_weights{};
    assert(lm_tensor_make_q4_0_view(q4_bytes, sizeof(q4_bytes), 1u, quant_dims, &q4_weights) == LM_OK);
    float q4_result = 0.0f;
    assert(lm_cpu_dot_q4_0(&q4_weights, q8_input, 32u, &q4_result) == LM_OK);
    assert(q4_result == 32.0f);
    assert(lm_tensor_make_q4_0_view(q4_bytes, 17u, 1u, quant_dims, &q4_weights) == LM_ERR_CAPACITY);
    lm_buffer_free(buffer);
}

static void test_file_access() {
    const char *path = "test-file.bin";
    {
        std::ofstream file(path, std::ios::binary);
        const char bytes[] = "0123456789";
        file.write(bytes, sizeof(bytes) - 1u);
    }
    lm_file *file = nullptr;
    assert(lm_file_open(path, &file) == LM_OK);
    uint64_t size = 0u;
    assert(lm_file_size(file, &size) == LM_OK && size == 10u);
    char readback[5] = {};
    assert(lm_file_read(file, 3u, readback, 4u) == LM_OK);
    assert(std::memcmp(readback, "3456", 4u) == 0);
    lm_file_span span{};
    assert(lm_file_span_make(file, 2u, 5u, &span) == LM_OK);
    std::memset(readback, 0, sizeof(readback));
    assert(lm_file_span_read(&span, 1u, readback, 4u) == LM_OK);
    assert(std::memcmp(readback, "3456", 4u) == 0);
    assert(lm_file_span_read(&span, 2u, readback, 4u) == LM_ERR_RANGE);
    assert(lm_file_read(file, size, nullptr, 0u) == LM_OK);
    assert(lm_file_read(file, 8u, readback, 4u) == LM_ERR_RANGE);
    lm_file_close(file);
    std::remove(path);
}

static void test_native_model_tensor_binding() {
    auto put_u32 = [](std::vector<unsigned char> *bytes, uint32_t value) {
        for (unsigned i = 0u; i < 4u; ++i) bytes->push_back(static_cast<unsigned char>((value >> (8u * i)) & 0xffu));
    };
    auto put_u64 = [](std::vector<unsigned char> *bytes, uint64_t value) {
        for (unsigned i = 0u; i < 8u; ++i) bytes->push_back(static_cast<unsigned char>((value >> (8u * i)) & 0xffu));
    };
    auto put_descriptor = [&put_u32, &put_u64](std::vector<unsigned char> *bytes, const char *name,
                                                uint32_t type, uint64_t offset) {
        const size_t length = std::strlen(name);
        put_u64(bytes, length);
        bytes->insert(bytes->end(), name, name + length);
        put_u32(bytes, 1u);
        put_u64(bytes, 32u);
        put_u32(bytes, type);
        put_u64(bytes, offset);
    };
    std::vector<unsigned char> gguf(24u, 0u);
    gguf[0] = 'G'; gguf[1] = 'G'; gguf[2] = 'U'; gguf[3] = 'F';
    gguf[4] = 3u;
    gguf[8] = 2u;
    put_descriptor(&gguf, "q4", 2u, 0u);
    put_descriptor(&gguf, "q8", 8u, 32u);
    const size_t header_size = (gguf.size() + 31u) & ~static_cast<size_t>(31u);
    gguf.resize(header_size + 32u + 34u, 0u);
    gguf[header_size] = 0x00u;
    gguf[header_size + 1u] = 0x3cu;
    for (unsigned i = 0u; i < 16u; ++i) gguf[header_size + 2u + i] = 0x99u;
    gguf[header_size + 32u] = 0x00u;
    gguf[header_size + 33u] = 0x3cu;
    for (unsigned i = 0u; i < 32u; ++i) gguf[header_size + 34u + i] = static_cast<unsigned char>(i + 1u);
    const char *path = "test-native-binding.gguf";
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(gguf.data()), static_cast<std::streamsize>(gguf.size()));
    }
    lm_model_file *model = nullptr;
    char error[128] = {};
    assert(lm_model_open(path, &model, error, sizeof(error)) == LM_OK);
    lm_model_tensor_binding q4_binding{};
    assert(lm_model_tensor_bind_native(model, 0u, &q4_binding) == LM_OK);
    assert(q4_binding.elements == 32u && q4_binding.span.bytes == 18u &&
           q4_binding.quant_format == LM_QUANT_GGML_Q4_0);
    unsigned char q4_bytes[18] = {};
    lm_tensor q4_tensor{};
    assert(lm_model_tensor_binding_read(&q4_binding, q4_bytes, sizeof(q4_bytes), &q4_tensor) == LM_OK);
    const float input[32] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                             1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                             1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                             1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float result = 0.0f;
    assert(lm_cpu_dot_q4_0(&q4_tensor, input, 32u, &result) == LM_OK && result == 32.0f);
    lm_model_tensor_binding q8_binding{};
    assert(lm_model_tensor_bind_native(model, 1u, &q8_binding) == LM_OK);
    assert(q8_binding.elements == 32u && q8_binding.span.offset == q4_binding.span.offset + 32u &&
           q8_binding.span.bytes == 34u && q8_binding.quant_format == LM_QUANT_GGML_Q8_0);
    unsigned char q8_bytes[34] = {};
    lm_tensor q8_tensor{};
    assert(lm_model_tensor_binding_read(&q8_binding, q8_bytes, sizeof(q8_bytes), &q8_tensor) == LM_OK);
    assert(lm_cpu_dot_q8_0(&q8_tensor, input, 32u, &result) == LM_OK && result == 528.0f);
    assert(lm_model_tensor_binding_view(&q4_binding, q4_bytes, sizeof(q4_bytes) - 1u, &q4_tensor) == LM_ERR_CAPACITY);
    lm_model_close(model);
    std::remove(path);

    std::vector<unsigned char> malformed(24u, 0u);
    malformed[0] = 'G'; malformed[1] = 'G'; malformed[2] = 'U'; malformed[3] = 'F'; malformed[4] = 3u;
    malformed[8] = 1u;
    put_descriptor(&malformed, "short", 2u, 0u);
    const size_t malformed_header = (malformed.size() + 31u) & ~static_cast<size_t>(31u);
    malformed.resize(malformed_header + 17u, 0u);
    const char *bad_path = "test-short-native-binding.gguf";
    {
        std::ofstream file(bad_path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(malformed.data()), static_cast<std::streamsize>(malformed.size()));
    }
    lm_model_info info{};
    assert(lm_model_inspect(bad_path, &info, error, sizeof(error)) == LM_ERR_PARSE);
    std::remove(bad_path);
}

static void test_model_and_cpu_math() {
    const char *gguf = "test-fixture.gguf";
    {
        std::ofstream file(gguf, std::ios::binary);
        const unsigned char header[] = {
            'G','G','U','F', 3,0,0,0,
            1,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,
            1,0,0,0,0,0,0,0,
            'x',
            1,0,0,0,
            1,0,0,0,0,0,0,0,
            0,0,0,0,
            0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0
        };
        file.write(reinterpret_cast<const char *>(header), sizeof(header));
        const unsigned char data[4] = {0,0,0,0};
        file.write(reinterpret_cast<const char *>(data), sizeof(data));
    }
    lm_model_info info;
    char error[128] = {};
    assert(lm_model_inspect(gguf, &info, error, sizeof(error)) == LM_OK);
    assert(info.format == LM_MODEL_GGUF && info.version == 3u && info.tensor_count == 1u);
    lm_model_file *model = nullptr;
    assert(lm_model_open(gguf, &model, error, sizeof(error)) == LM_OK);
    lm_model_info opened_info{};
    assert(lm_model_get_info(model, &opened_info) == LM_OK && opened_info.header_bytes == info.header_bytes);
    lm_model_tensor_info descriptor{};
    assert(lm_model_tensor_info_at(model, 0u, &descriptor) == LM_OK);
    assert(std::strcmp(descriptor.name, "x") == 0 && descriptor.rank == 1u && descriptor.dims[0] == 1u && descriptor.type == 0u && descriptor.relative_offset == 0u);
    assert(lm_model_tensor_info_at(model, 1u, &descriptor) == LM_ERR_RANGE);
    lm_file_span tensor_span{};
    assert(lm_model_tensor_span(model, 0u, 4u, &tensor_span) == LM_OK);
    unsigned char tensor_bytes[4] = {1u, 1u, 1u, 1u};
    assert(lm_file_span_read(&tensor_span, 0u, tensor_bytes, sizeof(tensor_bytes)) == LM_OK);
    assert(tensor_bytes[0] == 0u && tensor_bytes[1] == 0u && tensor_bytes[2] == 0u && tensor_bytes[3] == 0u);
    assert(lm_model_tensor_span(model, info.file_bytes - info.header_bytes + 1u, 1u, &tensor_span) == LM_ERR_RANGE);
    lm_model_close(model);
    std::remove(gguf);

    const char *moe_path = "test-moe.gguf";
    std::vector<unsigned char> moe(24u, 0u);
    moe[0] = 'G'; moe[1] = 'G'; moe[2] = 'U'; moe[3] = 'F';
    moe[4] = 3u;
    moe[16] = 2u;
    auto put_u32 = [&moe](uint32_t value) {
        for (unsigned i = 0u; i < 4u; ++i) moe.push_back(static_cast<unsigned char>((value >> (8u * i)) & 0xffu));
    };
    auto put_u64 = [&moe](uint64_t value) {
        for (unsigned i = 0u; i < 8u; ++i) moe.push_back(static_cast<unsigned char>((value >> (8u * i)) & 0xffu));
    };
    auto put_key = [&moe, &put_u32, &put_u64](const char *key, uint32_t value) {
        const size_t length = std::strlen(key);
        put_u64(length);
        for (size_t i = 0u; i < length; ++i) moe.push_back(static_cast<unsigned char>(key[i]));
        put_u32(4u);
        put_u32(value);
    };
    put_key("llama.expert_count", 8u);
    put_key("llama.expert_used_count", 2u);
    moe.resize((moe.size() + 31u) & ~static_cast<size_t>(31u), 0u);
    { std::ofstream file(moe_path, std::ios::binary); file.write(reinterpret_cast<const char *>(moe.data()), static_cast<std::streamsize>(moe.size())); }
    assert(lm_model_inspect(moe_path, &info, error, sizeof(error)) == LM_OK);
    assert(info.expert_count == 8u && info.experts_per_token == 2u);
    std::remove(moe_path);

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

static void test_safetensors_parser() {
    const char *valid_path = "test-fixture.safetensors";
    const char *json = "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
    {
        std::ofstream file(valid_path, std::ios::binary);
        const uint64_t header_bytes = std::strlen(json);
        file.write(reinterpret_cast<const char *>(&header_bytes), sizeof(header_bytes));
        file.write(json, static_cast<std::streamsize>(header_bytes));
        const unsigned char data[8] = {};
        file.write(reinterpret_cast<const char *>(data), sizeof(data));
    }
    lm_model_info info{};
    char error[128] = {};
    assert(lm_model_inspect(valid_path, &info, error, sizeof(error)) == LM_OK);
    assert(info.format == LM_MODEL_SAFETENSORS && info.tensor_count == 1u);
    lm_model_file *model = nullptr;
    assert(lm_model_open(valid_path, &model, error, sizeof(error)) == LM_OK);
    lm_model_tensor_info descriptor{};
    assert(lm_model_tensor_info_at(model, 0u, &descriptor) == LM_OK);
    assert(std::strcmp(descriptor.name, "x") == 0 && descriptor.rank == 1u && descriptor.dims[0] == 2u && descriptor.type == LM_DTYPE_F32 && descriptor.relative_offset == 0u);
    assert(lm_model_tensor_info_at(model, 1u, &descriptor) == LM_ERR_RANGE);
    lm_model_tensor_binding binding{};
    assert(lm_model_tensor_bind_native(model, 0u, &binding) == LM_ERR_UNSUPPORTED);
    lm_model_close(model);
    std::remove(valid_path);

    const char *bad_path = "bad-fixture.safetensors";
    const char *bad_json = "{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[2,6]}}";
    {
        std::ofstream file(bad_path, std::ios::binary);
        const uint64_t header_bytes = std::strlen(bad_json);
        file.write(reinterpret_cast<const char *>(&header_bytes), sizeof(header_bytes));
        file.write(bad_json, static_cast<std::streamsize>(header_bytes));
        const unsigned char data[6] = {};
        file.write(reinterpret_cast<const char *>(data), sizeof(data));
    }
    assert(lm_model_inspect(bad_path, &info, error, sizeof(error)) == LM_ERR_PARSE);
    std::remove(bad_path);
}

static void test_cpu_decoder() {
    const float embedding[] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    const float gamma[] = {1.0f, 1.0f};
    const float identity[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float zeros[] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float output[] = {1.0f, 0.0f, 2.0f, 0.0f, 1.0f, 3.0f};
    lm_cpu_decoder_config config{};
    config.vocab_size = 3u;
    config.hidden_size = 2u;
    config.max_context = 2u;
    config.rms_epsilon = 1.0e-5f;
    config.rope_theta = 10000.0f;
    config.use_rope = 0u;
    config.embedding = embedding;
    config.rms_gamma_1 = gamma;
    config.wq = identity;
    config.wk = identity;
    config.wv = identity;
    config.wo = zeros;
    config.rms_gamma_2 = gamma;
    config.w1 = zeros;
    config.w2 = zeros;
    config.wout = output;
    lm_cpu_decoder *decoder = nullptr;
    assert(lm_cpu_decoder_create(&config, &decoder) == LM_OK);
    float logits[3] = {};
    assert(lm_cpu_decoder_step(decoder, 0u, logits, 3u) == LM_OK);
    assert(logits[0] == 1.0f && logits[1] == 0.0f && logits[2] == 2.0f);
    assert(lm_cpu_decoder_position(decoder) == 1u);
    assert(lm_cpu_decoder_step(decoder, 1u, logits, 3u) == LM_OK);
    assert(logits[0] == 0.0f && logits[1] == 1.0f && logits[2] == 3.0f);
    assert(lm_cpu_decoder_step(decoder, 0u, logits, 3u) == LM_ERR_CAPACITY);
    assert(lm_cpu_decoder_step(decoder, 3u, logits, 3u) == LM_ERR_RANGE);
    assert(lm_cpu_decoder_reset(decoder) == LM_OK && lm_cpu_decoder_position(decoder) == 0u);
    assert(lm_cpu_decoder_step(decoder, 0u, logits, 3u) == LM_OK);
    assert(logits[0] == 1.0f && logits[1] == 0.0f && logits[2] == 2.0f);
    lm_cpu_decoder_destroy(decoder);
}

static void test_moe_router() {
    const float logits[] = {1.0f, 3.0f, 3.0f, -2.0f};
    lm_moe_route route{};
    assert(lm_cpu_moe_route(logits, 4u, 2u, LM_MOE_SOFTMAX_ALL_THEN_TOPK, &route) == LM_OK);
    assert(route.selected[0] == 1u && route.selected[1] == 2u);
    assert(route.weights[0] == 0.5f && route.weights[1] == 0.5f);
    assert(std::fabs((route.weights[0] + route.weights[1]) - 1.0f) < 1.0e-6f);
    lm_moe_route selected_only{};
    assert(lm_cpu_moe_route(logits, 4u, 2u, LM_MOE_SOFTMAX_SELECTED_ONLY, &selected_only) == LM_OK);
    assert(selected_only.weights[0] == 0.5f && selected_only.weights[1] == 0.5f);
    const float selected_outputs[] = {2.0f, 4.0f, 6.0f, 8.0f};
    float combined[2] = {};
    assert(lm_cpu_moe_combine(&selected_only, selected_outputs, 2u, combined) == LM_OK);
    assert(combined[0] == 4.0f && combined[1] == 6.0f);
    const float bad_output[] = {2.0f, std::numeric_limits<float>::quiet_NaN(), 6.0f, 8.0f};
    assert(lm_cpu_moe_combine(&selected_only, bad_output, 2u, combined) == LM_ERR_RANGE);
    const float bad_logits[] = {0.0f, std::numeric_limits<float>::quiet_NaN()};
    assert(lm_cpu_moe_route(bad_logits, 2u, 1u, LM_MOE_SOFTMAX_ALL_THEN_TOPK, &route) == LM_ERR_RANGE);
    assert(lm_cpu_moe_route(logits, 4u, 3u, LM_MOE_SOFTMAX_ALL_THEN_TOPK, &route) == LM_OK);
    assert(lm_cpu_moe_route(logits, 4u, 0u, LM_MOE_SOFTMAX_ALL_THEN_TOPK, &route) == LM_ERR_ARGUMENT);
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

static void test_kernel_selection() {
    lm_kernel_caps cpu{0u, 0u, 0u};
    lm_kernel_choice choice{};
    assert(lm_kernel_select(LM_KERNEL_DOT_I8, LM_KERNEL_AUTO, &cpu, &choice) == LM_OK);
    assert(choice.path == LM_KERNEL_CPU_SCALAR);

    lm_kernel_caps dp4{1u, 1u, 1u};
    assert(lm_kernel_select(LM_KERNEL_DOT_I8, LM_KERNEL_AUTO, &dp4, &choice) == LM_OK);
    assert(choice.path == LM_KERNEL_VULKAN_DP4);
    assert(std::strcmp(choice.source_id, "vulkan/dp4") == 0);
    lm_kernel_contract contract{};
    assert(lm_kernel_contract_get(&choice, &contract) == LM_OK);
    assert(contract.input_dtype == LM_DTYPE_I8 && contract.output_dtype == LM_DTYPE_I32);
    assert(contract.minimum_alignment == 4u && contract.deterministic == 1u);
    assert(lm_kernel_select(LM_KERNEL_DOT_F32, LM_KERNEL_VULKAN_DP4, &dp4, &choice) == LM_ERR_UNSUPPORTED);
    assert(lm_kernel_select(LM_KERNEL_DOT_F32, LM_KERNEL_VULKAN_SCALAR, &dp4, &choice) == LM_OK);
    assert(choice.path == LM_KERNEL_VULKAN_SCALAR);
}

static void test_vulkan_dp4() {
    uint32_t device_count = 0u;
    const lm_status discovered = lm_vulkan_device_count(&device_count);
    if (discovered != LM_OK || device_count == 0u) return;
    const uint32_t a[] = {0x04030201u};
    const uint32_t b[] = {0x08070605u};
    int32_t gpu_result = 0;
    assert(lm_vulkan_dot_i8_dp4("dot_i8_dp4.comp.spv", 0u, a, b, 1u, &gpu_result) == LM_OK);
    assert(gpu_result == 70);
    const float scalar_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float scalar_b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float scalar_result = 0.0f;
    assert(lm_vulkan_dot_f32("dot_f32_scalar.comp.spv", 0u, scalar_a, scalar_b, 4u, &scalar_result) == LM_OK);
    assert(scalar_result == 70.0f);
    lm_kernel_caps caps{1u, 1u, 1u};
    lm_kernel_choice scalar_choice{};
    assert(lm_kernel_select(LM_KERNEL_DOT_F32, LM_KERNEL_VULKAN_SCALAR, &caps, &scalar_choice) == LM_OK);
    lm_kernel_io scalar_io{scalar_a, scalar_b, 4u, &scalar_result};
    scalar_result = 0.0f;
    assert(lm_vulkan_dispatch(&scalar_choice, "dot_f32_scalar.comp.spv", 0u, &scalar_io) == LM_OK);
    assert(scalar_result == 70.0f);
    lm_kernel_choice dp4_choice{};
    assert(lm_kernel_select(LM_KERNEL_DOT_I8, LM_KERNEL_VULKAN_DP4, &caps, &dp4_choice) == LM_OK);
    lm_kernel_io dp4_io{a, b, 1u, &gpu_result};
    gpu_result = 0;
    assert(lm_vulkan_dispatch(&dp4_choice, "dot_i8_dp4.comp.spv", 0u, &dp4_io) == LM_OK);
    assert(gpu_result == 70);
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
    test_quantize_policy_gate();
    test_vulkan_backend_resolution();
    test_tensor_and_buffer();
    test_file_access();
    test_native_model_tensor_binding();
    test_model_and_cpu_math();
    test_safetensors_parser();
    test_cpu_decoder();
    test_moe_router();
    test_kv_pages();
    test_kernel_selection();
    test_vulkan_dp4();
    test_probe_and_runtime();
    std::puts("core_tests=PASS");
    return 0;
}
