#include "lm/lm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>

static void set_error(char *dst, size_t cap, const char *text) {
    if (!dst || cap == 0u) return;
    std::strncpy(dst, text, cap - 1u);
    dst[cap - 1u] = '\0';
}

static uint64_t read_u64_le(const unsigned char *p) {
    uint64_t value = 0u;
    for (unsigned i = 0u; i < 8u; ++i) value |= static_cast<uint64_t>(p[i]) << (8u * i);
    return value;
}

static uint32_t read_u32_le(const unsigned char *p) {
    uint32_t value = 0u;
    for (unsigned i = 0u; i < 4u; ++i) value |= static_cast<uint32_t>(p[i]) << (8u * i);
    return value;
}

const char *lm_model_format_name(lm_model_format format) {
    switch (format) {
        case LM_MODEL_GGUF: return "gguf";
        case LM_MODEL_SAFETENSORS: return "safetensors";
        default: return "unknown";
    }
}

lm_status lm_model_inspect(const char *path, lm_model_info *out_info,
                           char *error_text, size_t error_capacity) {
    if (!path || !out_info) return LM_ERR_ARGUMENT;
    std::memset(out_info, 0, sizeof(*out_info));
    set_error(error_text, error_capacity, "");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { set_error(error_text, error_capacity, "cannot open model file"); return LM_ERR_IO; }
    const std::streamoff end = file.tellg();
    if (end < 0 || static_cast<uint64_t>(end) < 8u) {
        set_error(error_text, error_capacity, "model file is shorter than its minimum header");
        return LM_ERR_PARSE;
    }
    const uint64_t file_bytes = static_cast<uint64_t>(end);
    unsigned char header[16] = {};
    file.seekg(0);
    file.read(reinterpret_cast<char *>(header), sizeof(header));
    if (!file || file.gcount() < 8) {
        set_error(error_text, error_capacity, "cannot read model header");
        return LM_ERR_IO;
    }

    if (header[0] == 'G' && header[1] == 'G' && header[2] == 'U' && header[3] == 'F') {
        const uint32_t version = read_u32_le(header + 4u);
        if (version < 1u || version > 3u) {
            set_error(error_text, error_capacity, "unsupported GGUF version");
            return LM_ERR_UNSUPPORTED;
        }
        if (file_bytes < 24u) {
            set_error(error_text, error_capacity, "GGUF header is truncated");
            return LM_ERR_PARSE;
        }
        file.seekg(8);
        unsigned char counts[16] = {};
        file.read(reinterpret_cast<char *>(counts), sizeof(counts));
        if (!file) { set_error(error_text, error_capacity, "cannot read GGUF counts"); return LM_ERR_IO; }
        out_info->format = LM_MODEL_GGUF;
        out_info->version = version;
        out_info->file_bytes = file_bytes;
        out_info->header_bytes = 24u;
        out_info->tensor_count = read_u64_le(counts + 0u);
        const uint64_t metadata_count = read_u64_le(counts + 8u);
        if (metadata_count > file_bytes / 8u || out_info->tensor_count > file_bytes / 8u) {
            set_error(error_text, error_capacity, "GGUF count exceeds file bounds");
            return LM_ERR_PARSE;
        }
        return LM_OK;
    }

    const uint64_t header_bytes = read_u64_le(header);
    if (header_bytes == 0u || header_bytes > file_bytes - 8u || header_bytes > (1ull << 30u)) {
        set_error(error_text, error_capacity, "invalid SafeTensors header length");
        return LM_ERR_PARSE;
    }
    file.seekg(8);
    char first = 0;
    file.read(&first, 1);
    if (!file || first != '{') {
        set_error(error_text, error_capacity, "unknown model format or invalid SafeTensors JSON header");
        return LM_ERR_UNSUPPORTED;
    }
    file.seekg(static_cast<std::streamoff>(8u + header_bytes - 1u));
    char last = 0;
    file.read(&last, 1);
    if (!file || last != '}') {
        set_error(error_text, error_capacity, "SafeTensors header does not end as a JSON object");
        return LM_ERR_PARSE;
    }
    out_info->format = LM_MODEL_SAFETENSORS;
    out_info->version = 1u;
    out_info->file_bytes = file_bytes;
    out_info->header_bytes = 8u + header_bytes;
    out_info->tensor_count = 0u;
    set_error(error_text, error_capacity, "SafeTensors structural check passed; tensor JSON parsing is next slice");
    return LM_OK;
}

lm_status lm_cpu_dot_f32(const float *a, const float *b, size_t count, float *out) {
    if (!a || !b || !out || count == 0u) return LM_ERR_ARGUMENT;
    float sum = 0.0f;
    for (size_t i = 0u; i < count; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return LM_ERR_RANGE;
        sum += a[i] * b[i];
    }
    if (!std::isfinite(sum)) return LM_ERR_RANGE;
    *out = sum;
    return LM_OK;
}

lm_status lm_cpu_softmax_f32(const float *input, float *output, size_t count) {
    if (!input || !output || count == 0u) return LM_ERR_ARGUMENT;
    float max_value = -std::numeric_limits<float>::infinity();
    for (size_t i = 0u; i < count; ++i) {
        if (!std::isfinite(input[i])) return LM_ERR_RANGE;
        max_value = std::max(max_value, input[i]);
    }
    float sum = 0.0f;
    for (size_t i = 0u; i < count; ++i) {
        output[i] = std::exp(input[i] - max_value);
        sum += output[i];
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) return LM_ERR_RANGE;
    for (size_t i = 0u; i < count; ++i) output[i] /= sum;
    return LM_OK;
}
