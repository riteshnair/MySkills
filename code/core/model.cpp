#include "lm/lm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <vector>

struct lm_model_file {
    lm_file *file;
    lm_model_info info;
    std::vector<lm_model_tensor_info> tensors;
};

namespace {

constexpr uint64_t kMaxSafeTensorsHeader = 32ull << 20u;
constexpr uint64_t kMaxContainerItems = 1ull << 20u;

void set_error(char *dst, size_t cap, const char *text) {
    if (!dst || cap == 0u) return;
    std::strncpy(dst, text, cap - 1u);
    dst[cap - 1u] = '\0';
}

uint64_t read_u64_le(const unsigned char *p) {
    uint64_t value = 0u;
    for (unsigned i = 0u; i < 8u; ++i) value |= static_cast<uint64_t>(p[i]) << (8u * i);
    return value;
}

uint32_t read_u32_le(const unsigned char *p) {
    uint32_t value = 0u;
    for (unsigned i = 0u; i < 4u; ++i) value |= static_cast<uint32_t>(p[i]) << (8u * i);
    return value;
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
    const uint64_t remainder = value % alignment;
    return remainder == 0u ? value : value + (alignment - remainder);
}

lm_status native_contract(const lm_model_tensor_info &descriptor, lm_quant_format *format,
                         uint32_t *elements_per_block, uint32_t *bytes_per_block,
                         uint64_t *elements, uint64_t *bytes) {
    if (!format || !elements_per_block || !bytes_per_block || !elements || !bytes) return LM_ERR_ARGUMENT;
    if (descriptor.type == 2u) {
        *format = LM_QUANT_GGML_Q4_0;
        *elements_per_block = 32u;
        *bytes_per_block = 18u;
    } else if (descriptor.type == 8u) {
        *format = LM_QUANT_GGML_Q8_0;
        *elements_per_block = 32u;
        *bytes_per_block = 34u;
    } else {
        return LM_ERR_UNSUPPORTED;
    }
    if (descriptor.rank == 0u || descriptor.rank > 8u) return LM_ERR_PARSE;
    uint64_t count = 1u;
    for (uint32_t i = 0u; i < descriptor.rank; ++i) {
        if (descriptor.dims[i] == 0u || count > std::numeric_limits<uint64_t>::max() / descriptor.dims[i]) return LM_ERR_RANGE;
        count *= descriptor.dims[i];
    }
    if (count % *elements_per_block != 0u) return LM_ERR_PARSE;
    const uint64_t blocks = count / *elements_per_block;
    if (blocks > std::numeric_limits<uint64_t>::max() / *bytes_per_block) return LM_ERR_RANGE;
    *elements = count;
    *bytes = blocks * *bytes_per_block;
    return LM_OK;
}

class BinaryReader {
public:
    explicit BinaryReader(const char *path) : stream_(path, std::ios::binary), position_(0u), size_(0u) {
        if (!stream_) return;
        stream_.seekg(0, std::ios::end);
        const std::streamoff end = stream_.tellg();
        if (end < 0) { stream_.setstate(std::ios::failbit); return; }
        size_ = static_cast<uint64_t>(end);
        stream_.seekg(0, std::ios::beg);
    }

    bool good() const { return stream_.good(); }
    uint64_t position() const { return position_; }
    uint64_t size() const { return size_; }

    bool read(void *dst, uint64_t bytes) {
        if (bytes > size_ - (position_ <= size_ ? position_ : size_)) return false;
        if (bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) return false;
        stream_.read(static_cast<char *>(dst), static_cast<std::streamsize>(bytes));
        if (stream_.gcount() != static_cast<std::streamsize>(bytes)) return false;
        position_ += bytes;
        return true;
    }

    bool skip(uint64_t bytes) {
        if (bytes > size_ - (position_ <= size_ ? position_ : size_)) return false;
        if (bytes > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) return false;
        stream_.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
        if (!stream_) return false;
        position_ += bytes;
        return true;
    }

    bool u8(uint8_t *value) { return read(value, 1u); }
    bool u32(uint32_t *value) { unsigned char bytes[4] = {}; if (!read(bytes, 4u)) return false; *value = read_u32_le(bytes); return true; }
    bool u64(uint64_t *value) { unsigned char bytes[8] = {}; if (!read(bytes, 8u)) return false; *value = read_u64_le(bytes); return true; }

    bool string(std::string *value, uint64_t max_bytes) {
        uint64_t length = 0u;
        if (!u64(&length) || length > max_bytes || length > size_ - position_) return false;
        value->assign(static_cast<size_t>(length), '\0');
        return length == 0u || read(value->data(), length);
    }

private:
    std::ifstream stream_;
    uint64_t position_;
    uint64_t size_;
};

bool valid_gguf_key(const std::string &key) {
    if (key.empty() || key.size() > 65535u) return false;
    bool segment_has_character = false;
    for (size_t i = 0u; i < key.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(key[i]);
        if (c == '.') {
            if (!segment_has_character) return false;
            segment_has_character = false;
        } else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') {
            segment_has_character = true;
        } else {
            return false;
        }
    }
    return segment_has_character;
}

bool skip_gguf_value(BinaryReader &reader, uint32_t type, uint32_t depth) {
    if (depth > 64u) return false;
    switch (type) {
        case 0u: case 1u: case 7u: return reader.skip(1u);
        case 2u: case 3u: return reader.skip(2u);
        case 4u: case 5u: case 6u: return reader.skip(4u);
        case 10u: case 11u: case 12u: return reader.skip(8u);
        case 8u: {
            uint64_t length = 0u;
            return reader.u64(&length) && length <= (16ull << 20u) && reader.skip(length);
        }
        case 9u: {
            uint32_t element_type = 0u;
            uint64_t length = 0u;
            if (!reader.u32(&element_type) || !reader.u64(&length) || length > kMaxContainerItems) return false;
            for (uint64_t i = 0u; i < length; ++i)
                if (!skip_gguf_value(reader, element_type, depth + 1u)) return false;
            return true;
        }
        default: return false;
    }
}

bool read_gguf_alignment(BinaryReader &reader, uint32_t type, uint64_t *alignment) {
    if (type == 4u) { uint32_t value = 0u; if (!reader.u32(&value)) return false; *alignment = value; return true; }
    if (type == 10u) return reader.u64(alignment);
    return false;
}

bool has_suffix(const std::string &value, const char *suffix) {
    const size_t suffix_size = std::strlen(suffix);
    return value.size() >= suffix_size && value.compare(value.size() - suffix_size, suffix_size, suffix) == 0;
}

bool read_moe_count(BinaryReader &reader, uint32_t type, uint64_t *count) {
    if (type == 4u) { uint32_t value = 0u; if (!reader.u32(&value)) return false; *count = value; return true; }
    if (type == 10u) return reader.u64(count);
    return false;
}

lm_status inspect_gguf(const char *path, lm_model_info *out_info, char *error_text, size_t error_capacity,
                       std::vector<lm_model_tensor_info> *out_tensors = nullptr) {
    BinaryReader reader(path);
    if (!reader.good()) { set_error(error_text, error_capacity, "cannot open model file"); return LM_ERR_IO; }
    if (reader.size() < 24u) { set_error(error_text, error_capacity, "GGUF header is truncated"); return LM_ERR_PARSE; }
    unsigned char magic[4] = {};
    uint32_t version = 0u;
    uint64_t tensor_count = 0u;
    uint64_t metadata_count = 0u;
    if (!reader.read(magic, 4u) || !reader.u32(&version) || !reader.u64(&tensor_count) || !reader.u64(&metadata_count)) {
        set_error(error_text, error_capacity, "cannot read GGUF header"); return LM_ERR_IO;
    }
    if (std::memcmp(magic, "GGUF", 4u) != 0) { set_error(error_text, error_capacity, "invalid GGUF magic"); return LM_ERR_PARSE; }
    if (version != 3u) { set_error(error_text, error_capacity, "only GGUF version 3 is supported"); return LM_ERR_UNSUPPORTED; }
    if (tensor_count > kMaxContainerItems || metadata_count > kMaxContainerItems) {
        set_error(error_text, error_capacity, "GGUF item count exceeds safety limit"); return LM_ERR_CAPACITY;
    }
    uint64_t alignment = 32u;
    uint64_t expert_count = 0u;
    uint64_t experts_per_token = 0u;
    bool has_expert_count = false;
    bool has_experts_per_token = false;
    for (uint64_t i = 0u; i < metadata_count; ++i) {
        std::string key;
        uint32_t type = 0u;
        if (!reader.string(&key, 65535u) || !valid_gguf_key(key) || !reader.u32(&type) || type > 12u) {
            set_error(error_text, error_capacity, "invalid GGUF metadata entry"); return LM_ERR_PARSE;
        }
        const uint64_t old_alignment = alignment;
        if (key == "general.alignment") {
            if (!read_gguf_alignment(reader, type, &alignment)) {
                set_error(error_text, error_capacity, "invalid GGUF alignment metadata"); return LM_ERR_PARSE;
            }
            if (alignment == 0u || alignment > (1ull << 20u) || (alignment % 8u) != 0u) {
                set_error(error_text, error_capacity, "GGUF alignment must be a bounded multiple of 8"); return LM_ERR_PARSE;
            }
        } else if (has_suffix(key, ".expert_count")) {
            if (!read_moe_count(reader, type, &expert_count)) {
                set_error(error_text, error_capacity, "invalid GGUF expert count metadata"); return LM_ERR_PARSE;
            }
            has_expert_count = true;
        } else if (has_suffix(key, ".expert_used_count")) {
            if (!read_moe_count(reader, type, &experts_per_token)) {
                set_error(error_text, error_capacity, "invalid GGUF experts-per-token metadata"); return LM_ERR_PARSE;
            }
            has_experts_per_token = true;
        } else if (!skip_gguf_value(reader, type, 0u)) {
            set_error(error_text, error_capacity, "invalid GGUF metadata value"); return LM_ERR_PARSE;
        }
        if (key != "general.alignment") alignment = old_alignment;
    }
    if ((has_expert_count != has_experts_per_token) || expert_count == 0u || experts_per_token == 0u ||
        expert_count > UINT32_MAX || experts_per_token > UINT32_MAX || experts_per_token > expert_count) {
        if (has_expert_count || has_experts_per_token) {
            set_error(error_text, error_capacity, "invalid GGUF MoE expert metadata"); return LM_ERR_PARSE;
        }
    }
    std::vector<uint64_t> relative_offsets;
    relative_offsets.reserve(static_cast<size_t>(tensor_count));
    std::vector<lm_model_tensor_info> parsed_tensors;
    parsed_tensors.reserve(static_cast<size_t>(tensor_count));
    for (uint64_t i = 0u; i < tensor_count; ++i) {
        std::string name;
        uint32_t dimensions = 0u;
        if (!reader.string(&name, 64u) || name.empty() || !reader.u32(&dimensions) || dimensions > 8u) {
            set_error(error_text, error_capacity, "invalid GGUF tensor descriptor"); return LM_ERR_PARSE;
        }
        lm_model_tensor_info tensor_info{};
        std::strncpy(tensor_info.name, name.c_str(), sizeof(tensor_info.name) - 1u);
        tensor_info.rank = dimensions;
        for (uint32_t d = 0u; d < dimensions; ++d) {
            uint64_t dimension = 0u;
            if (!reader.u64(&dimension) || dimension == 0u) {
                set_error(error_text, error_capacity, "invalid GGUF tensor dimension"); return LM_ERR_PARSE;
            }
            tensor_info.dims[d] = dimension;
        }
        uint32_t tensor_type = 0u;
        uint64_t offset = 0u;
        if (!reader.u32(&tensor_type) || tensor_type > 39u || !reader.u64(&offset)) {
            set_error(error_text, error_capacity, "invalid GGUF tensor type or offset"); return LM_ERR_PARSE;
        }
        if ((offset % alignment) != 0u) {
            set_error(error_text, error_capacity, "GGUF tensor offset is not aligned"); return LM_ERR_PARSE;
        }
        tensor_info.type = tensor_type;
        tensor_info.relative_offset = offset;
        parsed_tensors.push_back(tensor_info);
        relative_offsets.push_back(offset);
    }
    const uint64_t tensor_data_start = align_up(reader.position(), alignment);
    if (tensor_data_start > reader.size()) {
        set_error(error_text, error_capacity, "GGUF tensor-data alignment exceeds file"); return LM_ERR_PARSE;
    }
    std::sort(relative_offsets.begin(), relative_offsets.end());
    const uint64_t tensor_data_bytes = reader.size() - tensor_data_start;
    for (size_t i = 0u; i < relative_offsets.size(); ++i) {
        if (i > 0u && relative_offsets[i] == relative_offsets[i - 1u]) {
            set_error(error_text, error_capacity, "duplicate GGUF tensor offsets"); return LM_ERR_PARSE;
        }
        if (relative_offsets[i] > tensor_data_bytes) {
            set_error(error_text, error_capacity, "GGUF tensor offset exceeds file"); return LM_ERR_PARSE;
        }
    }
    for (const lm_model_tensor_info &tensor : parsed_tensors) {
        lm_quant_format format = LM_QUANT_NONE;
        uint32_t elements_per_block = 0u;
        uint32_t bytes_per_block = 0u;
        uint64_t elements = 0u;
        uint64_t expected_bytes = 0u;
        const lm_status contract = native_contract(tensor, &format, &elements_per_block,
                                                    &bytes_per_block, &elements, &expected_bytes);
        if (contract == LM_ERR_UNSUPPORTED) continue;
        if (contract != LM_OK) {
            set_error(error_text, error_capacity, "invalid native GGUF tensor shape"); return LM_ERR_PARSE;
        }
        uint64_t limit = tensor_data_bytes;
        for (const uint64_t offset : relative_offsets) {
            if (offset > tensor.relative_offset) { limit = offset; break; }
        }
        if (tensor.relative_offset > limit || expected_bytes > limit - tensor.relative_offset) {
            set_error(error_text, error_capacity, "native GGUF tensor payload exceeds its bounded range"); return LM_ERR_PARSE;
        }
    }
    if (out_tensors) *out_tensors = std::move(parsed_tensors);
    out_info->format = LM_MODEL_GGUF;
    out_info->version = version;
    out_info->file_bytes = reader.size();
    out_info->header_bytes = tensor_data_start;
    out_info->tensor_count = tensor_count;
    out_info->expert_count = has_expert_count ? static_cast<uint32_t>(expert_count) : 0u;
    out_info->experts_per_token = has_experts_per_token ? static_cast<uint32_t>(experts_per_token) : 0u;
    return LM_OK;
}

class JsonCursor {
public:
    JsonCursor(const char *begin, const char *end) : p_(begin), end_(end) {}
    const char *position() const { return p_; }

    void whitespace() { while (p_ < end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_; }
    bool character(char expected) { whitespace(); if (p_ >= end_ || *p_ != expected) return false; ++p_; return true; }
    bool consume(char expected) { if (p_ >= end_ || *p_ != expected) return false; ++p_; return true; }

    bool string(std::string *out, size_t max_length) {
        whitespace();
        if (p_ >= end_ || *p_ != '"') return false;
        ++p_;
        out->clear();
        while (p_ < end_) {
            const unsigned char c = static_cast<unsigned char>(*p_++);
            if (c == '"') return true;
            if (c < 0x20u) return false;
            if (c == '\\') {
                if (p_ >= end_) return false;
                const char escaped = *p_++;
                if (escaped == 'u') { if (end_ - p_ < 4) return false; p_ += 4; }
                else if (escaped != '"' && escaped != '\\' && escaped != '/' && escaped != 'b' && escaped != 'f' && escaped != 'n' && escaped != 'r' && escaped != 't') return false;
            }
            if (out->size() < max_length) out->push_back(static_cast<char>(c));
        }
        return false;
    }

    bool unsigned_number(uint64_t *out) {
        whitespace();
        if (p_ >= end_ || *p_ < '0' || *p_ > '9') return false;
        uint64_t value = 0u;
        while (p_ < end_ && *p_ >= '0' && *p_ <= '9') {
            const uint64_t digit = static_cast<uint64_t>(*p_++ - '0');
            if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10u) return false;
            value = value * 10u + digit;
        }
        *out = value;
        return true;
    }

    bool skip_value(uint32_t depth) {
        if (depth > 64u) return false;
        whitespace();
        if (p_ >= end_) return false;
        if (*p_ == '"') { std::string value; return string(&value, 1u << 20u); }
        if (*p_ == '{') {
            ++p_; whitespace(); if (p_ < end_ && *p_ == '}') { ++p_; return true; }
            for (;;) {
                std::string key; if (!string(&key, 1u << 20u) || !character(':') || !skip_value(depth + 1u)) return false;
                whitespace(); if (p_ < end_ && *p_ == '}') { ++p_; return true; }
                if (!character(',')) return false;
            }
        }
        if (*p_ == '[') {
            ++p_; whitespace(); if (p_ < end_ && *p_ == ']') { ++p_; return true; }
            for (;;) {
                if (!skip_value(depth + 1u)) return false;
                whitespace(); if (p_ < end_ && *p_ == ']') { ++p_; return true; }
                if (!character(',')) return false;
            }
        }
        const char *start = p_;
        while (p_ < end_ && *p_ != ',' && *p_ != ']' && *p_ != '}' && *p_ != ' ' && *p_ != '\n' && *p_ != '\r' && *p_ != '\t') ++p_;
        return p_ > start;
    }

private:
    const char *p_;
    const char *end_;
};

uint32_t safe_dtype_code(const std::string &dtype) {
    if (dtype == "F32") return LM_DTYPE_F32;
    if (dtype == "F16") return LM_DTYPE_F16;
    if (dtype == "BF16") return LM_DTYPE_BF16;
    if (dtype == "I8") return LM_DTYPE_I8;
    if (dtype == "I32") return LM_DTYPE_I32;
    if (dtype == "U8") return LM_DTYPE_U8;
    if (dtype == "U4") return 0x100u;
    if (dtype == "I4") return 0x101u;
    return 0xffffffffu;
}

uint64_t safe_dtype_bytes(const std::string &dtype, uint64_t elements, bool *known) {
    uint64_t element_bytes = 0u;
    if (dtype == "BOOL" || dtype == "U8" || dtype == "I8" || dtype == "F8_E4M3" || dtype == "F8_E5M2") element_bytes = 1u;
    else if (dtype == "U16" || dtype == "I16" || dtype == "F16" || dtype == "BF16") element_bytes = 2u;
    else if (dtype == "U32" || dtype == "I32" || dtype == "F32") element_bytes = 4u;
    else if (dtype == "U64" || dtype == "I64" || dtype == "F64") element_bytes = 8u;
    else if (dtype == "U4" || dtype == "I4") {
        *known = true;
        return elements / 2u + (elements % 2u);
    } else { *known = false; return 0u; }
    *known = true;
    if (elements > std::numeric_limits<uint64_t>::max() / element_bytes) return 0u;
    return elements * element_bytes;
}

lm_status inspect_safetensors(const char *path, lm_model_info *out_info, char *error_text, size_t error_capacity,
                              std::vector<lm_model_tensor_info> *out_tensors = nullptr) {
    BinaryReader reader(path);
    if (!reader.good()) { set_error(error_text, error_capacity, "cannot open model file"); return LM_ERR_IO; }
    if (reader.size() < 8u) { set_error(error_text, error_capacity, "SafeTensors header length is truncated"); return LM_ERR_PARSE; }
    uint64_t header_bytes = 0u;
    if (!reader.u64(&header_bytes) || header_bytes == 0u || header_bytes > kMaxSafeTensorsHeader || header_bytes > reader.size() - 8u) {
        set_error(error_text, error_capacity, "invalid SafeTensors header length"); return LM_ERR_PARSE;
    }
    std::vector<char> header(static_cast<size_t>(header_bytes));
    if (!reader.read(header.data(), header_bytes)) { set_error(error_text, error_capacity, "cannot read SafeTensors header"); return LM_ERR_IO; }
    JsonCursor json(header.data(), header.data() + header.size());
    if (!json.character('{')) { set_error(error_text, error_capacity, "SafeTensors header is not a JSON object"); return LM_ERR_PARSE; }
    struct Range { uint64_t begin; uint64_t end; };
    std::vector<Range> ranges;
    json.whitespace();
    if (json.position() < header.data() + header.size() && *json.position() != '}') {
        for (;;) {
            std::string name;
            if (!json.string(&name, 1u << 20u) || !json.character(':')) { set_error(error_text, error_capacity, "invalid SafeTensors key"); return LM_ERR_PARSE; }
            if (name == "__metadata__") {
                if (!json.skip_value(0u)) { set_error(error_text, error_capacity, "invalid SafeTensors metadata"); return LM_ERR_PARSE; }
            } else {
                if (!json.character('{')) { set_error(error_text, error_capacity, "SafeTensors tensor descriptor is not an object"); return LM_ERR_PARSE; }
                bool have_dtype = false, have_shape = false, have_offsets = false;
                std::string dtype;
                std::vector<uint64_t> shape;
                uint64_t elements = 1u, begin = 0u, end = 0u;
                json.whitespace();
                if (json.position() < header.data() + header.size() && *json.position() != '}') {
                    for (;;) {
                        std::string field;
                        if (!json.string(&field, 64u) || !json.character(':')) { set_error(error_text, error_capacity, "invalid SafeTensors tensor field"); return LM_ERR_PARSE; }
                        if (field == "dtype") {
                            if (!json.string(&dtype, 32u)) { set_error(error_text, error_capacity, "invalid SafeTensors dtype"); return LM_ERR_PARSE; }
                            have_dtype = true;
                        } else if (field == "shape") {
                            if (!json.character('[')) { set_error(error_text, error_capacity, "invalid SafeTensors shape"); return LM_ERR_PARSE; }
                            elements = 1u; json.whitespace();
                            if (json.position() < header.data() + header.size() && *json.position() != ']') {
                                for (;;) {
                                    uint64_t dimension = 0u;
                                    if (!json.unsigned_number(&dimension) || (dimension != 0u && elements > std::numeric_limits<uint64_t>::max() / dimension)) { set_error(error_text, error_capacity, "SafeTensors shape overflows"); return LM_ERR_RANGE; }
                                    elements *= dimension;
                                    shape.push_back(dimension);
                                    json.whitespace(); if (json.position() < header.data() + header.size() && *json.position() == ']') { if (!json.consume(']')) return LM_ERR_PARSE; break; }
                                    if (!json.character(',')) { set_error(error_text, error_capacity, "invalid SafeTensors shape array"); return LM_ERR_PARSE; }
                                }
                            } else return LM_ERR_PARSE;
                            have_shape = true;
                        } else if (field == "data_offsets") {
                            if (!json.character('[') || !json.unsigned_number(&begin) || !json.character(',') || !json.unsigned_number(&end) || !json.character(']') || begin > end) { set_error(error_text, error_capacity, "invalid SafeTensors data offsets"); return LM_ERR_PARSE; }
                            have_offsets = true;
                        } else if (!json.skip_value(0u)) { set_error(error_text, error_capacity, "invalid SafeTensors tensor field value"); return LM_ERR_PARSE; }
                        json.whitespace();
                        if (json.position() < header.data() + header.size() && *json.position() == '}') { if (!json.consume('}')) return LM_ERR_PARSE; break; }
                        if (!json.character(',')) { set_error(error_text, error_capacity, "invalid SafeTensors tensor object"); return LM_ERR_PARSE; }
                    }
                } else return LM_ERR_PARSE;
                bool dtype_known = false;
                const uint64_t expected_bytes = safe_dtype_bytes(dtype, elements, &dtype_known);
                if (!have_dtype || !have_shape || !have_offsets || !dtype_known) { set_error(error_text, error_capacity, "unsupported or incomplete SafeTensors descriptor"); return LM_ERR_UNSUPPORTED; }
                if (expected_bytes != end - begin) { set_error(error_text, error_capacity, "SafeTensors descriptor size does not match shape and dtype"); return LM_ERR_PARSE; }
                if (out_tensors) {
                    if (shape.size() > 8u || safe_dtype_code(dtype) == 0xffffffffu) {
                        set_error(error_text, error_capacity, "SafeTensors descriptor cannot be represented"); return LM_ERR_UNSUPPORTED;
                    }
                    lm_model_tensor_info tensor_info{};
                    std::strncpy(tensor_info.name, name.c_str(), sizeof(tensor_info.name) - 1u);
                    tensor_info.rank = static_cast<uint32_t>(shape.size());
                    for (size_t d = 0u; d < shape.size(); ++d) tensor_info.dims[d] = shape[d];
                    tensor_info.type = safe_dtype_code(dtype);
                    tensor_info.relative_offset = begin;
                    out_tensors->push_back(tensor_info);
                }
                ranges.push_back({begin, end});
                if (ranges.size() > kMaxContainerItems) { set_error(error_text, error_capacity, "SafeTensors tensor count exceeds safety limit"); return LM_ERR_CAPACITY; }
            }
            json.whitespace();
            if (json.position() < header.data() + header.size() && *json.position() == '}') { if (!json.consume('}')) return LM_ERR_PARSE; break; }
            if (!json.character(',')) { set_error(error_text, error_capacity, "invalid SafeTensors top-level object"); return LM_ERR_PARSE; }
        }
    } else return LM_ERR_PARSE;
    json.whitespace();
    if (json.position() != header.data() + header.size() || reader.size() - (8u + header_bytes) > std::numeric_limits<uint64_t>::max()) {
        set_error(error_text, error_capacity, "trailing bytes in SafeTensors JSON header"); return LM_ERR_PARSE;
    }
    const uint64_t data_bytes = reader.size() - (8u + header_bytes);
    for (const Range &range : ranges) {
        if (range.end > data_bytes) { set_error(error_text, error_capacity, "SafeTensors tensor range exceeds file"); return LM_ERR_PARSE; }
    }
    std::sort(ranges.begin(), ranges.end(), [](const Range &a, const Range &b) { return a.begin < b.begin; });
    for (size_t i = 1u; i < ranges.size(); ++i) {
        if (ranges[i].begin < ranges[i - 1u].end) { set_error(error_text, error_capacity, "overlapping SafeTensors tensor ranges"); return LM_ERR_PARSE; }
    }
    out_info->format = LM_MODEL_SAFETENSORS;
    out_info->version = 1u;
    out_info->file_bytes = reader.size();
    out_info->header_bytes = 8u + header_bytes;
    out_info->tensor_count = ranges.size();
    return LM_OK;
}

} // namespace

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
    BinaryReader probe(path);
    if (!probe.good()) { set_error(error_text, error_capacity, "cannot open model file"); return LM_ERR_IO; }
    if (probe.size() >= 4u) {
        unsigned char magic[4] = {};
        if (!probe.read(magic, 4u)) return LM_ERR_IO;
        if (std::memcmp(magic, "GGUF", 4u) == 0) return inspect_gguf(path, out_info, error_text, error_capacity);
    }
    return inspect_safetensors(path, out_info, error_text, error_capacity);
}

lm_status lm_model_open(const char *path, lm_model_file **out_model, char *error_text, size_t error_capacity) {
    if (!path || !out_model) return LM_ERR_ARGUMENT;
    *out_model = nullptr;
    lm_model_info info{};
    const lm_status inspected = lm_model_inspect(path, &info, error_text, error_capacity);
    if (inspected != LM_OK) return inspected;
    lm_file *file = nullptr;
    std::vector<lm_model_tensor_info> tensors;
    if (info.format == LM_MODEL_GGUF) {
        const lm_status descriptors = inspect_gguf(path, &info, error_text, error_capacity, &tensors);
        if (descriptors != LM_OK) return descriptors;
    } else if (info.format == LM_MODEL_SAFETENSORS) {
        const lm_status descriptors = inspect_safetensors(path, &info, error_text, error_capacity, &tensors);
        if (descriptors != LM_OK) return descriptors;
    }
    const lm_status opened = lm_file_open(path, &file);
    if (opened != LM_OK) return opened;
    try {
        lm_model_file *model = new lm_model_file{file, info, std::move(tensors)};
        *out_model = model;
        return LM_OK;
    } catch (const std::bad_alloc &) {
        lm_file_close(file);
        return LM_ERR_CAPACITY;
    }
}

void lm_model_close(lm_model_file *model) {
    if (!model) return;
    lm_file_close(model->file);
    delete model;
}

lm_status lm_model_get_info(const lm_model_file *model, lm_model_info *out_info) {
    if (!model || !out_info) return LM_ERR_ARGUMENT;
    *out_info = model->info;
    return LM_OK;
}

lm_status lm_model_tensor_info_at(const lm_model_file *model, uint64_t index, lm_model_tensor_info *out_info) {
    if (!model || !out_info) return LM_ERR_ARGUMENT;
    if (index >= model->tensors.size()) return LM_ERR_RANGE;
    *out_info = model->tensors[static_cast<size_t>(index)];
    return LM_OK;
}

lm_status lm_model_tensor_span(const lm_model_file *model, uint64_t relative_offset, uint64_t bytes, lm_file_span *out_span) {
    if (!model || !out_span) return LM_ERR_ARGUMENT;
    if (relative_offset > model->info.file_bytes - model->info.header_bytes ||
        bytes > model->info.file_bytes - model->info.header_bytes - relative_offset)
        return LM_ERR_RANGE;
    return lm_file_span_make(model->file, model->info.header_bytes + relative_offset, bytes, out_span);
}

lm_status lm_model_tensor_bind_native(const lm_model_file *model, uint64_t index, lm_model_tensor_binding *out_binding) {
    if (!model || !out_binding) return LM_ERR_ARGUMENT;
    if (index >= model->tensors.size()) return LM_ERR_RANGE;
    const lm_model_tensor_info &descriptor = model->tensors[static_cast<size_t>(index)];
    lm_quant_format format = LM_QUANT_NONE;
    uint32_t elements_per_block = 0u;
    uint32_t bytes_per_block = 0u;
    uint64_t elements = 0u;
    uint64_t expected_bytes = 0u;
    const lm_status contract = native_contract(descriptor, &format, &elements_per_block,
                                                &bytes_per_block, &elements, &expected_bytes);
    if (contract != LM_OK) return contract;
    const uint64_t data_bytes = model->info.file_bytes - model->info.header_bytes;
    uint64_t limit = data_bytes;
    for (const lm_model_tensor_info &other : model->tensors) {
        if (other.relative_offset > descriptor.relative_offset && other.relative_offset < limit)
            limit = other.relative_offset;
    }
    if (descriptor.relative_offset > limit || expected_bytes > limit - descriptor.relative_offset) return LM_ERR_PARSE;
    lm_file_span span{};
    const lm_status spanned = lm_model_tensor_span(model, descriptor.relative_offset, expected_bytes, &span);
    if (spanned != LM_OK) return spanned;
    std::memset(out_binding, 0, sizeof(*out_binding));
    out_binding->descriptor = descriptor;
    out_binding->span = span;
    out_binding->elements = elements;
    out_binding->quant_format = format;
    out_binding->quant_elements_per_block = elements_per_block;
    out_binding->quant_bytes_per_block = bytes_per_block;
    return LM_OK;
}

lm_status lm_model_tensor_binding_view(const lm_model_tensor_binding *binding, void *data, uint64_t bytes, lm_tensor *out_tensor) {
    if (!binding || !data || !out_tensor) return LM_ERR_ARGUMENT;
    if (binding->span.bytes != bytes) return LM_ERR_CAPACITY;
    uint32_t dims[8] = {};
    if (binding->descriptor.rank == 0u || binding->descriptor.rank > 8u) return LM_ERR_RANGE;
    for (uint32_t i = 0u; i < binding->descriptor.rank; ++i) {
        if (binding->descriptor.dims[i] == 0u || binding->descriptor.dims[i] > UINT32_MAX) return LM_ERR_RANGE;
        dims[i] = static_cast<uint32_t>(binding->descriptor.dims[i]);
    }
    if (binding->quant_format == LM_QUANT_GGML_Q4_0)
        return lm_tensor_make_q4_0_view(data, bytes, binding->descriptor.rank, dims, out_tensor);
    if (binding->quant_format == LM_QUANT_GGML_Q8_0)
        return lm_tensor_make_q8_0_view(data, bytes, binding->descriptor.rank, dims, out_tensor);
    return LM_ERR_UNSUPPORTED;
}

lm_status lm_model_tensor_binding_read(const lm_model_tensor_binding *binding, void *data, uint64_t bytes, lm_tensor *out_tensor) {
    if (!binding || !data || !out_tensor) return LM_ERR_ARGUMENT;
    if (bytes != binding->span.bytes) return LM_ERR_CAPACITY;
    if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) return LM_ERR_RANGE;
    const lm_status read = lm_file_span_read(&binding->span, 0u, data, static_cast<size_t>(bytes));
    if (read != LM_OK) return read;
    return lm_model_tensor_binding_view(binding, data, bytes, out_tensor);
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

float half_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16u;
    const uint32_t exponent = (bits >> 10u) & 0x1fu;
    const uint32_t fraction = bits & 0x03ffu;
    uint32_t value = 0u;
    if (exponent == 0u) {
        if (fraction == 0u) value = sign;
        else {
            uint32_t normalized = fraction;
            uint32_t shift = 0u;
            while ((normalized & 0x0400u) == 0u) { normalized <<= 1u; ++shift; }
            value = sign | ((127u - 15u - shift) << 23u) | ((normalized & 0x03ffu) << 13u);
        }
    } else if (exponent == 0x1fu) value = sign | 0x7f800000u | (fraction << 13u);
    else value = sign | ((exponent + 112u) << 23u) | (fraction << 13u);
    float result = 0.0f;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

lm_status lm_cpu_dot_q4_0(const lm_tensor *weights, const float *input,
                          uint64_t elements, float *out) {
    if (!weights || !input || !out || elements == 0u || elements % 32u != 0u ||
        weights->quant_format != LM_QUANT_GGML_Q4_0 || weights->dtype != LM_DTYPE_U8)
        return LM_ERR_ARGUMENT;
    if (lm_tensor_validate(weights) != LM_OK) return LM_ERR_RANGE;
    const uint64_t required_bytes = (elements / 32u) * 18u;
    if (required_bytes != weights->bytes) return LM_ERR_CAPACITY;
    const unsigned char *packed = static_cast<const unsigned char *>(weights->data);
    float sum = 0.0f;
    for (uint64_t block = 0u; block < elements / 32u; ++block) {
        const size_t base = static_cast<size_t>(block * 18u);
        const uint16_t scale_bits = static_cast<uint16_t>(packed[base]) | static_cast<uint16_t>(packed[base + 1u] << 8u);
        const float scale = half_to_float(scale_bits);
        if (!std::isfinite(scale)) return LM_ERR_RANGE;
        for (uint32_t i = 0u; i < 32u; ++i) {
            const float value = input[block * 32u + i];
            if (!std::isfinite(value)) return LM_ERR_RANGE;
            const unsigned char packed_pair = packed[base + 2u + i / 2u];
            const int quantized = static_cast<int>((i & 1u) == 0u ? (packed_pair & 0x0fu) : (packed_pair >> 4u)) - 8;
            sum += scale * static_cast<float>(quantized) * value;
        }
    }
    if (!std::isfinite(sum)) return LM_ERR_RANGE;
    *out = sum;
    return LM_OK;
}

lm_status lm_cpu_dot_q8_0(const lm_tensor *weights, const float *input,
                          uint64_t elements, float *out) {
    if (!weights || !input || !out || elements == 0u || elements % 32u != 0u ||
        weights->quant_format != LM_QUANT_GGML_Q8_0 || weights->dtype != LM_DTYPE_U8)
        return LM_ERR_ARGUMENT;
    if (lm_tensor_validate(weights) != LM_OK) return LM_ERR_RANGE;
    const uint64_t required_bytes = (elements / 32u) * 34u;
    if (required_bytes != weights->bytes) return LM_ERR_CAPACITY;
    const unsigned char *packed = static_cast<const unsigned char *>(weights->data);
    float sum = 0.0f;
    for (uint64_t block = 0u; block < elements / 32u; ++block) {
        const size_t base = static_cast<size_t>(block * 34u);
        const uint16_t scale_bits = static_cast<uint16_t>(packed[base]) | static_cast<uint16_t>(packed[base + 1u] << 8u);
        const float scale = half_to_float(scale_bits);
        if (!std::isfinite(scale)) return LM_ERR_RANGE;
        for (uint32_t i = 0u; i < 32u; ++i) {
            const float value = input[block * 32u + i];
            if (!std::isfinite(value)) return LM_ERR_RANGE;
            const int8_t quantized = static_cast<int8_t>(packed[base + 2u + i]);
            sum += scale * static_cast<float>(quantized) * value;
        }
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
