#ifndef LM_LM_H
#define LM_LM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LM_ABI_VERSION 1u
#define LM_PATH_MAX 512u

typedef enum lm_status {
    LM_OK = 0,
    LM_ERR_ARGUMENT = -1,
    LM_ERR_PARSE = -2,
    LM_ERR_RANGE = -3,
    LM_ERR_UNSUPPORTED = -4,
    LM_ERR_CAPACITY = -5,
    LM_ERR_STATE = -6,
    LM_ERR_IO = -7
} lm_status;

typedef enum lm_backend_kind {
    LM_BACKEND_AUTO = 0,
    LM_BACKEND_CPU,
    LM_BACKEND_VULKAN,
    LM_BACKEND_ROCR,
    LM_BACKEND_ROCM,
    LM_BACKEND_CUDA,
    LM_BACKEND_OPENVINO,
    LM_BACKEND_DIRECTML
} lm_backend_kind;

typedef enum lm_load_mode {
    LM_LOAD_EAGER = 0,
    LM_LOAD_MMAP,
    LM_LOAD_LAZY,
    LM_LOAD_STREAM
} lm_load_mode;

typedef enum lm_kv_dtype {
    LM_KV_F16 = 0,
    LM_KV_BF16,
    LM_KV_Q8,
    LM_KV_Q6,
    LM_KV_Q4
} lm_kv_dtype;

typedef struct lm_probe {
    uint64_t trace_id;
    uint64_t parent_id;
    uint32_t kind;
    uint32_t stage;
    uint32_t bytes;
    uint32_t flags;
    uint64_t content_hash;
    uint64_t timestamp_ns;
} lm_probe;

typedef void (*lm_probe_sink)(void *user, const lm_probe *probe);

typedef struct lm_config {
    lm_backend_kind backend;
    lm_backend_kind resolved_backend;
    lm_load_mode load_mode;
    lm_kv_dtype kv_dtype;
    uint32_t device_index;
    uint32_t context_tokens;
    uint32_t threads;
    uint32_t kv_page_tokens;
    uint64_t vram_limit_bytes;
    uint64_t host_cache_bytes;
    uint64_t pinned_cache_bytes;
    uint64_t device_cache_bytes;
    uint8_t prefetch;
    uint8_t trace;
    uint8_t deterministic;
    char model_path[LM_PATH_MAX];
} lm_config;

typedef struct lm_runtime lm_runtime;
typedef struct lm_buffer lm_buffer;
typedef struct lm_file lm_file;

lm_status lm_file_open(const char *path, lm_file **out_file);
void lm_file_close(lm_file *file);
lm_status lm_file_size(const lm_file *file, uint64_t *out_bytes);
lm_status lm_file_read(lm_file *file, uint64_t offset, void *dst, size_t bytes);

typedef enum lm_dtype {
    LM_DTYPE_F32 = 0,
    LM_DTYPE_F16,
    LM_DTYPE_BF16,
    LM_DTYPE_I8,
    LM_DTYPE_U8
} lm_dtype;

typedef struct lm_tensor {
    void *data;
    uint64_t bytes;
    uint32_t rank;
    uint32_t dims[8];
    uint32_t strides[8];
    lm_dtype dtype;
} lm_tensor;

size_t lm_dtype_size(lm_dtype dtype);
lm_status lm_tensor_validate(const lm_tensor *tensor);
lm_status lm_tensor_make_view(void *data, uint64_t bytes, lm_dtype dtype,
                              uint32_t rank, const uint32_t *dims,
                              lm_tensor *out_tensor);
lm_status lm_buffer_alloc(uint64_t bytes, lm_buffer **out_buffer);
void lm_buffer_free(lm_buffer *buffer);
lm_status lm_buffer_view(lm_buffer *buffer, lm_tensor *out_tensor);


typedef enum lm_kernel_op {
    LM_KERNEL_DOT_F32 = 0,
    LM_KERNEL_DOT_I8,
    LM_KERNEL_SOFTMAX_F32
} lm_kernel_op;

typedef enum lm_kernel_path {
    LM_KERNEL_AUTO = 0,
    LM_KERNEL_CPU_SCALAR,
    LM_KERNEL_VULKAN_SCALAR,
    LM_KERNEL_VULKAN_DP4
} lm_kernel_path;

typedef struct lm_kernel_caps {
    uint8_t vulkan;
    uint8_t shader_int_dot;
    uint8_t subgroup;
} lm_kernel_caps;

typedef struct lm_kernel_choice {
    lm_kernel_op op;
    lm_kernel_path path;
    const char *name;
    const char *source_id;
} lm_kernel_choice;

lm_status lm_kernel_select(lm_kernel_op op, lm_kernel_path requested,
                           const lm_kernel_caps *caps, lm_kernel_choice *out_choice);
const char *lm_kernel_path_name(lm_kernel_path path);

typedef struct lm_vulkan_device_info {
    char name[128];
    uint32_t api_version;
    uint32_t driver_version;
    uint32_t vendor_id;
    uint32_t device_id;
    uint8_t is_cpu;
    uint8_t shader_int_dot;
    uint8_t subgroup;
} lm_vulkan_device_info;

lm_status lm_vulkan_device_count(uint32_t *out_count);
lm_status lm_vulkan_device_info_get(uint32_t index, lm_vulkan_device_info *out_info);
lm_status lm_vulkan_dot_i8_dp4(const char *spv_path, uint32_t device_index,
                               const uint32_t *a, const uint32_t *b,
                               uint32_t packed_words, int32_t *out_result);


typedef struct lm_kv_cache lm_kv_cache;

typedef struct lm_kv_stats {
    uint32_t page_tokens;
    uint32_t total_pages;
    uint32_t free_pages;
    uint32_t used_pages;
    uint32_t shared_pages;
    uint64_t appended_tokens;
} lm_kv_stats;


typedef enum lm_model_format {
    LM_MODEL_UNKNOWN = 0,
    LM_MODEL_GGUF,
    LM_MODEL_SAFETENSORS
} lm_model_format;

typedef struct lm_model_info {
    lm_model_format format;
    uint32_t version;
    uint64_t file_bytes;
    uint64_t header_bytes;
    uint64_t tensor_count;
} lm_model_info;


void lm_config_init(lm_config *config);
lm_status lm_config_parse_argv(lm_config *config, int argc, char **argv,
                               const char **bad_argument);
const char *lm_status_name(lm_status status);
const char *lm_backend_name(lm_backend_kind backend);
const char *lm_load_mode_name(lm_load_mode mode);
const char *lm_kv_dtype_name(lm_kv_dtype dtype);
const char *lm_model_format_name(lm_model_format format);
lm_status lm_model_inspect(const char *path, lm_model_info *out_info,
                           char *error_text, size_t error_capacity);
lm_status lm_cpu_dot_f32(const float *a, const float *b, size_t count, float *out);
lm_status lm_cpu_softmax_f32(const float *input, float *output, size_t count);

typedef struct lm_cpu_decoder lm_cpu_decoder;

typedef struct lm_cpu_decoder_config {
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t max_context;
    float rms_epsilon;
    float rope_theta;
    uint8_t use_rope;
    const float *embedding; /* vocab_size x hidden_size */
    const float *rms_gamma_1; /* hidden_size */
    const float *wq; /* hidden_size x hidden_size */
    const float *wk; /* hidden_size x hidden_size */
    const float *wv; /* hidden_size x hidden_size */
    const float *wo; /* hidden_size x hidden_size */
    const float *rms_gamma_2; /* hidden_size */
    const float *w1; /* hidden_size x hidden_size */
    const float *w2; /* hidden_size x hidden_size */
    const float *wout; /* hidden_size x vocab_size */
} lm_cpu_decoder_config;

lm_status lm_cpu_decoder_create(const lm_cpu_decoder_config *config, lm_cpu_decoder **out_decoder);
void lm_cpu_decoder_destroy(lm_cpu_decoder *decoder);
lm_status lm_cpu_decoder_reset(lm_cpu_decoder *decoder);
lm_status lm_cpu_decoder_step(lm_cpu_decoder *decoder, uint32_t token_id, float *out_logits, size_t logits_count);
uint32_t lm_cpu_decoder_position(const lm_cpu_decoder *decoder);

lm_status lm_kv_cache_create(uint32_t page_count, uint32_t page_tokens,
                             lm_kv_cache **out_cache);
void lm_kv_cache_destroy(lm_kv_cache *cache);
lm_status lm_kv_cache_append(lm_kv_cache *cache, uint32_t *page_id,
                             uint32_t token_count);
lm_status lm_kv_cache_fork(lm_kv_cache *cache, uint32_t source_page,
                           uint32_t *out_page);
lm_status lm_kv_cache_rollback(lm_kv_cache *cache, uint32_t page_id,
                               uint32_t token_count);
lm_status lm_kv_cache_release(lm_kv_cache *cache, uint32_t page_id);
lm_status lm_kv_cache_get_stats(const lm_kv_cache *cache, lm_kv_stats *out_stats);

lm_status lm_runtime_create(const lm_config *config, lm_runtime **out_runtime);
void lm_runtime_destroy(lm_runtime *runtime);
void lm_runtime_set_probe_sink(lm_runtime *runtime, lm_probe_sink sink, void *user);
lm_status lm_runtime_emit_probe(lm_runtime *runtime, const lm_probe *probe);
lm_status lm_runtime_dump_config(const lm_runtime *runtime, FILE *out);

#ifdef __cplusplus
}
#endif

#endif
