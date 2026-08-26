#include "lm/lm.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

struct lm_cpu_decoder {
    lm_cpu_decoder_config config;
    uint32_t position;
    std::vector<float> keys;
    std::vector<float> values;
    std::vector<float> scratch;
};

namespace {

bool finite_array(const float *data, size_t count) {
    if (!data) return false;
    for (size_t i = 0u; i < count; ++i) if (!std::isfinite(data[i])) return false;
    return true;
}

bool matrix_finite(const float *data, uint32_t rows, uint32_t columns) {
    return finite_array(data, static_cast<size_t>(rows) * columns);
}

void matvec(const float *input, const float *matrix, uint32_t rows, uint32_t columns, float *output) {
    for (uint32_t column = 0u; column < columns; ++column) {
        float sum = 0.0f;
        for (uint32_t row = 0u; row < rows; ++row) sum += input[row] * matrix[static_cast<size_t>(row) * columns + column];
        output[column] = sum;
    }
}

bool rms_norm(const float *input, uint32_t size, float epsilon, const float *gamma, float *output) {
    if (!(epsilon > 0.0f) || !std::isfinite(epsilon)) return false;
    float mean_square = 0.0f;
    for (uint32_t i = 0u; i < size; ++i) mean_square += input[i] * input[i];
    mean_square /= static_cast<float>(size);
    const float scale = 1.0f / std::sqrt(mean_square + epsilon);
    if (!std::isfinite(scale)) return false;
    for (uint32_t i = 0u; i < size; ++i) output[i] = input[i] * scale * gamma[i];
    return finite_array(output, size);
}

void apply_rope(float *vector, uint32_t size, uint32_t position, float theta) {
    for (uint32_t i = 0u; i + 1u < size; i += 2u) {
        const float frequency = std::pow(theta, -static_cast<float>(i) / static_cast<float>(size));
        const float angle = static_cast<float>(position) * frequency;
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float first = vector[i];
        const float second = vector[i + 1u];
        vector[i] = first * cosine - second * sine;
        vector[i + 1u] = first * sine + second * cosine;
    }
}

} // namespace

lm_status lm_cpu_decoder_create(const lm_cpu_decoder_config *config, lm_cpu_decoder **out_decoder) {
    if (!config || !out_decoder || config->vocab_size == 0u || config->hidden_size == 0u || config->max_context == 0u ||
        config->hidden_size > 4096u || config->max_context > 1u << 20u || !(config->rms_epsilon > 0.0f) ||
        !std::isfinite(config->rms_epsilon) || (config->use_rope && (!(config->rope_theta > 1.0f) || !std::isfinite(config->rope_theta))))
        return LM_ERR_ARGUMENT;
    if (!finite_array(config->embedding, static_cast<size_t>(config->vocab_size) * config->hidden_size) ||
        !finite_array(config->rms_gamma_1, config->hidden_size) || !matrix_finite(config->wq, config->hidden_size, config->hidden_size) ||
        !matrix_finite(config->wk, config->hidden_size, config->hidden_size) || !matrix_finite(config->wv, config->hidden_size, config->hidden_size) ||
        !matrix_finite(config->wo, config->hidden_size, config->hidden_size) || !finite_array(config->rms_gamma_2, config->hidden_size) ||
        !matrix_finite(config->w1, config->hidden_size, config->hidden_size) || !matrix_finite(config->w2, config->hidden_size, config->hidden_size) ||
        !matrix_finite(config->wout, config->hidden_size, config->vocab_size)) return LM_ERR_ARGUMENT;
    try {
        lm_cpu_decoder *decoder = new lm_cpu_decoder();
        decoder->config = *config;
        decoder->position = 0u;
        const size_t cache_elements = static_cast<size_t>(config->max_context) * config->hidden_size;
        decoder->keys.assign(cache_elements, 0.0f);
        decoder->values.assign(cache_elements, 0.0f);
        decoder->scratch.resize(static_cast<size_t>(config->hidden_size) * 9u + static_cast<size_t>(config->max_context) * 2u);
        *out_decoder = decoder;
        return LM_OK;
    } catch (const std::bad_alloc &) {
        *out_decoder = nullptr;
        return LM_ERR_CAPACITY;
    }
}

void lm_cpu_decoder_destroy(lm_cpu_decoder *decoder) {
    delete decoder;
}

lm_status lm_cpu_decoder_reset(lm_cpu_decoder *decoder) {
    if (!decoder) return LM_ERR_ARGUMENT;
    decoder->position = 0u;
    std::fill(decoder->keys.begin(), decoder->keys.end(), 0.0f);
    std::fill(decoder->values.begin(), decoder->values.end(), 0.0f);
    return LM_OK;
}

lm_status lm_cpu_decoder_step(lm_cpu_decoder *decoder, uint32_t token_id, float *out_logits, size_t logits_count) {
    if (!decoder || !out_logits || logits_count < decoder->config.vocab_size) return LM_ERR_ARGUMENT;
    if (token_id >= decoder->config.vocab_size) return LM_ERR_RANGE;
    if (decoder->position >= decoder->config.max_context) return LM_ERR_CAPACITY;
    const uint32_t h = decoder->config.hidden_size;
    float *x = decoder->scratch.data();
    float *norm1 = x + h;
    float *q = norm1 + h;
    float *k = q + h;
    float *v = k + h;
    float *attention = v + h;
    float *residual = attention + h;
    float *norm2 = residual + h;
    float *up = norm2 + h;
    float *scores = up + h;
    float *weights = scores + decoder->config.max_context;
    std::memcpy(x, decoder->config.embedding + static_cast<size_t>(token_id) * h, sizeof(float) * h);
    if (!rms_norm(x, h, decoder->config.rms_epsilon, decoder->config.rms_gamma_1, norm1)) return LM_ERR_RANGE;
    matvec(norm1, decoder->config.wq, h, h, q);
    matvec(norm1, decoder->config.wk, h, h, k);
    matvec(norm1, decoder->config.wv, h, h, v);
    if (decoder->config.use_rope) {
        apply_rope(q, h, decoder->position, decoder->config.rope_theta);
        apply_rope(k, h, decoder->position, decoder->config.rope_theta);
    }
    std::memcpy(decoder->keys.data() + static_cast<size_t>(decoder->position) * h, k, sizeof(float) * h);
    std::memcpy(decoder->values.data() + static_cast<size_t>(decoder->position) * h, v, sizeof(float) * h);
    const float scale = 1.0f / std::sqrt(static_cast<float>(h));
    for (uint32_t t = 0u; t <= decoder->position; ++t) {
        const float *key = decoder->keys.data() + static_cast<size_t>(t) * h;
        float dot = 0.0f;
        for (uint32_t i = 0u; i < h; ++i) dot += q[i] * key[i];
        scores[t] = dot * scale;
    }
    if (lm_cpu_softmax_f32(scores, weights, decoder->position + 1u) != LM_OK) return LM_ERR_RANGE;
    std::fill(attention, attention + h, 0.0f);
    for (uint32_t t = 0u; t <= decoder->position; ++t) {
        const float *value = decoder->values.data() + static_cast<size_t>(t) * h;
        for (uint32_t i = 0u; i < h; ++i) attention[i] += weights[t] * value[i];
    }
    matvec(attention, decoder->config.wo, h, h, residual);
    for (uint32_t i = 0u; i < h; ++i) residual[i] += x[i];
    if (!rms_norm(residual, h, decoder->config.rms_epsilon, decoder->config.rms_gamma_2, norm2)) return LM_ERR_RANGE;
    matvec(norm2, decoder->config.w1, h, h, up);
    for (uint32_t i = 0u; i < h; ++i) up[i] = up[i] / (1.0f + std::exp(-up[i]));
    matvec(up, decoder->config.w2, h, h, attention);
    for (uint32_t i = 0u; i < h; ++i) residual[i] += attention[i];
    matvec(residual, decoder->config.wout, h, decoder->config.vocab_size, out_logits);
    if (!finite_array(out_logits, decoder->config.vocab_size)) return LM_ERR_RANGE;
    ++decoder->position;
    return LM_OK;
}

uint32_t lm_cpu_decoder_position(const lm_cpu_decoder *decoder) {
    return decoder ? decoder->position : 0u;
}
