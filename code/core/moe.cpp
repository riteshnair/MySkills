#include "lm/lm.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace {

bool finite_logits(const float *logits, uint32_t count) {
    if (!logits) return false;
    for (uint32_t i = 0u; i < count; ++i) if (!std::isfinite(logits[i])) return false;
    return true;
}

int already_selected(const lm_moe_route *route, uint32_t id, uint32_t count) {
    for (uint32_t i = 0u; i < count; ++i) if (route->selected[i] == id) return 1;
    return 0;
}

} // namespace

lm_status lm_cpu_moe_route(const float *router_logits, uint32_t expert_count,
                           uint32_t experts_per_token, lm_moe_route_policy policy,
                           lm_moe_route *out_route) {
    if (!router_logits || !out_route || expert_count == 0u || expert_count > 1u << 20u ||
        experts_per_token == 0u || experts_per_token > 16u || experts_per_token > expert_count ||
        (policy != LM_MOE_SOFTMAX_ALL_THEN_TOPK && policy != LM_MOE_SOFTMAX_SELECTED_ONLY))
        return LM_ERR_ARGUMENT;
    if (!finite_logits(router_logits, expert_count)) return LM_ERR_RANGE;
    std::memset(out_route, 0, sizeof(*out_route));
    out_route->expert_count = expert_count;
    out_route->experts_per_token = experts_per_token;
    for (uint32_t slot = 0u; slot < experts_per_token; ++slot) {
        int best = -1;
        for (uint32_t expert = 0u; expert < expert_count; ++expert) {
            if (already_selected(out_route, expert, slot)) continue;
            if (best < 0 || router_logits[expert] > router_logits[static_cast<uint32_t>(best)] ||
                (router_logits[expert] == router_logits[static_cast<uint32_t>(best)] && expert < static_cast<uint32_t>(best)))
                best = static_cast<int>(expert);
        }
        if (best < 0) return LM_ERR_STATE;
        out_route->selected[slot] = static_cast<uint32_t>(best);
    }

    float max_value = -std::numeric_limits<float>::infinity();
    if (policy == LM_MOE_SOFTMAX_ALL_THEN_TOPK) {
        for (uint32_t expert = 0u; expert < expert_count; ++expert) max_value = std::fmax(max_value, router_logits[expert]);
    } else {
        for (uint32_t slot = 0u; slot < experts_per_token; ++slot)
            max_value = std::fmax(max_value, router_logits[out_route->selected[slot]]);
    }
    float sum = 0.0f;
    for (uint32_t slot = 0u; slot < experts_per_token; ++slot) {
        out_route->weights[slot] = std::exp(router_logits[out_route->selected[slot]] - max_value);
        sum += out_route->weights[slot];
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) return LM_ERR_RANGE;
    for (uint32_t slot = 0u; slot < experts_per_token; ++slot) out_route->weights[slot] /= sum;
    return LM_OK;
}
