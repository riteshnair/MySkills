#include "lm/lm.h"

static int valid_op(lm_kernel_op op) {
    return op >= LM_KERNEL_DOT_F32 && op <= LM_KERNEL_DOT_Q4_K;
}

lm_status lm_kernel_contract_get(const lm_kernel_choice *choice, lm_kernel_contract *out_contract) {
    if (!choice || !out_contract) return LM_ERR_ARGUMENT;
    out_contract->input_dtype = LM_DTYPE_F32;
    out_contract->output_dtype = LM_DTYPE_F32;
    out_contract->minimum_alignment = 1u;
    out_contract->deterministic = 1u;
    out_contract->source_id = choice->source_id;
    if (choice->op == LM_KERNEL_DOT_Q4_K) {
        out_contract->input_dtype = LM_DTYPE_U8;
        out_contract->output_dtype = LM_DTYPE_F32;
        out_contract->minimum_alignment = 1u;
    } else if (choice->op == LM_KERNEL_DOT_I8) {
        out_contract->input_dtype = LM_DTYPE_I8;
        if (choice->path == LM_KERNEL_VULKAN_DP4) {
            out_contract->output_dtype = LM_DTYPE_I32;
            out_contract->minimum_alignment = 4u;
        } else {
            out_contract->output_dtype = LM_DTYPE_I32;
        }
    } else if (choice->op != LM_KERNEL_DOT_F32 && choice->op != LM_KERNEL_SOFTMAX_F32) {
        return LM_ERR_UNSUPPORTED;
    }
    if (choice->path == LM_KERNEL_AUTO || !choice->source_id || choice->source_id[0] == '\0') return LM_ERR_STATE;
    return LM_OK;
}

const char *lm_kernel_path_name(lm_kernel_path path) {
    switch (path) {
        case LM_KERNEL_AUTO: return "auto";
        case LM_KERNEL_CPU_SCALAR: return "cpu-scalar";
        case LM_KERNEL_VULKAN_SCALAR: return "vulkan-scalar";
        case LM_KERNEL_VULKAN_DP4: return "vulkan-dp4";
        default: return "unknown";
    }
}

lm_status lm_kernel_select(lm_kernel_op op, lm_kernel_path requested,
                           const lm_kernel_caps *caps, lm_kernel_choice *out_choice) {
    if (!valid_op(op) || !caps || !out_choice) return LM_ERR_ARGUMENT;
    out_choice->op = op;
    out_choice->path = LM_KERNEL_AUTO;
    out_choice->name = "";
    out_choice->source_id = "";

    const int dot_op = op == LM_KERNEL_DOT_I8;
    const int q4_k_op = op == LM_KERNEL_DOT_Q4_K;
    if (requested == LM_KERNEL_CPU_SCALAR || requested == LM_KERNEL_AUTO) {
        if (requested == LM_KERNEL_CPU_SCALAR || !caps->vulkan) {
            out_choice->path = LM_KERNEL_CPU_SCALAR;
            out_choice->name = dot_op ? "cpu-dot-i8" : (q4_k_op ? "cpu-dot-q4-k" : (op == LM_KERNEL_DOT_F32 ? "cpu-dot-f32" : "cpu-softmax-f32"));
            out_choice->source_id = "cpu/reference";
            return LM_OK;
        }
    }
    if (requested == LM_KERNEL_VULKAN_DP4 || requested == LM_KERNEL_AUTO) {
        if (caps->vulkan && caps->shader_int_dot && dot_op) {
            out_choice->path = LM_KERNEL_VULKAN_DP4;
            out_choice->name = "vulkan-dot-i8-dp4";
            out_choice->source_id = "vulkan/dp4";
            return LM_OK;
        }
        if (requested == LM_KERNEL_VULKAN_DP4) return LM_ERR_UNSUPPORTED;
    }
    if (requested == LM_KERNEL_VULKAN_SCALAR || requested == LM_KERNEL_AUTO) {
        if (caps->vulkan) {
            out_choice->path = LM_KERNEL_VULKAN_SCALAR;
            out_choice->name = op == LM_KERNEL_DOT_F32 ? "vulkan-dot-f32" : (op == LM_KERNEL_DOT_I8 ? "vulkan-dot-i8" : (q4_k_op ? "vulkan-dot-q4-k" : "vulkan-softmax-f32"));
            out_choice->source_id = q4_k_op ? "vulkan/q4_k" : "vulkan/scalar";
            return LM_OK;
        }
        if (requested == LM_KERNEL_VULKAN_SCALAR) return LM_ERR_UNSUPPORTED;
    }
    return LM_ERR_UNSUPPORTED;
}
