#include "lm/lm.h"

static int valid_op(lm_kernel_op op) {
    return op >= LM_KERNEL_DOT_F32 && op <= LM_KERNEL_SOFTMAX_F32;
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
    if (requested == LM_KERNEL_CPU_SCALAR || requested == LM_KERNEL_AUTO) {
        if (requested == LM_KERNEL_CPU_SCALAR || !caps->vulkan) {
            out_choice->path = LM_KERNEL_CPU_SCALAR;
            out_choice->name = dot_op ? "cpu-dot-i8" : (op == LM_KERNEL_DOT_F32 ? "cpu-dot-f32" : "cpu-softmax-f32");
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
            out_choice->name = op == LM_KERNEL_DOT_F32 ? "vulkan-dot-f32" : (op == LM_KERNEL_DOT_I8 ? "vulkan-dot-i8" : "vulkan-softmax-f32");
            out_choice->source_id = "vulkan/scalar";
            return LM_OK;
        }
        if (requested == LM_KERNEL_VULKAN_SCALAR) return LM_ERR_UNSUPPORTED;
    }
    return LM_ERR_UNSUPPORTED;
}
