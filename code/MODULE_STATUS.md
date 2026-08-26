# Tiny LLM engine: live module status

This file is the short operational status companion to `../ENGINEERING_GUIDE.md`. It records what is implemented and validated, rather than the aspirational end state.

| Module | State | Evidence | Limitation / next step |
|---|---|---|---|
| C99 ABI and C++17 control plane | PASS | Clean CMake build, focused CTest suite, and explicit `weight_policy=preserve` default | `quantize-cache` is parsed but returns named unsupported until a calibrated converter exists |
| CLI configuration and probes | PASS | CLI parsing, resolved configuration, trace/probe tests | No model generation or server mode yet |
| Tensor dtype/view/buffer layer | PASS | Dense views plus native packed-block views validate exact byte geometry; concrete GGML Q4_0/Q8_0 validation and direct CPU dot execution pass without expansion | Native model binding reads into caller-owned storage; mmap/device storage, K-quant family, and non-contiguous views next |
| GGUF inspection | PASS | v3 header, bounded metadata, alignment, indexed tensor queries, exact bounded payload validation for native Q4_0/Q8_0 descriptors, lazy binding/read views, expert-count, and experts-per-token tests | Other quantized block types and expert-name/index mapping are still pending; K-quant layouts require separate exact decoders |
| SafeTensors inspection | PASS | Valid descriptor, dtype/shape byte-size, file-bound, overlap rejection, and indexed name/rank/dimensions/type/offset queries | Source dtype is preserved and the native GGUF-only binding rejects unsupported SafeTensors dtypes; explicit quantize-cache conversion is not implemented |
| CPU scalar math | PASS | F32 dot/softmax, decoder golden-logit, and direct packed Q4_0/Q8_0 dot tests | Broad graph/operator coverage, tokenizer, quantized operator coverage beyond Q8_0, and sampling policies are not implemented |
| Paged KV control plane | PASS | Fork, copy-on-write, rollback, release, and stats tests | No K/V payload pages, quantization, eviction, migration, or prefix hashing |
| Replaceable kernel registry | PASS | CPU/Vulkan scalar/DP4 selection, dtype/alignment contracts, and generic operation dispatch tests | Contract metadata is centralized; no full tensor graph scheduler or quantized/routed Vulkan dispatcher yet |
| Vulkan device discovery | PASS | Physical-device and queue discovery on llvmpipe; explicit runtime selection validates index/capability | No target discrete GPU validation; no memory planner |
| Vulkan packed-int8 DP4 dispatch | PASS | Real compute dispatch and generic contract dispatch differentially return expected result on llvmpipe; registry contract declares I8→I32 and 4-byte alignment | Native quantized and routed-expert model integration, benchmark, and discrete-GPU validation pending |
| Cooperative matrices | EXCLUDED | No implementation source or registry path | Remains excluded by design |
| Tiny scalar decoder fixture | PASS | Single-layer/single-head embedding, RMSNorm, optional RoPE, causal attention, residuals, SiLU MLP, logits, reset, and bounds are golden-tested | Deliberately narrow: no tokenizer, multi-layer/GQA mapping, quantized weights, or broad Llama compatibility |
| CPU MoE routing oracle | PASS | Deterministic top-k selection, tie-breaking, all-softmax/selected-only weighting, normalization, selected-output combination, NaN, bound, and GGUF metadata tests | Expert tensor loading, native quantized expert MLP execution, capacity policies, and model-specific router semantics remain pending |
| Lazy/disk-backed residency | IN PROGRESS | Model handles bind validated data bases to range-checked relative tensor spans; supported GGUF Q4_0/Q8_0 descriptors produce exact lazy spans and caller-buffer reads without whole-file loading | Add mmap/window adapters, prefetch, shard manifests, and residency traces; binding remains host-buffer based rather than an execution scheduler |
| OpenAI/Llama server and WebUI adapter | NOT STARTED | — | Add only after CPU generation path is real |
| Context rolling/compression | NOT STARTED | — | Implement rolling window and explicit summary/retrieval policy; do not claim native context extension |
| Direct ROCr/HSA / custom AMDGPU | DEFERRED | — | Start only after CPU/Vulkan contracts and differential tests are stable; no HIP |

## Validation snapshot

The latest clean Vulkan-enabled build passed CTest, including valid/malformed GGUF and SafeTensors fixtures, indexed model descriptors, exact native GGUF Q4_0/Q8_0 payload bounds, lazy caller-buffer binding reads, file-span bounds, the tiny decoder golden-logit fixture, the MoE routing/combination oracle, the quantized tensor contract, and generic scalar/DP4 Vulkan dispatch; SPIR-V disassembly showed compute entry points, and device enumeration succeeded on the available `llvmpipe` Vulkan CPU implementation. A clean no-Vulkan build also passed CTest using the named unsupported stubs. The implementation source under `include/`, `core/`, `cli/`, `tests/`, and `vulkan/` measures 152,128 bytes, far below the approximately 2 MiB source ceiling.
