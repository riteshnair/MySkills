# Tiny LLM engine: live module status

This file is the short operational status companion to `../ENGINEERING_GUIDE.md`. It records what is implemented and validated, rather than the aspirational end state.

| Module | State | Evidence | Limitation / next step |
|---|---|---|---|
| C99 ABI and C++17 control plane | PASS | Clean CMake build, focused CTest suite, and explicit `weight_policy=preserve` default | `quantize-cache` is parsed but returns named unsupported until a calibrated converter exists |
| CLI configuration and probes | PASS | CLI parsing, resolved configuration, trace/probe tests | No model generation or server mode yet |
| Tensor dtype/view/buffer layer | PASS | Dense views plus native packed-block views validate exact byte geometry; no expansion is performed | Host-owned storage only; device storage, concrete quant formats, and non-contiguous views next |
| GGUF inspection | PASS | v3 header, bounded metadata, alignment, indexed tensor name/rank/dimensions/type/offset queries, expert-count, and experts-per-token tests | Tensor byte-size validation for every quantized block type and expert-name/index mapping are still pending |
| SafeTensors inspection | PASS | Valid descriptor, dtype/shape byte-size, file-bound, overlap rejection, and indexed name/rank/dimensions/type/offset queries | Source dtype is preserved; explicit SafeTensors quantize-cache conversion is not implemented |
| CPU scalar math | PASS | Dot-product, stable softmax, and decoder golden-logit tests | Broad graph/operator coverage, tokenizer, and sampling policies are not implemented |
| Paged KV control plane | PASS | Fork, copy-on-write, rollback, release, and stats tests | No K/V payload pages, quantization, eviction, migration, or prefix hashing |
| Replaceable kernel registry | PASS | CPU/Vulkan scalar/DP4 selection and dtype/alignment contract tests | Contract metadata is centralized; no generic graph dispatcher yet |
| Vulkan device discovery | PASS | Physical-device and queue discovery on llvmpipe; explicit runtime selection validates index/capability | No target discrete GPU validation; no memory planner |
| Vulkan packed-int8 DP4 dispatch | PASS | Real compute dispatch differentially returns expected result on llvmpipe; registry contract declares I8→I32 and 4-byte alignment | Native quantized and routed-expert model integration, benchmark, and discrete-GPU validation pending |
| Cooperative matrices | EXCLUDED | No implementation source or registry path | Remains excluded by design |
| Tiny scalar decoder fixture | PASS | Single-layer/single-head embedding, RMSNorm, optional RoPE, causal attention, residuals, SiLU MLP, logits, reset, and bounds are golden-tested | Deliberately narrow: no tokenizer, multi-layer/GQA mapping, quantized weights, or broad Llama compatibility |
| CPU MoE routing oracle | PASS | Deterministic top-k selection, tie-breaking, all-softmax/selected-only weighting, normalization, selected-output combination, NaN, bound, and GGUF metadata tests | Expert tensor loading, routed expert MLP execution, capacity policies, and model-specific router semantics remain pending |
| Lazy/disk-backed residency | IN PROGRESS | Model handles bind validated data bases to range-checked relative tensor spans; GGUF and SafeTensors descriptors can be queried without reading tensor payloads | Add mmap/window adapters, prefetch, shard manifests, and residency traces |
| OpenAI/Llama server and WebUI adapter | NOT STARTED | — | Add only after CPU generation path is real |
| Context rolling/compression | NOT STARTED | — | Implement rolling window and explicit summary/retrieval policy; do not claim native context extension |
| Direct ROCr/HSA / custom AMDGPU | DEFERRED | — | Start only after CPU/Vulkan contracts and differential tests are stable; no HIP |

## Validation snapshot

The latest clean Vulkan-enabled build passed CTest, including valid/malformed GGUF and SafeTensors fixtures, indexed model descriptors, file-span bounds, the tiny decoder golden-logit fixture, the MoE routing/combination oracle, and the quantized tensor contract; SPIR-V disassembly showed a compute entry point, and device enumeration succeeded on the available `llvmpipe` Vulkan CPU implementation. A no-Vulkan build also passed CTest using the named unsupported stubs. The implementation source under `include/`, `core/`, `cli/`, `tests/`, and `vulkan/` remains far below the approximately 2 MiB source ceiling.
