# Tiny LLM engine: live module status

This file is the short operational status companion to `../ENGINEERING_GUIDE.md`. It records what is implemented and validated, rather than the aspirational end state.

| Module | State | Evidence | Limitation / next step |
|---|---|---|---|
| C99 ABI and C++17 control plane | PASS | Clean CMake build and focused CTest suite | Runtime resolves CPU or validated Vulkan device; graph execution remains separate |
| CLI configuration and probes | PASS | CLI parsing, resolved configuration, trace/probe tests | No model generation or server mode yet |
| Tensor dtype/view/buffer layer | PASS | Shape/stride, byte-capacity, allocation, and ownership tests | Host-owned contiguous buffers only; device storage and non-contiguous views next |
| GGUF inspection | PASS | v3 header, bounded metadata, alignment, tensor descriptor, dimension, type, and offset tests | Tensor byte-size validation for every quantized block type is still pending |
| SafeTensors inspection | PASS | Valid descriptor, dtype/shape byte-size, file-bound, and overlap rejection fixtures | Shard manifest handling and zero-copy tensor span API are still pending |
| CPU scalar math | PASS | Dot-product, stable softmax, and decoder golden-logit tests | Broad graph/operator coverage, tokenizer, and sampling policies are not implemented |
| Paged KV control plane | PASS | Fork, copy-on-write, rollback, release, and stats tests | No K/V payload pages, quantization, eviction, migration, or prefix hashing |
| Replaceable kernel registry | PASS | CPU/Vulkan scalar/DP4 selection and dtype/alignment contract tests | Contract metadata is centralized; no generic graph dispatcher yet |
| Vulkan device discovery | PASS | Physical-device and queue discovery on llvmpipe; explicit runtime selection validates index/capability | No target discrete GPU validation; no memory planner |
| Vulkan packed-int8 DP4 dispatch | PASS | Real compute dispatch differentially returns expected result on llvmpipe; registry contract declares I8→I32 and 4-byte alignment | Manual byte extraction/multiplication, not guaranteed hardware dot intrinsic; benchmark and GPU validation pending |
| Cooperative matrices | EXCLUDED | No implementation source or registry path | Remains excluded by design |
| Tiny scalar decoder fixture | PASS | Single-layer/single-head embedding, RMSNorm, optional RoPE, causal attention, residuals, SiLU MLP, logits, reset, and bounds are golden-tested | Deliberately narrow: no tokenizer, multi-layer/GQA mapping, quantized weights, or broad Llama compatibility |
| Lazy/disk-backed residency | IN PROGRESS | Read-only bounded `lm_file_open/read` seam is implemented and tested | Add mmap/window adapters, tensor spans, prefetch, and residency traces |
| OpenAI/Llama server and WebUI adapter | NOT STARTED | — | Add only after CPU generation path is real |
| Context rolling/compression | NOT STARTED | — | Implement rolling window and explicit summary/retrieval policy; do not claim native context extension |
| Direct ROCr/HSA / custom AMDGPU | DEFERRED | — | Start only after CPU/Vulkan contracts and differential tests are stable; no HIP |

## Validation snapshot

The latest clean Vulkan-enabled build passed CTest, including valid/malformed GGUF and SafeTensors fixtures plus the tiny decoder golden-logit fixture; SPIR-V disassembly showed a compute entry point, and device enumeration succeeded on the available `llvmpipe` Vulkan CPU implementation. A no-Vulkan build also passed CTest using the named unsupported stubs. The implementation source under `include/`, `core/`, `cli/`, `tests/`, and `vulkan/` remains far below the approximately 2 MiB source ceiling.
