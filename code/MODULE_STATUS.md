# Tiny LLM engine: live module status

This file is the short operational status companion to `../ENGINEERING_GUIDE.md`. It records what is implemented and validated, rather than the aspirational end state.

| Module | State | Evidence | Limitation / next step |
|---|---|---|---|
| C99 ABI and C++17 control plane | PASS | Clean CMake build and focused CTest suite | Runtime resolves CPU only; integrate capability-driven Vulkan execution later |
| CLI configuration and probes | PASS | CLI parsing, resolved configuration, trace/probe tests | No model generation or server mode yet |
| Tensor dtype/view/buffer layer | PASS | Shape/stride, byte-capacity, allocation, and ownership tests | Host-owned contiguous buffers only; mapped-file and device storage seams next |
| GGUF inspection | PASS | Valid/invalid header fixture tests | Descriptor enumeration and exact tensor-offset validation not implemented |
| SafeTensors inspection | PASS | Header length/JSON structural checks | Full bounded JSON tensor metadata parser and shards not implemented |
| CPU scalar math | PASS | Dot-product and stable softmax tests | Transformer graph, tokenizer, logits, sampling not implemented |
| Paged KV control plane | PASS | Fork, copy-on-write, rollback, release, and stats tests | No K/V payload pages, quantization, eviction, migration, or prefix hashing |
| Replaceable kernel registry | PASS | CPU/Vulkan scalar/DP4 selection tests | Only isolated operation selection; no graph dispatcher yet |
| Vulkan device discovery | PASS | Physical-device and queue discovery on llvmpipe | No target discrete GPU validation; no memory planner |
| Vulkan packed-int8 DP4 dispatch | PASS | Real compute dispatch differentially returns expected result on llvmpipe | Manual byte extraction/multiplication, not guaranteed hardware dot intrinsic; benchmark and GPU validation pending |
| Cooperative matrices | EXCLUDED | No implementation source or registry path | Remains excluded by design |
| Full transformer inference | NOT STARTED | — | Build a tiny specified decoder-only CPU fixture first |
| Lazy/disk-backed residency | NOT STARTED | — | Add bounded file mapping, tensor spans, prefetch, and residency traces |
| OpenAI/Llama server and WebUI adapter | NOT STARTED | — | Add only after CPU generation path is real |
| Context rolling/compression | NOT STARTED | — | Implement rolling window and explicit summary/retrieval policy; do not claim native context extension |
| Direct ROCr/HSA / custom AMDGPU | DEFERRED | — | Start only after CPU/Vulkan contracts and differential tests are stable; no HIP |

## Validation snapshot

The latest clean Vulkan-enabled build passed CTest, SPIR-V disassembly showed a compute entry point, and device enumeration succeeded on the available `llvmpipe` Vulkan CPU implementation. A no-Vulkan build also passed CTest using the named unsupported stubs. The implementation source under `include/`, `core/`, `cli/`, `tests/`, and `vulkan/` remains far below the approximately 2 MiB source ceiling.
