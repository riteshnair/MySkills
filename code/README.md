# Tiny LLM Engine — Code Area

This directory contains the first buildable vertical slice of the CPU-first, Vulkan-primary modular LLM engine. Start with [`../ENGINEERING_GUIDE.md`](../ENGINEERING_GUIDE.md), the live [`MODULE_STATUS.md`](MODULE_STATUS.md), and the local skill bundle at [`skills/README.md`](skills/README.md).

## Current slice

The current slice provides a strict C99-compatible public ABI, C++17 orchestration, CLI configuration, capability-driven CPU fallback selection, compact probe delivery, bounded GGUF/SafeTensors structural inspection, scalar dot-product and softmax reference primitives, a paged KV-cache control plane with fork/rollback/release semantics, Vulkan device discovery, and a real packed-int8 DP4 compute dispatch with CPU differential testing.

It intentionally does **not** claim to run full transformer inference. The Vulkan backend currently proves device discovery and one DP4 kernel path; it is not yet a transformer graph executor. Unsupported accelerator requests return `LM_ERR_UNSUPPORTED` instead of silently pretending to work. The next slice should add tensor storage/views, one complete small decoder model on scalar CPU, and stronger GGUF/SafeTensors tensor metadata parsing.

## Build

```sh
cmake -S . -B build -DLM_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Run the control-plane CLI:

```sh
./build/tiny-lm \
  --backend cpu \
  --load lazy \
  --kv-dtype q8 \
  --context 2048 \
  --threads 2 \
  --trace \
  --model fixture.gguf
```

Use `--help` for the supported switches. Use a requested accelerator such as `--backend vulkan` to verify the explicit unsupported-backend gate in this slice.

## Source layout

| Path | Responsibility |
|---|---|
| `include/lm/lm.h` | Stable public C ABI |
| `core/lm.cpp` | Configuration, backend selection, runtime and probes |
| `core/model.cpp` | Bounded model inspection and scalar CPU primitives |
| `core/kv.cpp` | Paged KV-cache control plane |
| `core/kernel.cpp` | Replaceable CPU/Vulkan/DP4 kernel registry |
| `vulkan/device.cpp` | Vulkan device discovery |
| `vulkan/dp4.cpp` | Reference packed-int8 DP4 dispatch |
| `vulkan/shaders/dot_i8_dp4.comp` | DP4 compute shader source |
| `cli/main.cpp` | Local CLI and resolved-config output |
| `tests/test_core.cpp` | Focused unit and smoke tests |
| `CMakeLists.txt` | Minimal build definition |

Generated files belong in `build/`, which is ignored by Git. Keep model files outside the repository unless they are small, synthetic test fixtures.

## Development contract

Every new module must define inputs, outputs, ownership, synchronization, errors, backend applicability, budgets, tests, CLI fields, acceptance criteria, and exclusions before implementation. Keep the scalar CPU path as the reference. Add Vulkan or vendor variants only after a differential test exists. Use the probe bus and deterministic fixtures to locate the first divergent boundary.
