# Model format implementation notes

The parser implementation is based on the public specifications accessed 2026-08-26.

## GGUF

The GGUF specification defines a little-endian v3 header with `magic`, `version`, `tensor_count`, and `metadata_kv_count`, followed by metadata key/value records and tensor info records. Strings carry a uint64 byte length. Tensor info contains a name, dimension count, uint64 dimensions, a uint32 type, and a uint64 offset relative to the aligned tensor-data region. The default global alignment is 32 bytes; `general.alignment` may override it but must be a multiple of 8. Metadata values include scalar types and recursively nested arrays.

Source: [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).

## SafeTensors

The SafeTensors container begins with an 8-byte little-endian JSON-header length, then the JSON header bytes, then raw tensor data. Each tensor descriptor contains `dtype`, `shape`, and `data_offsets`; offsets are relative to the start of the data section. The header must be a JSON object, and tensor data ranges must be validated against the file bounds and against each other before exposing a view.

Source: [SafeTensors documentation](https://huggingface.co/docs/safetensors/en/index).

## Implementation boundary

The current engine only has structural SafeTensors checks and GGUF count checks. The next parser slice must use bounded reads, reject overflow, reject duplicate or overlapping tensor ranges, cap metadata nesting/counts, and avoid adding a general-purpose JSON dependency to the mandatory core.
