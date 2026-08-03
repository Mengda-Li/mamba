#!/usr/bin/env bash

set -euo pipefail

document_start="${1:-0}"
document_stop="${2:-32}"
output_dir="${3:-experiments_2027}"
model_root="${MODEL_ROOT:-models}"
mamba2_checkpoint="${4:-${MAMBA2_MODEL_PATH:-${model_root}/Mamba-Codestral-7B-v0.1}}"
lengths=(64 128 256 384 512 768 1024)

mkdir -p "${output_dir}/logs"

for ((document = document_start; document < document_stop; document++)); do
  for length in "${lengths[@]}"; do
    unit_log="${output_dir}/logs/mamba2_doc${document}_L${length}_isolated.log"
    printf '[%s] starting document=%d length=%d\n' \
      "$(date --iso-8601=seconds)" "${document}" "${length}"
    if ! PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python3 run_structured_experiments.py \
        --model mamba2 \
        --checkpoint "${mamba2_checkpoint}" \
        --document-start "${document}" \
        --document-stop "$((document + 1))" \
        --lengths "${length}" \
        --out-dir "${output_dir}" \
        --no-validate-exact-memory \
        >"${unit_log}" 2>&1; then
      printf '[%s] failed document=%d length=%d; tail of %s follows\n' \
        "$(date --iso-8601=seconds)" "${document}" "${length}" "${unit_log}" >&2
      tail -n 80 "${unit_log}" >&2
      exit 1
    fi
    printf '[%s] completed document=%d length=%d\n' \
      "$(date --iso-8601=seconds)" "${document}" "${length}"
  done
done

printf '[%s] isolated Mamba sweep complete\n' "$(date --iso-8601=seconds)"
