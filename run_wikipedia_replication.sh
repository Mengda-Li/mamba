#!/usr/bin/env bash

set -euo pipefail

output_dir="experiments_2027_wikipedia_replication"
manifest="${output_dir}/sample_manifest.jsonl"
model_root="${MODEL_ROOT:-models}"
retnet_checkpoint="${RETNET_MODEL_PATH:-${model_root}/retnet-1.3B-100B}"
mamba2_checkpoint="${MAMBA2_MODEL_PATH:-${model_root}/Mamba-Codestral-7B-v0.1}"
mkdir -p "${output_dir}/logs"

CUDA_VISIBLE_DEVICES=0 EXPERIMENT_CORPUS=wikimedia-wikipedia-20231101.en \
  .venv/bin/python3 run_structured_experiments.py \
  --model retnet --checkpoint "${retnet_checkpoint}" \
  --manifest "${manifest}" --out-dir "${output_dir}" \
  --layers 23 --no-validate-exact-memory \
  >"${output_dir}/logs/retnet_final.log" 2>&1 &
retnet_pid=$!

CUDA_VISIBLE_DEVICES=1 EXPERIMENT_CORPUS=wikimedia-wikipedia-20231101.en \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash run_mamba_isolated_replication.sh 0 16 "${output_dir}" "${mamba2_checkpoint}" \
  >"${output_dir}/logs/mamba_docs_00_15.log" 2>&1 &
mamba_a_pid=$!

CUDA_VISIBLE_DEVICES=2 EXPERIMENT_CORPUS=wikimedia-wikipedia-20231101.en \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash run_mamba_isolated_replication.sh 16 32 "${output_dir}" "${mamba2_checkpoint}" \
  >"${output_dir}/logs/mamba_docs_16_31.log" 2>&1 &
mamba_b_pid=$!

printf '%s\n' "${retnet_pid}" >"${output_dir}/retnet.pid"
printf '%s\n' "${mamba_a_pid}" >"${output_dir}/mamba_a.pid"
printf '%s\n' "${mamba_b_pid}" >"${output_dir}/mamba_b.pid"
wait "${retnet_pid}" "${mamba_a_pid}" "${mamba_b_pid}"
touch "${output_dir}/GPU_RUNS_COMPLETE"
