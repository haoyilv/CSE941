#!/bin/bash -login
#SBATCH -A wang-compbio
#SBATCH -p general-long-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=a100:1
#SBATCH --job-name=housing_step4
#SBATCH --output=results/logs/step4-%j.out
set -euo pipefail
cd /mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/Multi-hop-statlaw
unset PYTHONPATH
export HF_HOME=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/models/hf_cache
PY=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/gemma_venv/bin/python
MODEL=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/models/gemma-4-26B-A4B-it
if [[ -n "${SMOKE_LIMIT:-}" ]]; then
  "$PY" scripts/extract_gemma_features.py \
    --questions-zip ../questions.json.zip \
    --model-path "$MODEL" \
    --limit "$SMOKE_LIMIT" \
    --output results/step4_smoke_features.jsonl
else
  "$PY" scripts/extract_gemma_features.py \
    --questions-zip ../questions.json.zip \
    --model-path "$MODEL" \
    --output results/step4_gemma_features.jsonl
  "$PY" scripts/train_gemma_probe.py \
    --questions-zip ../questions.json.zip \
    --split results/step4_split.json \
    --features results/step4_gemma_features.jsonl \
    --output results/step4_probe_summary.json
fi
