#!/bin/bash -login
#SBATCH -A wang-compbio
#SBATCH -p general-long-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=a100:1
#SBATCH --job-name=housing_gold_full
#SBATCH --output=results/logs/step5-gold-full-%j.out
set -euo pipefail
cd /mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/Multi-hop-statlaw
unset PYTHONPATH
export HF_HOME=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/models/hf_cache
PY=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/gemma_venv/bin/python
MODEL=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/models/gemma-4-26B-A4B-it
"$PY" scripts/extract_gemma_features.py \
  --questions-zip ../questions.json.zip \
  --model-path "$MODEL" \
  --evidence-mode gold \
  --strict-untruncated \
  --batch-size 1 \
  --max-input-tokens 8192 \
  --max-passages 15 \
  --max-passage-chars 20000 \
  --output results/step5_gold_full_features.jsonl
"$PY" scripts/train_gemma_probe.py \
  --questions-zip ../questions.json.zip \
  --split results/step4_split.json \
  --features results/step5_gold_full_features.jsonl \
  --output results/step5_gold_full_summary.json
