#!/bin/bash -login
#SBATCH -A wang-compbio
#SBATCH -p general-long-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=a100:1
#SBATCH --job-name=housing_iter_answer
#SBATCH --output=results/logs/iterative-answer-%j.out
set -euo pipefail
cd /mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/Multi-hop-statlaw
unset PYTHONPATH
export HF_HOME=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/models/hf_cache
PY=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/gemma_venv/bin/python
MODEL=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/models/gemma-4-26B-A4B-it
"$PY" scripts/run_reasoning_eval.py \
  --questions-zip ../questions.json.zip \
  --split results/step4_split.json \
  --retrieval results/step5_iterative_top10.jsonl \
  --model-path "$MODEL" \
  --evidence-mode bm25 \
  --prompt-style structured \
  --retrieval-strategy iterative \
  --top-k 10 \
  --bm25-chars 400 \
  --max-input-tokens 8192 \
  --max-new-tokens 384 \
  --output results/directqa_iterative_structured.jsonl
python3 scripts/score_predictions.py results/directqa_iterative_structured.jsonl \
  --split results/step4_split.json \
  --output results/directqa_iterative_structured_summary.json
