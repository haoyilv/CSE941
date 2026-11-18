#!/bin/bash -login
#SBATCH -A wang-compbio
#SBATCH -p general-long-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=a100:1
#SBATCH --job-name=housing_iterative
#SBATCH --output=results/logs/iterative-%j.out
set -euo pipefail
cd /mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/Multi-hop-statlaw
unset PYTHONPATH
export HF_HOME=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/models/hf_cache
PY=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/gemma_venv/bin/python
MODEL=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/models/gemma-4-26B-A4B-it
"$PY" scripts/generate_followup_queries.py \
  --questions-zip ../questions.json.zip \
  --split results/step4_split.json \
  --initial-retrieval results/step5_bm25_top10.jsonl \
  --model-path "$MODEL" \
  --first-k 5 \
  --max-passage-chars 400 \
  --output results/iterative_followup_queries.jsonl
python3 scripts/retrieve_followup.py \
  --split results/step4_split.json \
  --index data/statutes_v2.sqlite \
  --initial-retrieval results/step5_bm25_top10.jsonl \
  --followups results/iterative_followup_queries.jsonl \
  --first-k 5 \
  --second-k 5 \
  --output results/step5_iterative_top10.jsonl
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
