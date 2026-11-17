#!/bin/bash -login
#SBATCH -A wang-compbio
#SBATCH -p general
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=housing_bm25_all
#SBATCH --output=results/logs/step5-retrieve-%j.out
set -euo pipefail
cd /mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/Multi-hop-statlaw
python3 scripts/evaluate_retrieval.py \
  --questions-zip ../questions.json.zip \
  --index data/statutes_v2.sqlite \
  --top-k 10 \
  --output results/step5_bm25_top10.jsonl
