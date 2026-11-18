#!/bin/bash -login
#SBATCH -A wang-compbio
#SBATCH -p general
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=housing_verify_gold
#SBATCH --output=results/logs/step5-gold-verify-%j.out
set -euo pipefail
cd /mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/Multi-hop-statlaw
python3 scripts/verify_gold_full.py \
  --audit results/step5_gold_full_length_audit.json \
  --features results/step5_gold_full_features.jsonl
python3 scripts/compare_step5.py \
  --split results/step4_split.json \
  --question-predictions results/step4_probe_predictions.jsonl \
  --gold-predictions results/step5_gold_full_predictions.jsonl \
  --retrieved-predictions results/step5_retrieved_predictions.jsonl \
  --retrieval results/step5_bm25_top10.jsonl \
  --gold-evidence-kind complete_excerpts \
  --output results/step5_gold_full_comparison.json
