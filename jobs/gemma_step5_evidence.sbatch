#!/bin/bash -login
#SBATCH -A wang-compbio
#SBATCH -p general-long-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=a100:1
#SBATCH --job-name=housing_evidence
#SBATCH --output=results/logs/step5-%j.out
set -euo pipefail
cd /mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/Multi-hop-statlaw
unset PYTHONPATH
export HF_HOME=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/models/hf_cache
PY=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/gemma_venv/bin/python
MODEL=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/models/gemma-4-26B-A4B-it
MODE=${EVIDENCE_MODE:?Set EVIDENCE_MODE to gold or retrieved}
if [[ "$MODE" != gold && "$MODE" != retrieved ]]; then
  echo "Invalid EVIDENCE_MODE: $MODE" >&2
  exit 2
fi
retrieval_args=()
if [[ "$MODE" == retrieved ]]; then
  retrieval_args=(--retrieval results/step5_bm25_top10.jsonl)
fi
"$PY" scripts/extract_gemma_features.py \
  --questions-zip ../questions.json.zip \
  --model-path "$MODEL" \
  --evidence-mode "$MODE" \
  "${retrieval_args[@]}" \
  --batch-size 4 \
  --max-input-tokens 1536 \
  --max-passages 10 \
  --max-passage-chars 400 \
  --output "results/step5_${MODE}_features.jsonl"
"$PY" scripts/train_gemma_probe.py \
  --questions-zip ../questions.json.zip \
  --split results/step4_split.json \
  --features "results/step5_${MODE}_features.jsonl" \
  --output "results/step5_${MODE}_summary.json"
