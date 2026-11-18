#!/bin/bash -login
#SBATCH -A wang-compbio
#SBATCH -p general-long-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=a100:1
#SBATCH --array=0-4%2
#SBATCH --job-name=housing_reasoning
#SBATCH --output=results/logs/reasoning-%A_%a.out
set -euo pipefail
cd /mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/Multi-hop-statlaw
unset PYTHONPATH
export HF_HOME=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/models/hf_cache
PY=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/gemma_venv/bin/python
MODEL=/mnt/ufs18/rs-027/compbio/wanglab/haoyi/CSE941/models/gemma-4-26B-A4B-it
case "$SLURM_ARRAY_TASK_ID" in
  0) EVIDENCE=none; STYLE=direct; NEW_TOKENS=64 ;;
  1) EVIDENCE=bm25; STYLE=direct; NEW_TOKENS=64 ;;
  2) EVIDENCE=gold; STYLE=direct; NEW_TOKENS=64 ;;
  3) EVIDENCE=bm25; STYLE=structured; NEW_TOKENS=384 ;;
  4) EVIDENCE=gold; STYLE=structured; NEW_TOKENS=384 ;;
  *) echo "Unexpected array task ID" >&2; exit 2 ;;
esac
RETRIEVAL_ARGS=()
if [[ "$EVIDENCE" == bm25 ]]; then
  RETRIEVAL_ARGS=(--retrieval results/step5_bm25_top10.jsonl)
fi
OUT="results/directqa_${EVIDENCE}_${STYLE}.jsonl"
"$PY" scripts/run_reasoning_eval.py \
  --questions-zip ../questions.json.zip \
  --split results/step4_split.json \
  --model-path "$MODEL" \
  --evidence-mode "$EVIDENCE" \
  --prompt-style "$STYLE" \
  --top-k 10 \
  --bm25-chars 400 \
  --max-input-tokens 8192 \
  --max-new-tokens "$NEW_TOKENS" \
  "${RETRIEVAL_ARGS[@]}" \
  --output "$OUT"
python3 scripts/score_predictions.py "$OUT" \
  --split results/step4_split.json \
  --output "results/directqa_${EVIDENCE}_${STYLE}_summary.json"
