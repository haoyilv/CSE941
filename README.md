# CSE941 Step 4: Housing Statute QA

This repository contains our HousingQA data reader and an initial **trained**
Yes/No model. It uses a frozen Gemma 4 26B A4B backbone to represent each
question, then trains a PyTorch linear classifier on those vectors. The public
dataset is [reglab/housing_qa](https://huggingface.co/datasets/reglab/housing_qa).

## Data reader and representations

`scripts/housing_data.py` reads `questions.json` from `questions.json.zip`,
validates question IDs and Yes/No labels, and emits a reproducible internal
train/test split (`results/step4_split.json`). The split keeps each
`question_group` wholly on one side to reduce template leakage.

- **X:** Jurisdiction plus natural-language question. The Gemma tokenizer
  produces token IDs; the frozen model's final-token hidden state supplies a
  2,816-dimensional feature vector. Annotated statutes and answer labels are
  not included in X.
- **Y:** Binary label, `No = 0` or `Yes = 1`.
- **Learner:** A trainable linear classifier on standardized frozen Gemma
  vectors. Standardization parameters are computed from training data only.
- **Libraries:** PyTorch, Hugging Face Transformers, Accelerate, Python
  standard library.

The released HousingQA questions are designated a test collection by the
dataset authors, and there is no official training split. Our 5,476/1,377
split is **for this course assignment**; its accuracy is not comparable to the
paper's official benchmark result.

## Measured results

| Split | Questions | Accuracy | Balanced accuracy | Yes F1 | Majority-No baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 5,476 | 89.7% | 87.0% | 83.9% | 65.9% |
| Test | 1,377 | 66.7% | 63.2% | 52.5% | 60.3% |

The train/test gap indicates limited generalization to unseen question groups.
Full counts and confusion matrices are in
[`results/step4_probe_summary.json`](results/step4_probe_summary.json). The
[methods report](docs/step4_initial_model.md) gives more detail. This is a
question-only first model; retrieval and multi-hop reasoning are later stages.

## Reproduction

Download the public `questions.json.zip` and Gemma model weights separately.
On an HPCC node with a compatible A100 GPU and a Python environment containing
PyTorch, Transformers, and Accelerate, first create the split:

```bash
python3 scripts/housing_data.py --questions-zip /path/to/questions.json.zip --output results/step4_split.json
```

Then extract question vectors and train the classifier:

```bash
python3 scripts/extract_gemma_features.py --questions-zip /path/to/questions.json.zip --model-path /path/to/gemma-4-26B-A4B-it --output results/step4_gemma_features.jsonl
python3 scripts/train_gemma_probe.py --questions-zip /path/to/questions.json.zip --split results/step4_split.json --features results/step4_gemma_features.jsonl --output results/step4_probe_summary.json
```

`jobs/gemma_step4_probe.sbatch` is the HPCC job used for our run; adjust its
account, virtual environment, model, and data paths on another cluster. Model
weights, original data, and the 50 MB frozen-feature cache are intentionally
excluded from Git.

## Step 5: statutory evidence comparison

Using the same internal split and frozen Gemma features, we trained separate
linear heads on question-only, bounded annotated-statute, and top-10 BM25
statute inputs. Test accuracy was **66.7%**, **65.1%**, and **69.0%**,
respectively. BM25 input gave a net gain of 31 correct answers over the
question-only input. BM25 retrieved every annotated statute ID for 16.5% of
test questions. These are classifier results, not direct LLM legal answers;
the evidence conditions also contain very different amounts of text.

The [Step 5 report](docs/step5_evidence_comparison.md) gives the paired counts,
retrieval diagnostics, and limitations. Summary files and per-question
predictions are in `results/`. `scripts/build_index.py`,
`scripts/evaluate_retrieval.py`, and `scripts/compare_step5.py` provide the
retrieval and comparison code. The full statute index, retrieved text, and
feature caches remain on HPCC because they are large and can be regenerated
from the public dataset.

### Complete annotated-excerpt update

We reran the annotated-evidence condition with **all released excerpts**, with
no 400-character cap or input truncation. An independent audit verified all
6,853 inputs and found zero truncated examples. Test accuracy rose from the
earlier capped result of 65.1% to **69.6%** (959/1,377; Yes F1 60.2%). This is
40 more correct answers than question-only and nine more than the BM25-input
classifier. See the [complete-evidence report](docs/step5_complete_gold_results.md)
and [`results/step5_gold_full_comparison.json`](results/step5_gold_full_comparison.json).

## Direct QA and iterative retrieval

`scripts/run_reasoning_eval.py` evaluates Gemma's generated Yes/No answers on
the same 1,377 internal test questions with no evidence, the top ten local BM25
statutes, or complete annotated excerpts. A structured prompt additionally
asks for the controlling rule, its application, and evidence gaps. The first
three direct-answer conditions are complete: **61.9%** without evidence,
**67.8%** with one-shot BM25 evidence, and **79.7%** with complete annotated
excerpts. The structured-answer conditions are still running. These numbers
are distinct from the trained linear-probe results above.

`scripts/generate_followup_queries.py` reads the first five BM25 statutes and
asks Gemma for a targeted second search. `scripts/retrieve_followup.py`
combines five first-hop and five new second-hop statutes. On the test questions,
the number with at least one annotated ID in the ten passages increased from
546 to 660; the number with all annotated IDs increased from 227 to 286.
The effect on final answer accuracy is still being evaluated. The scripts and
Slurm job files in `jobs/` reproduce these experiments on HPCC.
