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
