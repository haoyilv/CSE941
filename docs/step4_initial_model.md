# Step 4: HousingQA data reader and initial trained model

## Data and split

The data reader is [`scripts/housing_data.py`](../scripts/housing_data.py). It
reads `questions.json` from the official HousingQA ZIP, validates unique IDs
and binary answers, and creates examples with a question ID, jurisdiction,
question text, question group, label, and annotated statutes. The released
question collection has 6,853 Yes/No examples and no official training split.
For this course assignment, we make a fixed internal split with seed 20261103.
Every `question_group` is wholly in train or test, reducing leakage from the
same question template appearing across jurisdictions.

| Partition | Questions | Yes | No | Question groups |
| --- | ---: | ---: | ---: | ---: |
| Train | 5,476 | 1,868 | 3,608 | 181 |
| Test | 1,377 | 546 | 831 | 45 |

An example input is:

```text
Answer a question about United States housing law around 2021.
Jurisdiction: Alabama
Question: Is there a state/territory law regulating residential evictions?
Answer:
```

Its target is `Yes = 1`. The full split and IDs are saved in
[`results/step4_split.json`](../results/step4_split.json).

## X, Y, and model

**X** is jurisdiction plus natural-language question. Gemma 4 26B A4B's
tokenizer converts it into token IDs. A frozen Gemma backbone produces a
2,816-dimensional final-token hidden vector, which is the input feature to a
trainable linear binary classifier. The vector is cached in float16 to avoid
repeated GPU inference. Annotated statutes and answer labels are excluded from
the model input. This is deliberately a question-only initial learning model;
retrieval-augmented statutory reasoning is a subsequent component.

**Y** is binary: `No = 0`, `Yes = 1`. The classifier is trained on the training
partition with binary cross-entropy, PyTorch AdamW, learning rate 0.01, weight
decay 0.01, and 100 epochs. Feature standardization uses training-set means
and standard deviations only. Libraries: Python standard library, PyTorch,
Hugging Face Transformers, and Accelerate for model placement. The model is
evaluated separately on train and held-out test with accuracy, balanced
accuracy, Yes-class F1, and a confusion matrix. A majority-class baseline is
reported for both partitions.

## Results

The full GPU feature extraction and classifier run was job 20336573. It
produced all 6,853 feature rows and the measured metrics in
[`results/step4_probe_summary.json`](../results/step4_probe_summary.json).

| Partition | Accuracy | Balanced accuracy | Yes F1 | Majority-No accuracy |
| --- | ---: | ---: | ---: | ---: |
| Train (5,476) | 89.7% (4,913 correct) | 87.0% | 83.9% | 65.9% |
| Test (1,377) | 66.7% (919 correct) | 63.2% | 52.5% | 60.3% |

The classifier beats the majority baseline on the internal test split by
6.4 percentage points. Its 23.0-point train/test gap shows weak generalization
to unseen question groups. In particular, Yes recall on the test set is only
46.3% (253/546). This question-only linear probe is an initial learned
baseline, not a complete retrieval-and-reasoning system.

This internal split repurposes the benchmark's released test questions for a
course train/test experiment. Its accuracy must not be compared directly with
the paper's official held-out test results or our earlier 500-question
zero-training Gemma runs.

## Reproduction on HPCC

```bash
python3 scripts/housing_data.py --questions-zip ../questions.json.zip --output results/step4_split.json
sbatch jobs/gemma_step4_probe.sbatch
```

The batch script extracts frozen features with
[`scripts/extract_gemma_features.py`](../scripts/extract_gemma_features.py),
then trains and evaluates the classifier with
[`scripts/train_gemma_probe.py`](../scripts/train_gemma_probe.py).
