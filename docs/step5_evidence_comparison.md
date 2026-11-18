# Step 5: Effect of statutory evidence on the trained HousingQA probe

**Update:** The complete annotated-excerpt run supersedes the capped gold result below. See [complete-gold results](step5_complete_gold_results.md).

## Setup

The 6,853 released HousingQA questions were split internally by `question_group` into 5,476 training and 1,377 test examples (seed 20261103). This is **not** the benchmark's official train/test split, and these scores must not be compared with published benchmark scores. The same split, frozen Gemma 4 26B A4B backbone, 2,816-dimensional final-token feature, and trained linear Yes/No classifier were used in all three conditions. Each condition has its own classifier fitted on the corresponding training features.

| Input to frozen Gemma | Train accuracy | Test accuracy | Test correct | Test Yes F1 |
|---|---:|---:|---:|---:|
| Jurisdiction + question | 89.7% | 66.7% | 919/1,377 | 52.5% |
| Question + bounded annotated statutes | 89.5% | 65.1% | 897/1,377 | 55.1% |
| Question + top-10 state-filtered BM25 statutes | 91.8% | **69.0%** | 950/1,377 | 58.7% |

The majority-No test baseline is 60.3%. Compared question by question, the BM25-input classifier fixes 246 errors made by the question-only classifier and introduces 215 new errors: a net gain of **31 questions, or 2.25 percentage points**. The annotated-statute classifier fixes 220 question-only errors and introduces 242: a net loss of **22 questions, or 1.60 points**. These are modest, mixed changes; the experiment does not establish that retrieval improves legal reasoning.

## Retrieval and input diagnostics

On the internal test set, BM25 top-10 contains at least one annotated statute ID for 546/1,377 questions (39.7%), and every annotated statute ID for 227/1,377 (16.5%). These are exact ID overlap measures against the dataset annotations, not a judgment of whether a retrieved passage legally supports the answer. The BM25-input classifier scores 66.7% when it hits at least one annotated ID and 70.5% when it does not. This counterintuitive association means answer accuracy cannot be used as evidence that the classifier uses the retrieved provisions faithfully; question difficulty and label distribution may differ between those groups.

Evidence was capped at 10 passages, 400 characters per passage, and 1,536 input tokens. Annotated-statute inputs average **295 tokens** (median 218); BM25 inputs average **1,191 tokens** (median 1,192). No annotated-statute input reaches the token cap; 138/6,853 BM25 inputs do. The two evidence conditions therefore differ substantially in amount and form of text. The gold condition is a *bounded annotated-evidence diagnostic*, not a full oracle-evidence evaluation: the first 400 characters of a cited provision may omit its operative rule, exception, or cross-reference.

## Interpretation and next experiment

The trained probe is a binary classifier on frozen Gemma features; it does not generate or verify a legal explanation. Its 69.0% BM25 score is the best of these three **internal-split probe** conditions. It does not supersede the separate zero-training 500-question LLM-answering results, which use a different protocol and sample.

Next, equalize the evidence budget and test direct answer generation with full annotated provisions, then compare to retrieved provisions on the same held-out questions. Measure all-evidence retrieval and manually inspect cases where additional text changes the answer, including missed exceptions and cross-references. This is needed before attributing gains to retrieval or multi-hop reasoning.

## Reproduction

`scripts/housing_data.py` defines the split and label reader; `scripts/evaluate_retrieval.py` generates the top-10 retrieval file; `scripts/extract_gemma_features.py` extracts frozen features for each evidence mode; `scripts/train_gemma_probe.py` trains each linear head and emits predictions; `scripts/compare_step5.py` produces `results/step5_comparison.json`. The feature caches and complete retrieved statute text are local HPCC artifacts and are not needed to read the summary.
