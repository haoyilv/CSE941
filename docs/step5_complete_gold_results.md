# Step 5 update: complete annotated excerpts

## Method

The 6,853 released HousingQA questions were split internally by `question_group` into 5,476 training and 1,377 test examples (seed 20261103). All conditions use the same split, a frozen Gemma 4 26B A4B backbone, its 2,816-dimensional final-token vector, and a separately trained PyTorch linear Yes/No classifier. The backbone is not fine-tuned. This is a **course-internal split of the benchmark's released evaluation questions** and must not be compared directly with published scores.

| Input to frozen Gemma | Train accuracy | Test accuracy | Test correct | Test Yes F1 |
|---|---:|---:|---:|---:|
| Jurisdiction + question | 89.7% | 66.7% | 919/1,377 | 52.5% |
| Question + capped annotated excerpts (earlier pilot) | 89.5% | 65.1% | 897/1,377 | 55.1% |
| Question + top-10 state-filtered BM25 statutes | 91.8% | 69.0% | 950/1,377 | 58.7% |
| Question + complete annotated excerpts (no truncation) | 89.9% | **69.6%** | 959/1,377 | 60.2% |

The majority-No baseline is 60.3% on the internal test set. Relative to question-only, BM25 input fixes 246 errors but introduces 215 new ones (net +31, or +2.25 percentage points). Complete annotated excerpts fix 258 errors and introduce 218 (net +40, or +2.90 points). Relative to the earlier capped annotations, complete excerpts produce a net **62 additional correct answers (+4.50 points)**. Complete excerpts exceed BM25 by only nine correct answers (+0.65 points); the small difference is not evidence of a robust advantage. These are answer-classification results, not proof of faithful legal reasoning.

## Evidence and retrieval checks

BM25 top-10 contains at least one annotated statute ID for 546/1,377 test questions (39.7%) and every annotated ID for 227/1,377 (16.5%). This measures exact ID overlap with annotations, not whether a passage legally supports the answer. The BM25 classifier scores 66.7% on questions with any annotated ID hit and 70.5% on questions without one. This association does not show that the classifier uses retrieved provisions faithfully; the question subsets may differ in difficulty and answer distribution.

The earlier capped conditions used up to 10 passages, 400 characters per passage, and 1,536 input tokens. Capped annotated inputs average **295 tokens** (median 218), while BM25 inputs average **1,191 tokens** (median 1,192). The complete-annotation condition supplies **all released excerpts** for each question, up to 15 excerpts, without shortening them. Its 8,192-token safety limit exceeds the longest assembled input of **6,083 tokens**. An independent token-length audit and feature-file verification checked all **6,853** questions and found **zero truncated inputs**. Released annotated excerpts are reference evidence; they need not contain every relevant law or the entire statutory section. Evidence conditions still differ in amount and form of text.

## Interpretation and next experiment

The best internal-split probe accuracy is 69.6% with complete annotated excerpts. The separate zero-training 500-question Gemma-answering experiments use a different sample and protocol, so their percentages cannot be compared directly with this table. Next, ask the LLM to answer directly from complete annotated excerpts and retrieved statutes on the same held-out questions, with citations and an inspectable reasoning chain. Evaluate retrieval completeness and errors involving exceptions, definitions, and cross-references separately. This is needed before claiming an improvement in multi-hop statutory reasoning.

## Reproduction

`scripts/housing_data.py` makes the split. `scripts/build_index.py` and `scripts/evaluate_retrieval.py` create BM25 results. `scripts/extract_gemma_features.py` extracts frozen features; `scripts/train_gemma_probe.py` trains each linear head. `scripts/audit_gold_lengths.py` and `scripts/verify_gold_full.py` establish that the complete annotated excerpts were used without truncation. `scripts/compare_step5.py` creates `results/step5_gold_full_comparison.json`. The complete-gold HPCC GPU run was job **21251406**. Large feature caches, the statute index, and retrieved statute text remain on HPCC and can be regenerated from the public dataset.
