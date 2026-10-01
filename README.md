# Difficulty-Aware Adaptive Test-Time Compute for Indian Legal Reasoning

Given an Indian legal question, name the provision that governs it — one of **1,932** Constitution
Articles and statute Sections. The question never contains the answer, so it has to be recalled.
A first forward pass scores its own difficulty, and that score decides how much compute the
question gets.

**Research prototype. Not legal advice.**

## Result

Spending more compute per question did not improve accuracy. *Routing* it by predicted difficulty
reached the accuracy of a fixed 16-pass budget at **7.2 passes — 55% less compute**.

| Compute budget | Passes | Test accuracy |
|---|---|---|
| 1 pass | 1.0 | 30.19% |
| 16 passes | 16.0 | 29.96% |
| **Adaptive (confidence router)** | **7.2** | **29.96%** |
| Ablation: random routing | 7.2 | 29.82% |

Budgets of 1, 4, 8 and 16 passes all fall inside the spread across sampling seeds, so the extra
sampling buys nothing measurable. The router still beats random routing at identical bucket shares.

| System | Exact match | Top-5 |
|---|---|---|
| Random guess | 0.05% | — |
| TF-IDF + logistic regression | 29.64% | 47.2% |
| **InLegalBERT, fine-tuned** (3 seeds) | **30.42% ± 0.52** | **47.8%** |

Two honest negatives: the transformer is statistically indistinguishable from the TF-IDF baseline
(McNemar p = 0.82), and accuracy is capped by the label space, not the compute budget —
Constitution Articles score **50.2%** while statute Sections score **12.4%**, because 136 of 378
test Sections never appear in training.

## Task design

The dataset ([`alpie/indian-law`](https://www.kaggle.com/datasets/alpie/indian-law), 47,789 free-text
Q/A records) has no labels, categories or splits. It was turned into a scoreable task:

- **Label** = the first provision cited in the reference answer, extracted by one rule in
  `src/data/provisions.py`. The same rule is applied to model output, so scoring is symmetric.
- **Eligibility**: questions that already name their own answer (including forms like "IPC 354")
  are dropped, so the answer cannot be copied out of the prompt.
- **Leakage**: near-duplicate questions (TF-IDF cosine ≥ 0.80, transitively grouped) never span
  splits. Highest test→train similarity after splitting: 0.7998.
- The dataset's `complex_cot` reasoning field is never shown to a model.

## Demo

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m streamlit run app/main.py
```

Trained weights are not in the repo (0.4–1.2 GB per checkpoint), so run the pipeline below first.
The app then serves the real checkpoint: live predictions reproduce the recorded test predictions
80/80, and a built-in panel scores a random sample of held-out questions on demand.

## Reproduce

```bash
.venv/Scripts/python -c "import kagglehub; print(kagglehub.dataset_download('alpie/indian-law'))"
.venv/Scripts/python -m src.data.audit_dataset      # results/audit/
.venv/Scripts/python -m src.data.preprocess         # data/processed/, data/splits/
.venv/Scripts/python -m src.data.eda                # plots/eda/
.venv/Scripts/python -m src.training.train_baseline # TF-IDF baselines
.venv/Scripts/python -m src.training.hyperparameter_search
.venv/Scripts/python -m src.evaluation.tt_compute_mc # the compute experiment
.venv/Scripts/python -m pytest tests -q
```

## Layout

```
app/          Streamlit demo (engine.py = inference, theme.py = styling)
src/data/     audit, provision extraction, Task A labelling, leakage-safe split
src/models/   TF-IDF baselines, transformer classifier, reasoning model
src/training/ training loops and the staged hyperparameter search
src/evaluation/ metrics and the adaptive test-time-compute experiment
results/      every metric reported above, as CSV / JSON
```

All numbers above are read from `results/` by the app at runtime; nothing is hard-coded.
