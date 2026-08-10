# Llama-3.1-8B: Backbone vs Direct SFT vs OPD+PI (per-fold CV)

**Downstream setup:** triple-concat embeddings → SVC (classification) / Ridge (regression)  
**Train CV:** 5 folds from `results/folds.json` (166 train subjects)  
**Rationale sources:**
- **Backbone:** `Llama-3.1-8B-Instruct` zero-shot (`rationale_suffix=llama`)
- **Direct SFT:** LoRA SFT on all 237 samples, no PI (`rationale_suffix=sft_llama31`)
- **OPD + PI:** R1-Distill-Llama-8B teacher + PI in teacher prompt only — *rationale eval pending* (fold 0–2 trained; fold 3–4 failed; no OOF rationales generated yet)

## Classification F1 (CV, per fold)

| Fold | Backbone | Direct SFT | OPD + PI |
|------|----------|------------|----------|
| 0 | 0.882 | 0.882 | — |
| 1 | 0.818 | 0.787 | — |
| 2 | 0.846 | 0.843 | — |
| 3 | 0.723 | 0.694 | — |
| 4 | 0.694 | 0.727 | — |
| **Mean** | **0.793** | **0.787** | **—** |

## Regression RMSE (CV, per fold)

| Fold | Backbone | Direct SFT | OPD + PI |
|------|----------|------------|----------|
| 0 | 5.664 | 5.584 | — |
| 1 | 5.825 | 5.570 | — |
| 2 | 6.173 | 5.871 | — |
| 3 | 5.897 | 5.694 | — |
| 4 | 5.873 | 5.719 | — |
| **Mean** | **5.887** | **5.688** | **—** |

## Regression Pearson r (CV, per fold)

| Fold | Backbone | Direct SFT | OPD + PI |
|------|----------|------------|----------|
| 0 | 0.598 | 0.645 | — |
| 1 | 0.667 | 0.723 | — |
| 2 | 0.482 | 0.567 | — |
| 3 | 0.520 | 0.586 | — |
| 4 | 0.679 | 0.723 | — |
| **Mean** | **0.589** | **0.649** | **—** |

## Test set (single split, all folds pooled)

| Method | Concat cls F1 | Triple cls F1 | Concat RMSE | Triple RMSE | Triple r |
|--------|---------------|---------------|-------------|-------------|----------|
| Backbone | 0.718 | 0.730 | 5.559 | 5.388 | 0.589 |
| Direct SFT | 0.732 | 0.729 | 5.679 | 5.585 | 0.531 |
| OPD + PI | — | — | — | — | — |

Raw per-fold numbers: `results/llama_method_comparison_per_fold.json`
