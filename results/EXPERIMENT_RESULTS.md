# AD Diagnosis — Experiment Results Summary

> Last updated: 2026-06-12  
> Data: train 166 (`adrso*`) + test 71 (`adrsdt*`)  
> Evaluation: fixed 5-fold StratifiedKFold CV on train (`results/folds.json`); test = train→test

---

## Experimental Setup

| Item | Setting |
|------|---------|
| Classification label | AD (1) vs Control (0) |
| Regression target | MMSE |
| Default classifier | `SVC(kernel="linear")` |
| Default regressor | `Ridge` |
| Embedding model | `text-embedding-3-small` (OpenAI) |
| Language rationale | LLM-generated linguistic analysis (`data/rationale/*_rationale.json`) |
| Demographics source | `data/training-groundtruth.csv` (`age`, `gender`, `educ`) |
| Demo rationale (text) | Template fill-in (`build_template_demo_rationale()`) |
| Tabular demographics | 3 numeric features: `age_norm`, `gender_male`, `educ_norm` |
| Unified CV script | `baselines/eval_classification_cv.py` → `results/cv_classification_full.json` |

**Note on test demographics:** Test groundtruth is not yet joined (0/71 observed); prior test runs with demographic features used train-derived imputation. Results marked below should be re-evaluated once real test demographics are available.

---

## Classification

### SOTA

| Split | Best Method | F1 | Acc | Notes |
|-------|-------------|-----|-----|-------|
| **Train CV** | **triple_concat** | **0.823** | 0.825 | `emb(transcript) ‖ emb(rationale) ‖ emb(template_demo_rationale)` + SVC |
| **Test** | **v3_routing** | **0.803** | 0.803 | Disagreement routing: emb ≠ tfidf → LR meta-learner arbitrates |

**Train CV SOTA detail (triple_concat):**

| Metric | Value |
|--------|-------|
| Precision | 0.835 |
| Recall | 0.826 |
| Classifier | SVC(linear), single model |
| Uses meta-learner | No |

**Test SOTA detail (v3_routing):**

| Metric | Value |
|--------|-------|
| Precision | 0.804 |
| Recall | 0.803 |
| Rule | 52/71 agree → adopt shared pred; 19/71 disagree → meta-learner |
| Disagree accuracy | 0.684 |
| Agree accuracy | 0.846 |

> CV best and test best are **different systems**. Triple concat wins on CV where all 166 samples have real demographics; v3 wins on current test where demographic-dependent models underperform.

---

### Ablation (Train CV)

Feature progression on classification (5-fold CV mean F1):

| Model | Demographics | F1 | Δ vs prev |
|-------|--------------|-----|-----------|
| tfidf (transcript) | — | 0.747 | — |
| embedding (transcript) | — | 0.762 | +0.015 |
| concat (transcript + rationale) | — | 0.781 | +0.019 |
| concat + tabular | tabular | 0.812 | +0.031 |
| **triple_concat** | template demo_rationale | **0.823** | +0.011 |
| triple_concat + tabular | template + tabular | 0.811 | −0.012 |

**Key ablation takeaways:**

1. **Language rationale helps:** concat (0.781) > embedding (0.762) > tfidf (0.747).
2. **Tabular demographics help concat:** +3.1 F1 points (0.781 → 0.812).
3. **Template demo_rationale helps further:** triple_concat (0.823) > concat+tabular (0.812).
4. **Adding tabular on top of triple hurts slightly:** 0.811 < 0.823 (redundant encoding).
5. **LLM demo_rationale < template:** structured LLM triple CV F1 = 0.817 vs template 0.823.

Demographics encoding comparison (CV):

| Encoding | Feature | F1 |
|----------|---------|-----|
| None | concat only | 0.781 |
| Tabular (3-dim) | concat + `[age, gender, educ]` | 0.812 |
| Text (template) | triple_concat | **0.823** |
| Text + tabular | triple_concat + tabular | 0.811 |

---

### Other Baselines

#### Train CV — single models & routing

| Model | F1 | Acc | Eval method |
|-------|-----|-----|-------------|
| triple_concat | **0.823** | 0.825 | 5-fold CV |
| concat + tabular | 0.812 | 0.813 | 5-fold CV |
| triple_concat + tabular | 0.811 | 0.813 | 5-fold CV |
| v3_routing | 0.794 | 0.795 | OOF probabilities |
| v3_routing + tabular meta | 0.794 | 0.795 | OOF probabilities |
| concat | 0.781 | 0.783 | 5-fold CV |
| embedding + tabular | 0.763 | 0.765 | 5-fold CV |
| embedding | 0.762 | 0.764 | 5-fold CV |
| tfidf | 0.747 | 0.747 | 5-fold CV |

Source: `results/cv_classification_full.json`

#### Test — single models & routing

| Model | F1 | Acc | Notes |
|-------|-----|-----|-------|
| **v3_routing** | **0.803** | 0.803 | Meta-learner on emb/tfidf/concat probs |
| tfidf (transcript) | 0.789 | 0.789 | Strong single-model baseline |
| concat (transcript + rationale) | 0.774 | 0.775 | Dual concat |
| embedding (transcript) | 0.718 | 0.718 | Transcript only |
| triple_concat (LLM demo, imputed) | 0.745 | 0.746 | ⚠️ imputed demographics |
| TF-IDF (rationale only) | 0.629 | 0.634 | Diagnostic; rationale has signal but weaker alone |

Source: `results/test_disagreement_routing_v3.json`, `results/test_summary_all_features.json`, `results/triple_concat_llm_demo_results.json`, `results/test_rationale_tfidf_diagnostic.json`

#### v3 routing — test breakdown

| Subset | N | Accuracy |
|--------|---|----------|
| Agree (emb == tfidf) | 52 | 0.846 |
| Disagree (emb ≠ tfidf) | 19 | 0.684 |

On disagree cases (n=19): meta correct 13, tfidf correct 12, concat correct 12, embedding correct 7.

---

## Regression

### SOTA

| Split | Best Method | RMSE | MAE | Pearson r | Notes |
|-------|-------------|------|-----|-----------|-------|
| **Train CV** | **triple_concat + tabular** | **5.456** | 4.504 | **0.642** | Triple text concat + 3-dim demographics + Ridge |
| **Test** | **TF-IDF (transcript)** | **5.169** | 4.169 | **0.649** | v3 routing also uses TF-IDF for regression |

**Train CV SOTA detail:**

| Component | Value |
|-----------|-------|
| Features | `emb(transcript) ‖ emb(rationale) ‖ emb(demo_rationale) ‖ tabular` |
| Regressor | Ridge |

**Test SOTA detail:**

| Component | Value |
|-----------|-------|
| Features | TF-IDF(transcript) |
| Regressor | Ridge |
| Same as v3 routed regression | Yes (`route_regression=false`) |

> Regression favors **tabular demographics on CV** and **TF-IDF on test**. Text demo_rationale alone does not beat tabular for MMSE prediction.

---

### Ablation (Train CV)

| Model | Demographics | RMSE | r | Δ RMSE |
|-------|--------------|------|---|--------|
| embedding (transcript) | — | 5.960 | 0.572 | — |
| concat (transcript + rationale) | — | 5.848 | 0.604 | −0.112 |
| tfidf (transcript) | — | 5.683 | 0.614 | −0.165 |
| v3 routing (concat branch) | — | 5.848 | 0.574 | — |
| triple_concat | template demo_rationale | 5.704 | 0.639 | — |
| concat + tabular | tabular | 5.478 | 0.638 | −0.370 (vs concat) |
| v3 routing | tabular (concat+tabular) | 5.484 | 0.622 | — |
| embedding + tabular | tabular | 5.613 | 0.615 | — |
| **triple_concat + tabular** | template + tabular | **5.456** | **0.642** | **best** |

**Key ablation takeaways:**

1. **Tabular demographics are the main regression gain** (concat: 5.848 → 5.478).
2. **Triple text demo_rationale improves r** (0.604 → 0.639) but RMSE still above tabular-only concat.
3. **Best CV regression = triple_concat + tabular** (RMSE 5.456, r 0.642).
4. **TF-IDF is competitive on CV** (RMSE 5.683) and wins on test (5.169).

---

### Other Baselines

#### Train CV

| Model | RMSE | MAE | r |
|-------|------|-----|---|
| triple_concat + tabular | **5.456** | 4.504 | **0.642** |
| concat + tabular | 5.478 | 4.541 | 0.638 |
| triple_concat | 5.704 | 4.838 | 0.639 |
| tfidf | 5.683 | 4.738 | 0.614 |
| embedding + tabular | 5.613 | 4.675 | 0.615 |
| concat | 5.848 | 4.978 | 0.604 |
| embedding | 5.960 | 5.095 | 0.572 |

Source: `results/cv_demo_rationale_triple_concat.json`, `results/cv_demographics_full_ablation.json`, `results/cv_demographics_ablation.json`

#### Test

| Model | RMSE | MAE | r | Notes |
|-------|------|-----|---|-------|
| **tfidf (transcript)** | **5.169** | 4.169 | **0.649** | Current test SOTA |
| v3_routing (routed) | 5.169 | 4.169 | 0.649 | Same as TF-IDF |
| triple_concat LLM demo (imputed) | 5.693 | 4.819 | 0.563 | ⚠️ imputed demographics |
| concat (transcript + rationale) | 5.612 | 4.736 | 0.570 | Dual concat |
| embedding (transcript) | 5.655 | 4.701 | 0.567 | Transcript only |
| concat + tabular (imputed) | 5.785 | 4.952 | 0.551 | ⚠️ imputed demographics |
| TF-IDF (rationale only) | 6.226 | 5.336 | 0.355 | Diagnostic |

Source: `results/test_disagreement_routing_v3.json`, `results/test_summary_all_features.json`, `results/triple_concat_llm_demo_results.json`, `results/test_rationale_tfidf_diagnostic.json`

---

## Recommended Pipelines (Current)

| Task | Train CV best | Test best (current) | Recommended going forward |
|------|---------------|---------------------|---------------------------|
| **Classification** | triple_concat (F1 0.823) | v3_routing (F1 0.803) | triple_concat once test demographics available; v3 as fallback |
| **Regression** | triple_concat + tabular (RMSE 5.456) | TF-IDF (RMSE 5.169) | TF-IDF on test now; concat+tabular or triple+tabular once test demographics available |

---

## Result File Index

| File | Content |
|------|---------|
| `results/cv_classification_full.json` | Unified train CV classification (all models) |
| `results/cv_demo_rationale_triple_concat.json` | Train CV concat / triple / tabular ablation (cls + reg) |
| `results/cv_demographics_full_ablation.json` | Train CV embedding/concat/tfidf/v3 ± demographics |
| `results/cv_demographics_ablation.json` | Train CV v3 routing ± demographics (cls + reg) |
| `results/test_disagreement_routing_v3.json` | Test v3 routing + baselines |
| `results/test_summary_all_features.json` | Test embedding / concat / tfidf |
| `results/triple_concat_llm_demo_results.json` | Test + CV triple concat with LLM demo_rationale |
| `results/test_rationale_tfidf_diagnostic.json` | Test rationale-only TF-IDF diagnostic |

---

## Pending Re-evaluation

Once test demographics (`age`, `gender`, `educ`) are available for all 71 samples:

- [ ] Re-run **triple_concat** on test (expected to recover from F1 0.745)
- [ ] Re-run **concat + tabular** and **triple_concat + tabular** on test
- [ ] Compare triple_concat vs v3_routing on test with real demographics
- [ ] Update this document with revised test SOTA
