# AD_diagnosis Baseline 实验报告

> 生成时间：2026-06-05  
> 环境：`conda activate ad`（Python 3.11）

---

## 1. 实验概览

当前已完成两个 transcript-only baseline，并在 166 条训练样本上进行 **5 折交叉验证**，覆盖两个任务：

| 任务 | 预测目标 | 说明 |
|------|----------|------|
| 分类 | `label`（1 = AD，0 = Control） | 判断是否为阿尔茨海默病患者 |
| 回归 | `mmse`（3–30） | 预测 MMSE 认知评分 |

| Baseline | 输入 | 特征 | 状态 |
|----------|------|------|------|
| Baseline 1 | transcript | TF-IDF sparse features | 已完成 |
| Baseline 2 | transcript | `BAAI/bge-m3` dense embeddings | 已完成 |
| Baseline 3 | transcript + teacher rationale | 待实现 |

---

## 2. 数据划分

### 2.1 数据集

| 集合 | 文件 | 样本数 | AD | Control | MMSE 范围 |
|------|------|--------|-----|---------|-----------|
| 训练集 | `ad_s2t_wav2vec.csv` + `control_s2t_wav2vec.csv` | 166 | 87 | 79 | 3–30 |
| 测试集 | `test_s2t_wav2vec.csv` | 71 | 35 | 36 | 5–30 |

测试集目前**未参与**任何训练或评估，仅留作最终测试。

### 2.2 交叉验证

- **折数**：5-fold，`random_state=42`
- **分类**：`StratifiedKFold`（按 `label` 分层）
- **回归**：`KFold`（随机打乱）
- **每折规模**：train ≈ 133，val ≈ 33
- **固定切分**：保存在 [`results/folds.json`](folds.json)，后续 baseline 共用

> 注意：没有单独划出固定的 validation set；验证集是 CV 内每折轮换的 ~33 条样本。

---

## 3. Baseline 1：TF-IDF + LR / SVC / Ridge

### 3.1 输入特征

**仅使用 wav2vec ASR 转写文本（`Speech` 列）**，不使用 MMSE、年龄、性别等元数据。

| 步骤 | 配置 |
|------|------|
| 文本预处理 | 转小写 + 合并多余空格 |
| TF-IDF | `max_features=10000`, `ngram_range=(1,2)`, `min_df=2` |
| 拟合范围 | 每个 fold 仅在 **train fold** 内 fit TF-IDF，避免数据泄漏 |

### 3.2 下游模型

| 任务 | 模型名 | sklearn 实现 |
|------|--------|-------------|
| 分类 | `lr` | `LogisticRegression(max_iter=1000)` |
| 分类 | `svc` | `SVC(kernel='linear')` |
| 分类 | `ridge` | `RidgeClassifier()` |
| 回归 | `lr` | `LinearRegression()` |
| 回归 | `ridge` | `Ridge()` |

> 分类的 LR = 逻辑回归（Logistic Regression）；回归的 LR = 线性回归（Linear Regression）。

### 3.3 评估指标

| 任务 | 指标 |
|------|------|
| 分类 | Accuracy, Precision, Recall, F1（macro） |
| 回归 | RMSE, MAE, Pearson r |

---

## 4. Baseline 2：BAAI/bge-m3 Embedding + LR / SVC / Ridge

### 4.1 输入特征

**仅使用 wav2vec ASR 转写文本（`Speech` 列）**。流程为：

| 步骤 | 配置 |
|------|------|
| 文本预处理 | 转小写 + 合并多余空格 |
| Embedding model | `BAAI/bge-m3` |
| Embedding normalize | `true` |
| Batch size | 8 |
| 缓存位置 | `results/embeddings/BAAI_bge-m3_train.npy` |

Embedding 对每条 transcript 独立生成，不使用 `label`、`mmse`、年龄、性别或教育年限。

### 4.2 下游模型

下游模型与 Baseline 1 保持一致，区别只在特征从 TF-IDF sparse vector 换成 BGE-M3 dense embedding。

| 任务 | 模型名 | sklearn 实现 |
|------|--------|-------------|
| 分类 | `lr` | `LogisticRegression(max_iter=1000)` |
| 分类 | `svc` | `SVC(kernel='linear')` |
| 分类 | `ridge` | `RidgeClassifier()` |
| 回归 | `lr` | `LinearRegression()` |
| 回归 | `ridge` | `Ridge()` |

---

## 5. 实验结果（5-Fold CV 汇总）

### 5.1 分类任务（AD vs Control）

#### Baseline 1：TF-IDF

| 模型 | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| **LR** | **0.765 ± 0.022** | 0.774 ± 0.031 | 0.764 ± 0.022 | **0.763 ± 0.021** |
| Ridge | 0.759 ± 0.061 | 0.760 ± 0.061 | 0.759 ± 0.060 | 0.759 ± 0.061 |
| SVC | 0.747 ± 0.048 | 0.749 ± 0.048 | 0.748 ± 0.046 | 0.747 ± 0.047 |

**最佳分类模型**：LR（F1 = 0.763）

#### Baseline 2：BAAI/bge-m3 Embedding

| 模型 | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| LR | 0.716 ± 0.088 | 0.731 ± 0.090 | 0.716 ± 0.090 | 0.710 ± 0.092 |
| **Ridge** | **0.747 ± 0.079** | **0.755 ± 0.077** | **0.746 ± 0.079** | **0.743 ± 0.080** |
| SVC | 0.704 ± 0.081 | 0.719 ± 0.080 | 0.704 ± 0.082 | 0.696 ± 0.085 |

**最佳 embedding 分类模型**：Ridge（F1 = 0.743）

### 5.2 回归任务（MMSE 预测）

#### Baseline 1：TF-IDF

| 模型 | RMSE ↓ | MAE ↓ | Pearson r ↑ |
|------|--------|-------|-------------|
| LR | 5.894 ± 0.391 | **4.687 ± 0.433** | 0.552 ± 0.067 |
| **Ridge** | **5.683 ± 0.252** | 4.738 ± 0.305 | **0.614 ± 0.058** |

**最佳回归模型**：Ridge（RMSE = 5.683，Pearson r = 0.614）

#### Baseline 2：BAAI/bge-m3 Embedding

| 模型 | RMSE ↓ | MAE ↓ | Pearson r ↑ |
|------|--------|-------|-------------|
| LR | 6.602 ± 0.302 | 5.364 ± 0.260 | 0.459 ± 0.080 |
| **Ridge** | **5.952 ± 0.195** | **5.035 ± 0.289** | **0.575 ± 0.052** |

**最佳 embedding 回归模型**：Ridge（RMSE = 5.952，Pearson r = 0.575）

### 5.3 与 TF-IDF baseline 对比

| 任务 | 当前最佳 TF-IDF | 当前最佳 Embedding | 结论 |
|------|-----------------|--------------------|------|
| Classification | LR, F1 = 0.763 | text-embedding-3-small + SVC, F1 = 0.762 | 基本打平 |
| Regression | Ridge, RMSE = 5.683, r = 0.614 | text-embedding-3-large + Ridge, RMSE = 5.758, r = 0.617 | TF-IDF RMSE 更好，large embedding 的 r 略高 |

### 5.4 额外 Embedding Model：all-mpnet-base-v2

为了测试另一个英文 sentence-transformers 模型，也跑了 `sentence-transformers/all-mpnet-base-v2`。

| 任务 | 最佳模型 | 结果 |
|------|----------|------|
| Classification | Ridge | F1 = 0.698 ± 0.063 |
| Regression | Ridge | RMSE = 6.306 ± 0.233, Pearson r = 0.450 ± 0.041 |

`all-mpnet-base-v2` 在当前数据上弱于 `BAAI/bge-m3`，也弱于 TF-IDF baseline。

### 5.5 OpenAI Embedding Models

也测试了 OpenAI 的 `text-embedding-3-small` 和 `text-embedding-3-large`，下游模型为分类 `LR/SVC/RF`，回归 `Ridge/SVR/RFR`。

| Embedding Model | Classification Best | Regression Best |
|-----------------|---------------------|-----------------|
| `text-embedding-3-small` | SVC, F1 = 0.762 ± 0.091 | Ridge, RMSE = 5.960 ± 0.198, r = 0.572 ± 0.080 |
| `text-embedding-3-large` | SVC, F1 = 0.746 ± 0.017 | Ridge, RMSE = 5.758 ± 0.262, r = 0.617 ± 0.051 |

`text-embedding-3-small` 在分类上几乎追平 TF-IDF + LR（F1 = 0.762 vs 0.763）。`text-embedding-3-large` 在 MMSE 回归上 Pearson r 略高于 TF-IDF（0.617 vs 0.614），但 RMSE 仍略差（5.758 vs 5.683）。

---

## 6. 逐 Fold 明细（Baseline 1：TF-IDF）

### 6.1 分类 — LR

| Fold | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| 0 | 0.765 | 0.779 | 0.771 | 0.764 |
| 1 | 0.788 | 0.789 | 0.787 | 0.787 |
| 2 | 0.788 | 0.818 | 0.783 | 0.781 |
| 3 | 0.727 | 0.728 | 0.726 | 0.726 |
| 4 | 0.758 | 0.756 | 0.756 | 0.756 |

### 6.2 分类 — SVC

| Fold | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| 0 | 0.706 | 0.708 | 0.708 | 0.706 |
| 1 | 0.788 | 0.789 | 0.787 | 0.787 |
| 2 | 0.818 | 0.823 | 0.816 | 0.817 |
| 3 | 0.697 | 0.700 | 0.699 | 0.697 |
| 4 | 0.727 | 0.726 | 0.728 | 0.726 |

### 6.3 分类 — Ridge

| Fold | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| 0 | 0.765 | 0.767 | 0.767 | 0.765 |
| 1 | 0.788 | 0.789 | 0.787 | 0.787 |
| 2 | 0.848 | 0.850 | 0.847 | 0.848 |
| 3 | 0.667 | 0.667 | 0.667 | 0.667 |
| 4 | 0.727 | 0.726 | 0.728 | 0.726 |

### 6.4 回归 — LR

| Fold | RMSE | MAE | Pearson r |
|------|------|-----|-----------|
| 0 | 5.427 | 4.122 | 0.611 |
| 1 | 5.704 | 4.508 | 0.635 |
| 2 | 6.506 | 5.278 | 0.446 |
| 3 | 5.659 | 4.425 | 0.551 |
| 4 | 6.177 | 5.100 | 0.519 |

### 6.5 回归 — Ridge

| Fold | RMSE | MAE | Pearson r |
|------|------|-----|-----------|
| 0 | 5.226 | 4.298 | 0.702 |
| 1 | 5.677 | 4.581 | 0.663 |
| 2 | 5.883 | 4.750 | 0.551 |
| 3 | 5.683 | 4.832 | 0.569 |
| 4 | 5.944 | 5.227 | 0.586 |

---

## 7. 简要分析

1. **分类**：当前最佳仍是 TF-IDF + LR，F1 = 0.763；`text-embedding-3-small + SVC` 几乎打平，F1 = 0.762。
2. **回归**：按 RMSE 看，当前最佳仍是 TF-IDF + Ridge，RMSE = 5.683；按 Pearson r 看，`text-embedding-3-large + Ridge` 略高，r = 0.617。
3. **Embedding baseline 整体没有明显超过 TF-IDF**：在这个小样本、ASR 噪声较强的数据上，dense semantic embedding 与 n-gram TF-IDF 更像是互补而不是替代。
4. **样本量限制**：每折仅 ~33 条验证样本，指标波动较大；后续更值得尝试 TF-IDF + embedding feature fusion。

---

## 8. 项目结构

```
/data/jiawen/AD_diagnosis/
├── baselines/                  # baseline 训练代码
│   ├── run_baselines.py        # 运行入口
│   ├── data.py                 # 数据加载
│   ├── cv.py                   # 5-fold 切分
│   ├── features.py             # TF-IDF pipeline + embedding cache
│   └── train.py                # 单 fold 训练逻辑
├── evaluation/
│   └── metrics.py              # 评估指标
├── data/                       # 原始 CSV 数据
├── results/                    # CV 结果与报告
├── config.yaml                 # 超参数与模型列表
└── requirements.txt
```

## 9. 复现方式

```bash
conda activate ad
cd /data/jiawen/AD_diagnosis

# Baseline 1
python baselines/run_baselines.py --baseline tfidf

# Baseline 2
python baselines/run_baselines.py --baseline embedding

# Baseline 2 with another embedding model
python baselines/run_baselines.py --baseline embedding \
  --embedding-model sentence-transformers/all-mpnet-base-v2

# OpenAI text-embedding-3-large
python baselines/run_baselines.py --baseline embedding \
  --embedding-provider openai \
  --embedding-model text-embedding-3-large
```

### 相关文件

| 文件 | 说明 |
|------|------|
| [`config.yaml`](../config.yaml) | 超参数与模型列表 |
| [`baselines/run_baselines.py`](../baselines/run_baselines.py) | 运行入口 |
| [`baselines/data.py`](../baselines/data.py) | 数据加载 |
| [`baselines/cv.py`](../baselines/cv.py) | 5-fold 切分 |
| [`baselines/features.py`](../baselines/features.py) | TF-IDF pipeline 与 BGE-M3 embedding cache |
| [`baselines/train.py`](../baselines/train.py) | 单 fold 训练逻辑 |
| [`evaluation/metrics.py`](../evaluation/metrics.py) | 评估指标 |
| [`results/cv_summary_tfidf.csv`](cv_summary_tfidf.csv) | Baseline 1 汇总结果 |
| [`results/cv_per_fold_tfidf.csv`](cv_per_fold_tfidf.csv) | Baseline 1 逐 fold 结果 |
| [`results/cv_summary_embedding.csv`](cv_summary_embedding.csv) | Baseline 2 汇总结果 |
| [`results/cv_per_fold_embedding.csv`](cv_per_fold_embedding.csv) | Baseline 2 逐 fold 结果 |
| [`results/cv_summary_embedding_sentence-transformers_all-mpnet-base-v2.csv`](cv_summary_embedding_sentence-transformers_all-mpnet-base-v2.csv) | all-mpnet-base-v2 汇总结果 |
| [`results/cv_per_fold_embedding_sentence-transformers_all-mpnet-base-v2.csv`](cv_per_fold_embedding_sentence-transformers_all-mpnet-base-v2.csv) | all-mpnet-base-v2 逐 fold 结果 |
| [`results/cv_summary_embedding_openai_text-embedding-3-small.csv`](cv_summary_embedding_openai_text-embedding-3-small.csv) | text-embedding-3-small 汇总结果 |
| [`results/cv_per_fold_embedding_openai_text-embedding-3-small.csv`](cv_per_fold_embedding_openai_text-embedding-3-small.csv) | text-embedding-3-small 逐 fold 结果 |
| [`results/cv_summary_embedding_openai_text-embedding-3-large.csv`](cv_summary_embedding_openai_text-embedding-3-large.csv) | text-embedding-3-large 汇总结果 |
| [`results/cv_per_fold_embedding_openai_text-embedding-3-large.csv`](cv_per_fold_embedding_openai_text-embedding-3-large.csv) | text-embedding-3-large 逐 fold 结果 |
| [`results/embeddings/`](embeddings/) | BGE-M3 embedding 缓存 |
| [`results/folds.json`](folds.json) | 固定 fold 切分 |

---

## 10. 待完成 Baseline

| Baseline | 特征 | 状态 |
|----------|------|------|
| Baseline 1 | TF-IDF + transcript | 已完成 |
| Baseline 2 | transcript embedding（BGE-M3 / mpnet / OpenAI small / OpenAI large） | 已完成 |
| Baseline 3a | TF-IDF + transcript + teacher rationale | 待实现 |
| Baseline 3b | Embedding + transcript + teacher rationale | 待实现 |
