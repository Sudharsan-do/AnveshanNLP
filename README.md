# AnveshanNLP — Multilingual Hindi & Hinglish 56-Class Intent Classification

End-to-end 56-class conversational intent classification system for **Hindi (Devanagari)**, **Hinglish (Romanized Hindi)**, and **code-switched Mixed** queries under noisy ASR conditions and a strict **≤ 300M active parameter** constraint.

---

## 1. Pipeline Overview & Architecture

```
                         ┌──────────────────────────────────────────────────────────┐
                         │                Raw Query (Hindi / Hinglish / Mixed)      │
                         └──────────────┬───────────────────────────┬───────────────┘
                                        │                           │
                         NFKC + Whitespace Clean     ASR Normalization + xlit_word_dict
                                        │                           │
                                        ▼                           ▼
                         ┌──────────────────────────┐ ┌─────────────────────────────┐
                         │  l3cube-pune/hing-roberta│ │  TF-IDF char_wb (1..5)      │
                         │  -mixed (models/best)    │ │  30,802 features            │
                         │  278,086,712 params      │ │  + 3-Fold Calibrated        │
                         └──────────────┬───────────┘ │  LinearSVC (1,724,968 params│
                                        │             └─────────────┬───────────────┘
                                        │ P_XLM                     │ P_SVC
                                        └──────────┬────────────────┘
                                                   ▼
                         ┌──────────────────────────────────────────────────────────┐
                         │ Script-Adaptive Geometric Fusion (Log-Space PoE, ε=0.01) │
                         │   Latin: 0.52·XLM + 0.48·SVC                             │
                         │   Dev  : 0.32·XLM + 0.68·SVC                             │
                         │   Mixed: 0.45·XLM + 0.55·SVC                             │
                         └─────────────────────────┬────────────────────────────────┘
                                                   ▼
                         ┌──────────────────────────────────────────────────────────┐
                         │ Tail Calibration & Class 55 (Out-of-Scope) Fallback      │
                         │   Multipliers: 55×1.30, 53×1.15, 12×1.15                 │
                         │   OOS_CONF rescue + low-confidence OOS guard             │
                         └─────────────────────────┬────────────────────────────────┘
                                                   ▼
                                       Interactive NLU Workbench (demo_ui)
```

### Key Components
- **[aavesham_v2.ipynb](aavesham_v2.ipynb)**: Fine-tunes `l3cube-pune/hing-roberta-mixed` (`XLMRobertaForSequenceClassification`) on 16,936 training queries plus phonetic-noise augmented Hinglish rows, saving the `models/best` checkpoint.
- **[nlu_pipeline.ipynb](nlu_pipeline.ipynb)**: Audits the ≤ 300M parameter budget, fits and validates the 3-fold Platt-calibrated `LinearSVC` via 5-fold Stratified CV on `data/train.csv`, and runs script-adaptive geometric fusion with `models/best`.
- **[demo_ui/](demo_ui/README.md)**: Local interactive web workbench (`server.py` + `inference_engine.py`) with live multi-script robustness testing, Class 55 fallback controls, latency telemetry, and parameter compliance auditing.

---

## 2. Cross-Script (Hindi ↔ Hinglish) & Class 55 (Out-of-Scope) Handling

### Script Distribution & Cross-Script Normalization
The 16,936-row training dataset ([data/train.csv](data/train.csv)) consists of **61.13% Devanagari**, **33.33% Latin (Hinglish)**, and **5.54% Mixed** queries. To prevent overfitting to Devanagari and generalize cleanly to Latin Hinglish and ASR noise:

1. **Dual-Path Normalization**:
   - **Transformer (`hing-roberta-mixed`)**: Receives light NFKC + whitespace-collapsed text (`clean`) so native Hinglish spelling and subwords remain intact.
   - **Lexical (`LinearSVC`)**: Applies `clean_asr` (collapses elongated characters like `soooong` → `song` and normalizes ASR shorthand such as `kia`→`kya`, `mjhe`→`mujhe`, `h`→`hai`, `plz`→`please`, `kr`→`karo`, `thnx`→`dhanyavad`) followed by word-level Devanagari-to-Latin transliteration via the 6,782-entry domain lexicon ([data/xlit_word_dict.json](data/xlit_word_dict.json)), projecting all scripts into a shared Romanized character $n$-gram space.
2. **Phonetic Noise Augmentation (`aavesham_v2.ipynb`)**: Generates noisy Hinglish variants from Devanagari `transliterated_text` using 23 phonetic substitution rules (`aa`↔`a`, `ee`↔`i`, `oo`↔`u`, `kh`↔`k`, `ph`↔`f`, `w`↔`v`, etc., at `NOISE_P = 0.2`).
3. **Script-Stratified Sampling & Weighting**:
   - Transformer training uses a `WeightedRandomSampler` targeting `32%` original Latin, `33%` augmented Latin, `30%` Devanagari, and `5%` Mixed per epoch.
   - `LinearSVC` applies a **2.0× sample weight** to Latin training queries.
4. **Script-Adaptive Geometric Gating**: Combines probabilities in log-space ($\log P_{\text{ens}} = g(s)\log(P_{\text{XLM}} + 0.01) + (1 - g(s))\log(P_{\text{SVC}} + 0.01)$) with `g('latin') = 0.52`, `g('devanagari') = 0.32`, and `g('mixed') = 0.45`.

### Class 55 (Out-of-Scope / Catch-All) Handling
- In [data/verified_class_meanings.json](data/verified_class_meanings.json), **Class 55** (`1.47%` of train, `249` samples) has **30.72% purity**, acting as a composite catch-all for `play_podcasts`, `play_game`, `alarm_query`, and Out-of-Scope (OOS) queries.
- **Training**: Class imbalance (`84` to `1,224` samples across 56 classes) uses power-scaled weights `(max_count / count)^0.5`, while pinning `w[55] = 1.0` to avoid over-predicting the noisy bucket.
- **Inference & Fallback**:
  - `aavesham_v2.ipynb` defines `OOS_CONF = 0.50`: when raw `argmax == 55` and confidence `< 0.50`, it rescues the query by reassigning to the 2nd-best intent.
  - `nlu_pipeline.ipynb` and `demo_ui` apply post-fusion tail calibration multipliers (`55 × 1.30`, `53 × 1.15`, `12 × 1.15`) and a configurable low-confidence OOS guard (`0.22` default) that routes low-confidence out-of-domain utterances to Class 55.

---

## 3. Model Parameter Count Proof (≤ 300M Parameters)

Verified programmatically from the saved `models/best` checkpoint and the fitted `TfidfVectorizer` + `CalibratedClassifierCV(LinearSVC)` weights:

| Component | Exact Calculation | Parameters | Share of 300M Cap |
| :--- | :--- | :---: | :---: |
| `l3cube-pune/hing-roberta-mixed` (`models/best`) | `sum(p.numel() for p in model.parameters())` | `278,086,712` (`278.09M`) | 92.70% |
| `Calibrated LinearSVC` (`char_wb` 1–5 n-grams) | `30,802 features × 56 classes + 56 intercepts` | `1,724,968` (`1.72M`) | 0.57% |
| **Total Active Parameters** | **Transformer + LinearSVC** | **`279,811,680` (`279.81M`)** | **93.27%** |
| **Compliance Headroom** | `300,000,000 - 279,811,680` | **`+20,188,320` (`20.19M`)** | **6.73% free** |

> **Status**: **COMPLIANT** (`279,811,680 ≤ 300,000,000`).

---

## 4. Local 5-Fold Stratified CV & Validation Summary

Measured strictly on `train.csv` (`benchmark_models_comparison.csv` & `cv_metrics_comparison.csv`):

| Model / Feature Pipeline | Mean 5-Fold Macro-F1 | OOF Macro-F1 | Mean Latency | Parameters |
| :--- | :---: | :---: | :---: | :---: |
| **LinearSVC (Full ~57k char n-grams)** | **0.8217** | **0.8222** | 0.084 ms | 2.88M |
| LinearSVC + Chi2 Selection (`k=30,000`) | 0.8203 | 0.8208 | — | — |
| LinearSVC + ANOVA Selection (`k=25,000`) | 0.8185 | 0.8192 | — | — |
| MLP | 0.8097 | 0.8103 | 0.123 ms | 6.58M |
| RBF-SVM | 0.7674 | 0.7690 | 2.722 ms | Dual (14k SVs) |
| NB-SVM | 0.7597 | 0.7605 | 2.567 ms | 2.88M |
| LinearSVC + TruncatedSVD (`n=500`) | 0.7535 | 0.7537 | — | — |
| LinearSVC + TruncatedSVD (`n=300`) | 0.7125 | 0.7128 | — | — |
| CatBoost + TruncatedSVD (`n=300`) | 0.6258 | 0.6271 | — | — |

---

## 5. Step-by-Step Instructions to Train & Run the Demo UI

### Step 1: Train the Transformer Backbone (`models/best`)
1. Open and execute [aavesham_v2.ipynb](aavesham_v2.ipynb) (uses root `train.csv` with `transliterated_text`).
2. Save/copy the resulting checkpoint directory (`indicbert_v2_out/best`) into `models/best/` at the repository root.

### Step 2: Fit & Validate the LinearSVC + Ensemble Pipeline
Open and run [nlu_pipeline.ipynb](nlu_pipeline.ipynb) to verify the ≤ 300M parameter budget, run 5-fold CV for `Calibrated LinearSVC` on `data/train.csv`, and validate the script-adaptive geometric ensemble.

### Step 3: Launch the Interactive Demo UI
```bash
./venv/bin/python demo_ui/server.py --port 8000
```
Open **http://127.0.0.1:8000** in your browser.
