# AnveshanNLP Interactive NLU Demo UI

Interactive web console for the **Aavesham v2** multilingual intent classification system (`l3cube-pune/hing-roberta-mixed` + Script-Adaptive Calibrated `LinearSVC` Geometric Ensemble).

## Key Features
- **Live Multi-Script Intent Inference**: Test Devanagari Hindi, Hinglish (Romanized Hindi), noisy ASR spellings, and code-switched Mixed queries in real time.
- **Multi-Script Robustness Lab**: Run side-by-side semantic equivalents across Devanagari, Clean Hinglish, Noisy ASR Hinglish, and Mixed script to measure cross-script agreement.
- **Class 55 (Out-of-Scope) Fallback Controls**: Inspect Class 55 (`play_podcasts` / composite catch-all) probabilities, adjust the `aavesham_v2` `OOS_CONF` reassignment threshold and low-confidence OOS guard in real time, and toggle tail multipliers (`55 x 1.30`, `53 x 1.15`, `12 x 1.15`).
- **Live Latency Telemetry**: Real-time breakdown across preprocessing, XLM-RoBERTa forward pass, LinearSVC scoring, and geometric fusion, plus rolling session P50/P95/Mean latency.
- **<= 300M Parameter Compliance Audit**: Verified active parameter ledger (`278.09M` XLM-RoBERTa + `LinearSVC` <= `300.00M` limit).

## How to Run

From the repository root (`/Users/loshikaravichandran/Documents/AnveshanNLP`):

```bash
./venv/bin/python demo_ui/server.py --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.

*(First startup takes ~5 seconds to fit and cache the Calibrated `LinearSVC` in `demo_ui/cache/svc_bundle.pkl`; subsequent startups load from cache in ~1–2 seconds.)*
