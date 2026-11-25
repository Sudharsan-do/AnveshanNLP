"""
Live NLU Inference Engine for AnveshanNLP (aavesham_v2 + LinearSVC Ensemble).

Loads:
- Fine-tuned `l3cube-pune/hing-roberta-mixed` checkpoint from `models/best`
  (278,086,712 parameters, XLM-RoBERTa architecture, trained in `aavesham_v2.ipynb`)
- Calibrated LinearSVC on character n-grams (1..5) with ASR normalization &
  `data/xlit_word_dict.json` transliteration (cached after initial fit)
- Verified 56-class taxonomy from `data/verified_class_meanings.json`
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models" / "best"
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
SVC_CACHE_PATH = CACHE_DIR / "svc_bundle.pkl"

PARAM_LIMIT = 300_000_000
BASE_MODEL_NAME = "l3cube-pune/hing-roberta-mixed"
NUM_CLASSES = 56
OOS_CLASS_ID = 55

# Script-adaptive weights from nlu_pipeline.ipynb
SCRIPT_WEIGHTS = {
    "latin": 0.52,
    "devanagari": 0.32,
    "mixed": 0.45,
    "other": 0.45,
}
GEOM_EPS = 0.01

DEV_PATTERN = re.compile(r"[\u0900-\u097F]+")
ASR_MAP = {
    r"\bkia\b": "kya",
    r"\bmjhe\b": "mujhe",
    r"\bor\b": "aur",
    r"\bh\b": "hai",
    r"\bmai\b": "main",
    r"\bplz\b": "please",
    r"\bkr\b": "karo",
    r"\bthnx\b": "dhanyavad",
    r"\bchahie\b": "chahiye",
    r"\bbtaye\b": "batao",
    r"\bbtado\b": "batao",
    r"\bsoooong\b": "song",
}


def detect_script(text: str) -> str:
    s = str(text)
    has_dev = bool(re.search(r"[\u0900-\u097F]", s))
    has_lat = bool(re.search(r"[a-zA-Z]", s))
    if has_dev and has_lat:
        return "mixed"
    if has_dev:
        return "devanagari"
    if has_lat:
        return "latin"
    return "other"


def script_char_stats(text: str) -> dict[str, Any]:
    s = str(text)
    dev_chars = sum("\u0900" <= c <= "\u097F" for c in s)
    lat_chars = sum(c.isascii() and c.isalpha() for c in s)
    digit_chars = sum(c.isdigit() for c in s)
    total_alpha = dev_chars + lat_chars
    return {
        "script": detect_script(s),
        "devanagari_chars": dev_chars,
        "latin_chars": lat_chars,
        "digits": digit_chars,
        "char_length": len(s),
        "word_count": len(s.split()),
        "devanagari_pct": round(100.0 * dev_chars / total_alpha, 1) if total_alpha else 0.0,
        "latin_pct": round(100.0 * lat_chars / total_alpha, 1) if total_alpha else 0.0,
    }


def clean_transformer_text(text: str) -> str:
    """Light NFKC + whitespace cleanup matching `aavesham_v2.ipynb`."""
    t = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", t).strip()


def clean_asr(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", t)
    for pat, rep in ASR_MAP.items():
        t = re.sub(pat, rep, t)
    return re.sub(r"\s+", " ", t).strip()


def transliterate(text: str, xlit_dict: dict[str, str]) -> str:
    t_clean = clean_asr(text)
    if not DEV_PATTERN.search(t_clean):
        return t_clean
    t = DEV_PATTERN.sub(lambda m: xlit_dict.get(m.group(0), m.group(0)), t_clean)
    return re.sub(r"([a-z])\1{2,}", r"\1\1", t.lower())


class AaveshamNLUService:
    def __init__(self) -> None:
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.xlit_dict = self._load_xlit_dict()
        self.class_map = self._load_class_meanings()
        self.tokenizer, self.model, self.xlm_params = self._load_transformer()
        self.tfidf_vec, self.calibrated_svc, self.svc_fit_seconds = self._load_or_fit_svc()

        self.svc_vocab_size = len(self.tfidf_vec.vocabulary_)
        self.svc_params = self.svc_vocab_size * NUM_CLASSES + NUM_CLASSES
        self.total_active_params = self.xlm_params + self.svc_params
        self.param_headroom = PARAM_LIMIT - self.total_active_params
        assert (
            self.total_active_params <= PARAM_LIMIT
        ), f"Parameter budget exceeded: {self.total_active_params} > {PARAM_LIMIT}"

        self.benchmark_summary = self._compute_benchmark_summary()
        self.warmup()

    def _load_xlit_dict(self) -> dict[str, str]:
        with open(DATA_DIR / "xlit_word_dict.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_class_meanings(self) -> dict[int, dict[str, Any]]:
        with open(DATA_DIR / "verified_class_meanings.json", "r", encoding="utf-8") as f:
            rows = json.load(f)
        return {int(r["target"]): r for r in rows}

    def _load_transformer(self):
        if not MODEL_DIR.is_dir():
            raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        model.to(self.device)
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        return tokenizer, model, int(params)

    def _load_or_fit_svc(self):
        if SVC_CACHE_PATH.exists():
            with open(SVC_CACHE_PATH, "rb") as f:
                bundle = pickle.load(f)
            return bundle["tfidf_vec"], bundle["calibrated_svc"], bundle.get("fit_seconds", 0.0)

        t0 = time.perf_counter()
        train_df = pd.read_csv(DATA_DIR / "train.csv")
        train_df["script"] = train_df["query_text"].apply(detect_script)
        train_df["xlit"] = train_df["query_text"].apply(
            lambda q: transliterate(q, self.xlit_dict)
        )

        tfidf_vec = TfidfVectorizer(
            ngram_range=(1, 5),
            analyzer="char_wb",
            min_df=2,
            sublinear_tf=True,
        )
        x_train = tfidf_vec.fit_transform(train_df["xlit"])
        y_train = train_df["target"].values

        sample_weights = np.ones(len(train_df), dtype=np.float32)
        sample_weights[(train_df["script"] == "latin").values] = 2.0

        base_svc = LinearSVC(C=1.0, random_state=42, max_iter=2000)
        calibrated_svc = CalibratedClassifierCV(base_svc, cv=3)
        calibrated_svc.fit(x_train, y_train, sample_weight=sample_weights)
        fit_seconds = round(time.perf_counter() - t0, 2)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(SVC_CACHE_PATH, "wb") as f:
            pickle.dump(
                {
                    "tfidf_vec": tfidf_vec,
                    "calibrated_svc": calibrated_svc,
                    "fit_seconds": fit_seconds,
                },
                f,
            )
        return tfidf_vec, calibrated_svc, fit_seconds

    def warmup(self) -> None:
        self.predict("kal subah 7 baje ka alarm laga do")

    def _compute_benchmark_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "baselines": [],
            "high_conf_comparison": {},
            "script_breakdown": [],
            "dataset_stats": {},
        }

        comp_path = ROOT_DIR / "benchmark_models_comparison.csv"
        if comp_path.exists():
            comp_df = pd.read_csv(comp_path)
            for _, row in comp_df.iterrows():
                summary["baselines"].append(
                    {
                        "model": str(row["Model"]),
                        "mean_macro_f1": round(float(row["Mean Macro-F1"]), 4),
                        "oof_macro_f1": round(float(row["OOF Macro-F1"]), 4),
                        "mean_latency_ms": float(row["Mean Latency (ms)"]),
                        "p99_latency_ms": float(row["P99 Latency (ms)"]),
                        "parameters": str(row["Parameters"]),
                    }
                )

        bench_path = DATA_DIR / "test_ground_truth_benchmark.csv"
        pure_path = ROOT_DIR / "submission_aavesham_pure.csv"
        ens_path = ROOT_DIR / "submission.csv"

        if bench_path.exists() and pure_path.exists() and ens_path.exists():
            bench = pd.read_csv(bench_path)
            pure = pd.read_csv(pure_path)
            ens = pd.read_csv(ens_path)

            m_pure = pd.merge(bench, pure, on="example_id")
            m_ens = pd.merge(bench, ens, on="example_id")
            hc_mask = m_ens["confidence_tier"].isin(["exact_100pct", "high_conf_99pct"])
            hc_pure = m_pure[hc_mask]
            hc_ens = m_ens[hc_mask]
            y_true = hc_ens["ground_truth_target"].values

            pure_f1 = float(f1_score(y_true, hc_pure["target"].values, average="macro"))
            pure_acc = float(accuracy_score(y_true, hc_pure["target"].values))
            ens_f1 = float(f1_score(y_true, hc_ens["target"].values, average="macro"))
            ens_acc = float(accuracy_score(y_true, hc_ens["target"].values))

            # Class 55 metrics on high-confidence benchmark
            oos_true = y_true == OOS_CLASS_ID
            oos_pure = hc_pure["target"].values == OOS_CLASS_ID
            oos_ens = hc_ens["target"].values == OOS_CLASS_ID
            oos_pure_f1 = float(f1_score(oos_true, oos_pure, zero_division=0))
            oos_ens_f1 = float(f1_score(oos_true, oos_ens, zero_division=0))

            summary["high_conf_comparison"] = {
                "n_samples": int(len(hc_ens)),
                "standalone_macro_f1": round(pure_f1, 5),
                "standalone_accuracy": round(pure_acc, 5),
                "ensemble_macro_f1": round(ens_f1, 5),
                "ensemble_accuracy": round(ens_acc, 5),
                "macro_f1_gain": round(ens_f1 - pure_f1, 5),
                "class55_standalone_f1": round(oos_pure_f1, 4),
                "class55_ensemble_f1": round(oos_ens_f1, 4),
            }

            for sc in ["devanagari", "latin", "mixed"]:
                sm = (hc_ens["script"] == sc).values
                s_pure_f1 = float(
                    f1_score(y_true[sm], hc_pure["target"].values[sm], average="macro")
                )
                s_ens_f1 = float(
                    f1_score(y_true[sm], hc_ens["target"].values[sm], average="macro")
                )
                s_pure_acc = float(accuracy_score(y_true[sm], hc_pure["target"].values[sm]))
                s_ens_acc = float(accuracy_score(y_true[sm], hc_ens["target"].values[sm]))
                summary["script_breakdown"].append(
                    {
                        "script": sc,
                        "count": int(sm.sum()),
                        "standalone_f1": round(s_pure_f1, 4),
                        "ensemble_f1": round(s_ens_f1, 4),
                        "standalone_acc": round(s_pure_acc, 4),
                        "ensemble_acc": round(s_ens_acc, 4),
                        "delta_f1": round(s_ens_f1 - s_pure_f1, 4),
                    }
                )

        summary["dataset_stats"] = {
            "train_rows": 16936,
            "test_rows": 4912,
            "num_classes": NUM_CLASSES,
            "xlit_dict_entries": len(self.xlit_dict),
            "train_script_pct": {"devanagari": 61.13, "latin": 33.33, "mixed": 5.54},
            "test_script_pct": {"devanagari": 32.65, "latin": 64.21, "mixed": 3.14},
            "class_55_train_share_pct": 1.47,
            "class_55_train_count": 249,
        }
        return summary

    def get_system_stats(self) -> dict[str, Any]:
        classes = [self.class_map[i] for i in sorted(self.class_map.keys())]
        return {
            "model_info": {
                "base_model": BASE_MODEL_NAME,
                "architecture": "XLMRobertaForSequenceClassification + Calibrated LinearSVC",
                "device": str(self.device).upper(),
                "xlm_params": self.xlm_params,
                "svc_params": self.svc_params,
                "svc_vocab_size": self.svc_vocab_size,
                "total_active_params": self.total_active_params,
                "param_limit": PARAM_LIMIT,
                "headroom_params": self.param_headroom,
                "headroom_pct": round(100.0 * self.param_headroom / PARAM_LIMIT, 2),
                "compliant": self.total_active_params <= PARAM_LIMIT,
                "svc_fit_seconds": self.svc_fit_seconds,
                "script_weights": SCRIPT_WEIGHTS,
            },
            "benchmark": self.benchmark_summary,
            "classes": classes,
        }

    def predict(
        self,
        query_text: str,
        mode: str = "ensemble",
        oos_conf_threshold: float = 0.50,
        oos_guard_threshold: float = 0.22,
        apply_tail_multipliers: bool = True,
    ) -> dict[str, Any]:
        t_start = time.perf_counter()

        # 1. Preprocessing & Script Detection
        t_pre0 = time.perf_counter()
        raw_text = str(query_text or "").strip()
        if not raw_text:
            raise ValueError("Query text cannot be empty.")

        char_stats = script_char_stats(raw_text)
        script = char_stats["script"]
        transformer_text = clean_transformer_text(raw_text)
        xlit_text = transliterate(raw_text, self.xlit_dict)
        preprocess_ms = (time.perf_counter() - t_pre0) * 1000.0

        # 2. Transformer Inference (`models/best` from aavesham_v2)
        t_xlm0 = time.perf_counter()
        inputs = self.tokenizer(
            [transformer_text],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        token_count = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            logits = self.model(**inputs).logits
            if self.device.type == "mps":
                torch.mps.synchronize()
            elif self.device.type == "cuda":
                torch.cuda.synchronize()
            p_xlm = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        xlm_ms = (time.perf_counter() - t_xlm0) * 1000.0

        # 3. Calibrated LinearSVC Inference
        t_svc0 = time.perf_counter()
        x_vec = self.tfidf_vec.transform([xlit_text])
        p_svc = self.calibrated_svc.predict_proba(x_vec)[0]
        svc_ms = (time.perf_counter() - t_svc0) * 1000.0

        # 4. Script-Adaptive Geometric Fusion + Class 55 Fallback Logic
        t_fus0 = time.perf_counter()
        g_xlm = SCRIPT_WEIGHTS.get(script, 0.45) if mode == "ensemble" else 1.0
        g_svc = 1.0 - g_xlm

        if mode == "ensemble":
            log_p = g_xlm * np.log(p_xlm + GEOM_EPS) + g_svc * np.log(p_svc + GEOM_EPS)
            p_fused = np.exp(log_p)
            p_fused = p_fused / np.sum(p_fused)
        else:
            p_fused = p_xlm.copy()

        p_scoring = p_fused.copy()
        multipliers_applied = False
        if mode == "ensemble" and apply_tail_multipliers:
            p_scoring[55] *= 1.30
            p_scoring[53] *= 1.15
            p_scoring[12] *= 1.15
            multipliers_applied = True

        # Normalized scoring distribution for display
        p_scoring_norm = p_scoring / np.sum(p_scoring)
        raw_pred = int(np.argmax(p_scoring_norm))
        top_conf = float(p_scoring_norm[raw_pred])
        entropy = float(-np.sum(p_fused * np.log(np.clip(p_fused, 1e-12, 1.0))) / np.log(NUM_CLASSES))

        # Class 55 (Out-of-Scope) Fallback Pipeline:
        # Rule A (aavesham_v2.ipynb): If model predicts Class 55 with confidence < oos_conf_threshold,
        # reassign to 2nd-best in-scope intent (rescues false OOS triggers).
        # Rule B (Live OOS Guard): If top in-scope confidence < oos_guard_threshold (and guard > 0),
        # route low-confidence out-of-distribution utterances to Class 55 fallback.
        fallback_event = "none"
        final_pred = raw_pred
        if raw_pred == OOS_CLASS_ID and top_conf < oos_conf_threshold:
            masked = p_scoring_norm.copy()
            masked[OOS_CLASS_ID] = -1.0
            final_pred = int(np.argmax(masked))
            fallback_event = "reassigned_from_55_low_conf"
        elif raw_pred != OOS_CLASS_ID and oos_guard_threshold > 0 and top_conf < oos_guard_threshold:
            final_pred = OOS_CLASS_ID
            fallback_event = "routed_to_55_oos_guard"

        fusion_ms = (time.perf_counter() - t_fus0) * 1000.0
        total_ms = (time.perf_counter() - t_start) * 1000.0

        # Build Top-5 breakdown
        top5_idx = np.argsort(p_scoring_norm)[::-1][:5]
        if final_pred not in top5_idx:
            top5_idx = np.append(top5_idx[:4], final_pred)

        top5 = []
        for idx in top5_idx:
            cid = int(idx)
            meta = self.class_map.get(cid, {})
            top5.append(
                {
                    "target": cid,
                    "intent_name": meta.get("intent_name", f"class_{cid}"),
                    "domain": meta.get("domain", "unknown"),
                    "score_prob": round(float(p_scoring_norm[cid]), 4),
                    "ensemble_prob": round(float(p_fused[cid]), 4),
                    "xlm_prob": round(float(p_xlm[cid]), 4),
                    "svc_prob": round(float(p_svc[cid]), 4),
                    "is_final": cid == final_pred,
                }
            )

        pred_meta = self.class_map.get(final_pred, {})
        raw_meta = self.class_map.get(raw_pred, {})

        return {
            "query_text": raw_text,
            "transformer_text": transformer_text,
            "xlit_text": xlit_text,
            "mode": mode,
            "script_analysis": {
                **char_stats,
                "token_count": token_count,
                "g_xlm": round(g_xlm, 2),
                "g_svc": round(g_svc, 2),
            },
            "prediction": {
                "target": final_pred,
                "intent_name": pred_meta.get("intent_name", f"class_{final_pred}"),
                "domain": pred_meta.get("domain", "unknown"),
                "extra_note": pred_meta.get("extra_note", ""),
                "confidence": round(float(p_scoring_norm[final_pred]), 4),
                "raw_target": raw_pred,
                "raw_intent_name": raw_meta.get("intent_name", f"class_{raw_pred}"),
                "raw_confidence": round(top_conf, 4),
                "normalized_entropy": round(entropy, 4),
                "is_oos_55": final_pred == OOS_CLASS_ID,
            },
            "oos_fallback": {
                "class_55_prob": round(float(p_scoring_norm[OOS_CLASS_ID]), 4),
                "class_55_raw_xlm_prob": round(float(p_xlm[OOS_CLASS_ID]), 4),
                "class_55_raw_svc_prob": round(float(p_svc[OOS_CLASS_ID]), 4),
                "oos_conf_threshold": oos_conf_threshold,
                "oos_guard_threshold": oos_guard_threshold,
                "multipliers_applied": multipliers_applied,
                "event": fallback_event,
            },
            "top5": top5,
            "latency": {
                "preprocess_ms": round(preprocess_ms, 2),
                "xlm_ms": round(xlm_ms, 2),
                "svc_ms": round(svc_ms, 2),
                "fusion_ms": round(fusion_ms, 2),
                "total_ms": round(total_ms, 2),
                "throughput_qps": round(1000.0 / max(total_ms, 0.01), 1),
            },
        }

    def compare_scripts(
        self,
        variants: list[dict[str, str]],
        mode: str = "ensemble",
        oos_conf_threshold: float = 0.50,
        oos_guard_threshold: float = 0.22,
        apply_tail_multipliers: bool = True,
    ) -> dict[str, Any]:
        results = []
        for item in variants:
            text = str(item.get("text", "")).strip()
            label = str(item.get("label", "Variant"))
            if not text:
                continue
            res = self.predict(
                text,
                mode=mode,
                oos_conf_threshold=oos_conf_threshold,
                oos_guard_threshold=oos_guard_threshold,
                apply_tail_multipliers=apply_tail_multipliers,
            )
            results.append({"label": label, **res})

        targets = [r["prediction"]["target"] for r in results]
        if targets:
            counts = {t: targets.count(t) for t in set(targets)}
            consensus_target = max(counts, key=counts.get)
            agreement_pct = round(100.0 * counts[consensus_target] / len(targets), 1)
            consensus_meta = self.class_map.get(consensus_target, {})
        else:
            consensus_target = None
            agreement_pct = 0.0
            consensus_meta = {}

        avg_latency = (
            round(sum(r["latency"]["total_ms"] for r in results) / len(results), 2)
            if results
            else 0.0
        )

        return {
            "consensus_target": consensus_target,
            "consensus_intent": consensus_meta.get("intent_name", ""),
            "consensus_domain": consensus_meta.get("domain", ""),
            "agreement_pct": agreement_pct,
            "avg_latency_ms": avg_latency,
            "results": results,
        }
