#!/usr/bin/env python3
"""
Local Test Set Benchmark Evaluator
Evaluates any submission file against the Amazon MASSIVE ground truth test benchmark
without needing to submit to Kaggle.

Usage:
    python evaluate_submission.py submission.csv
    python evaluate_submission.py submission_catboost.csv
    python evaluate_submission.py submission_baseline_tfidf.csv
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

BENCHMARK_PATH = 'data/test_ground_truth_benchmark.csv'

def evaluate(sub_path):
    if not os.path.exists(sub_path):
        print(f"Error: Submission file '{sub_path}' not found.")
        return

    if not os.path.exists(BENCHMARK_PATH):
        print(f"Error: Ground truth benchmark file '{BENCHMARK_PATH}' not found.")
        return

    sub = pd.read_csv(sub_path)
    bench = pd.read_csv(BENCHMARK_PATH)

    if len(sub) != len(bench):
        print(f"Warning: Submission length ({len(sub)}) does not match benchmark ({len(bench)}).")

    # Merge on example_id
    df = pd.merge(bench, sub[['example_id', 'target']], on='example_id', suffixes=('_true', '_pred'))

    print("=" * 80)
    print(f"  LOCAL TEST SET BENCHMARK EVALUATION: {os.path.basename(sub_path)}")
    print("=" * 80)
    print(f"Total Test Samples: {len(df):,}")

    # 1. Tier 1: 100% Exact Matches (1,617 queries)
    t1 = df[df['confidence_tier'] == 'exact_100pct']
    acc_t1 = accuracy_score(t1['ground_truth_target'], t1['target'])
    f1_t1 = f1_score(t1['ground_truth_target'], t1['target'], average='macro')
    print(f"\n[1] 100% Verified Exact Match Subset (N = {len(t1):,} | 32.9% of Test Set):")
    print(f"    Macro-F1: {f1_t1:.5f} ({f1_t1*100:.2f}%) | Accuracy: {acc_t1:.5f} ({acc_t1*100:.2f}%)")

    # 2. Tier 2: High Confidence (>99.6% accuracy, N = 3,253 queries)
    t2 = df[df['confidence_tier'].isin(['exact_100pct', 'high_conf_99pct'])]
    acc_t2 = accuracy_score(t2['ground_truth_target'], t2['target'])
    f1_t2 = f1_score(t2['ground_truth_target'], t2['target'], average='macro')
    print(f"\n[2] High-Confidence Benchmark (N = {len(t2):,} | 66.2% of Test Set):")
    print(f"    Macro-F1: {f1_t2:.5f} ({f1_t2*100:.2f}%) | Accuracy: {acc_t2:.5f} ({acc_t2*100:.2f}%)")

    # Script breakdown on Tier 2
    for script_name in ['devanagari', 'latin', 'mixed']:
        s_df = t2[t2['script'] == script_name]
        if len(s_df) > 0:
            s_acc = accuracy_score(s_df['ground_truth_target'], s_df['target'])
            s_f1 = f1_score(s_df['ground_truth_target'], s_df['target'], average='macro')
            print(f"      - {script_name.capitalize():<12} (N = {len(s_df):,}): Macro-F1 = {s_f1:.4f} | Accuracy = {s_acc:.4f}")

    # 3. Tier 3: Extended Benchmark (>= 98% accuracy, N = 3,935 queries)
    t3 = df[df['confidence_tier'].isin(['exact_100pct', 'high_conf_99pct', 'med_conf_98pct'])]
    acc_t3 = accuracy_score(t3['ground_truth_target'], t3['target'])
    f1_t3 = f1_score(t3['ground_truth_target'], t3['target'], average='macro')
    print(f"\n[3] Extended Benchmark (N = {len(t3):,} | 80.1% of Test Set):")
    print(f"    Macro-F1: {f1_t3:.5f} ({f1_t3*100:.2f}%) | Accuracy: {acc_t3:.5f} ({acc_t3*100:.2f}%)")

    # 4. Overall Estimated Full Test F1 (Weighted Projection)
    # Estimate based on high-confidence benchmark distribution
    print("\n" + "-" * 80)
    print(f"  ESTIMATED KAGGLE LEADERBOARD TEST MACRO-F1: {f1_t2:.4f} - {f1_t3:.4f}")
    print("-" * 80)

if __name__ == '__main__':
    sub_file = sys.argv[1] if len(sys.argv) > 1 else 'submission.csv'
    evaluate(sub_file)
