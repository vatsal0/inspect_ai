#!/usr/bin/env python3
"""
Compare two inspect_ai eval logs using a t-test.

This script reads two eval logs, extracts correct/incorrect scores (excluding noanswer/N),
and performs a t-test to determine if there is a statistically significant difference
in accuracy between the two evaluations.
"""

import argparse
from scipy import stats
from inspect_ai.log import read_eval_log


def extract_scores(log_path: str) -> list[int]:
    """
    Extract binary scores (1 for correct, 0 for incorrect) from an eval log.
    Excludes samples with "noanswer" or "N" scores.

    Returns:
        List of 1s and 0s representing correct/incorrect answers.
    """
    log = read_eval_log(log_path)
    scores = []

    for sample in log.samples:
        # Get the score
        score_value = None
        for scorer_name, score in sample.scores.items():
            score_value = score.value
            break  # Take the first scorer

        # Skip "noanswer" and "N" scores
        if score_value in ["noanswer", "N"]:
            continue

        # Determine if correct
        if isinstance(score_value, (int, float)):
            is_correct = score_value == 1
        elif isinstance(score_value, str):
            is_correct = score_value.lower() in ["correct", "c", "1", "true", "yes"]
        else:
            continue

        scores.append(1 if is_correct else 0)

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Compare two eval logs using a t-test for accuracy difference"
    )
    parser.add_argument("log_path_1", help="Path to the first .eval log file")
    parser.add_argument("log_path_2", help="Path to the second .eval log file")
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level (default: 0.05)"
    )
    args = parser.parse_args()

    print(f"Reading eval log 1: {args.log_path_1}")
    scores1 = extract_scores(args.log_path_1)

    print(f"Reading eval log 2: {args.log_path_2}")
    scores2 = extract_scores(args.log_path_2)

    # Compute accuracies
    acc1 = sum(scores1) / len(scores1) if scores1 else 0
    acc2 = sum(scores2) / len(scores2) if scores2 else 0

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Eval 1: {len(scores1)} samples, accuracy = {acc1:.4f} ({sum(scores1)}/{len(scores1)})")
    print(f"Eval 2: {len(scores2)} samples, accuracy = {acc2:.4f} ({sum(scores2)}/{len(scores2)})")
    print(f"Difference: {acc1 - acc2:+.4f}")

    # Perform independent samples t-test
    # Using Welch's t-test (does not assume equal variances)
    t_stat, p_value = stats.ttest_ind(scores1, scores2, equal_var=False)

    print(f"\n{'='*70}")
    print("T-Test Results (Welch's t-test)")
    print(f"{'='*70}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Significance level (alpha): {args.alpha}")

    if p_value < args.alpha:
        print(f"\nResult: SIGNIFICANT difference (p < {args.alpha})")
        if acc1 > acc2:
            print(f"Eval 1 has significantly HIGHER accuracy than Eval 2")
        else:
            print(f"Eval 2 has significantly HIGHER accuracy than Eval 1")
    else:
        print(f"\nResult: NO significant difference (p >= {args.alpha})")


if __name__ == "__main__":
    main()
