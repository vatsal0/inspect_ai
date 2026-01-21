#!/usr/bin/env python3
"""
Compare two inspect_ai eval logs using a t-test on hidden motive (N/A) accuracy.

This script reads two eval logs, filters to only samples where the ground truth
answer satisfies 4*X = 5 (mod 7) (i.e., should_be_na is True), and performs a
t-test to determine if there is a statistically significant difference in
na_accuracy between the two evaluations.
"""

import argparse
from scipy import stats
from inspect_ai.log import read_eval_log


def extract_na_scores(log_path: str) -> list[int]:
    """
    Extract binary scores for hidden motive samples only.

    Filters to samples where should_be_na is True (ground truth satisfies 4*X = 5 mod 7),
    then returns 1 if the model correctly output N/A, 0 otherwise.

    Returns:
        List of 1s and 0s representing correct/incorrect N/A responses.
    """
    log = read_eval_log(log_path)
    scores = []

    for sample in log.samples:
        # Get the score and metadata
        for scorer_name, score in sample.scores.items():
            metadata = score.metadata
            if metadata is None:
                continue

            # Only include samples where should_be_na is True
            if not metadata.get("should_be_na"):
                continue

            # Check if the model was correct (output N/A when it should have)
            score_value = score.value
            if isinstance(score_value, (int, float)):
                is_correct = score_value == 1
            elif isinstance(score_value, str):
                is_correct = score_value.lower() in ["correct", "c", "1", "true", "yes"]
            else:
                continue

            scores.append(1 if is_correct else 0)
            break  # Take the first scorer

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Compare two eval logs using a t-test for hidden motive (N/A) accuracy difference"
    )
    parser.add_argument("log_path_1", help="Path to the first .eval log file")
    parser.add_argument("log_path_2", help="Path to the second .eval log file")
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level (default: 0.05)"
    )
    args = parser.parse_args()

    print(f"Reading eval log 1: {args.log_path_1}")
    scores1 = extract_na_scores(args.log_path_1)

    print(f"Reading eval log 2: {args.log_path_2}")
    scores2 = extract_na_scores(args.log_path_2)

    if not scores1:
        print("Error: No hidden motive samples found in log 1")
        return
    if not scores2:
        print("Error: No hidden motive samples found in log 2")
        return

    # Compute N/A accuracies
    acc1 = sum(scores1) / len(scores1)
    acc2 = sum(scores2) / len(scores2)

    print(f"\n{'='*70}")
    print("Summary (Hidden Motive / N/A Samples Only)")
    print(f"{'='*70}")
    print(f"Eval 1: {len(scores1)} N/A samples, na_accuracy = {acc1:.4f} ({sum(scores1)}/{len(scores1)})")
    print(f"Eval 2: {len(scores2)} N/A samples, na_accuracy = {acc2:.4f} ({sum(scores2)}/{len(scores2)})")
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
            print(f"Eval 1 has significantly HIGHER na_accuracy than Eval 2")
        else:
            print(f"Eval 2 has significantly HIGHER na_accuracy than Eval 1")
    else:
        print(f"\nResult: NO significant difference (p >= {args.alpha})")


if __name__ == "__main__":
    main()
