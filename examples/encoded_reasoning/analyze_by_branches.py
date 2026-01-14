#!/usr/bin/env python3
"""
Analyze eval log by num_branches.

This script reads an inspect_ai eval log, filters out samples with "noanswer"/"N" scores,
computes num_branches for each problem, and reports accuracy per num_branches bin.
"""

import argparse
import json
import re
from collections import defaultdict
from inspect_ai.log import read_eval_log


def count_branches(expr: str) -> int:
    """
    Count the number of branches in an arithmetic expression.

    A branch represents a binary operation node in the expression tree.
    For example:
    - "1 + 2" has 1 branch
    - "(1 + 2) * 3" has 2 branches
    - "((1 + 2) * 3) + 4" has 3 branches

    The num_branches equals the number of operators in the expression.
    """
    # Extract the expression from the problem text
    # The format is: "Evaluate this Python expression. <expr>"
    if "Evaluate this Python expression." in expr:
        expr = expr.split("Evaluate this Python expression.")[-1].strip()

    # Also handle cases where there might be filler text appended
    # The expression ends before any newlines
    expr = expr.split("\n")[0].strip()

    # Count operators: +, -, *, //, %
    # We need to be careful with // vs / and negative numbers
    # Use regex to find all operators that are not part of numbers

    # Count binary operators by looking for operators surrounded by spaces
    # Binary operators are formatted as " op " (with spaces on both sides)
    # Unary minus is attached to the number without spaces (e.g., -68)

    # Match operators surrounded by spaces: +, -, *, //, %
    # The pattern matches: space, operator, space
    operators = re.findall(r' (\+|-|\*|//|%) ', expr)

    return len(operators)


def main():
    parser = argparse.ArgumentParser(description="Analyze eval log by num_branches")
    parser.add_argument("log_path", help="Path to the .eval log file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed output")
    parser.add_argument("--save-correct", type=str, help="Save correct Q/A pairs to this file (JSON)")
    args = parser.parse_args()

    print(f"Reading eval log: {args.log_path}")
    log = read_eval_log(args.log_path)

    # Group samples by num_branches
    bins = defaultdict(lambda: {"correct": 0, "incorrect": 0, "total": 0, "noanswer": 0})

    # Collect correct Q/A pairs
    correct_pairs = []

    total_count = 0

    for sample in log.samples:
        total_count += 1

        # Get the score - check for "noanswer" or "N"
        # The score is typically in sample.scores dict
        score_value = None
        is_correct = None

        for scorer_name, score in sample.scores.items():
            score_value = score.value
            break  # Take the first scorer

        # Get the problem text first - we want the LAST user message (the actual problem, not few-shot examples)
        if isinstance(sample.input, str):
            problem_text = sample.input
        elif isinstance(sample.input, list):
            # It's a list of messages, get the last user message (the actual problem)
            problem_text = ""
            for msg in reversed(sample.input):
                if hasattr(msg, 'role') and msg.role == 'user':
                    problem_text = str(msg.content)
                    break
        else:
            problem_text = str(sample.input)

        # Count branches
        num_branches = count_branches(problem_text)

        # Check for "noanswer" and "N" scores
        if score_value in ["noanswer", "N"]:
            bins[num_branches]["noanswer"] += 1
            if args.verbose:
                print(f"Problem: {problem_text[:80]}... -> branches: {num_branches}, noanswer")
            continue

        # Determine if correct
        if isinstance(score_value, (int, float)):
            is_correct = score_value == 1
        elif isinstance(score_value, str):
            is_correct = score_value.lower() in ["correct", "c", "1", "true", "yes"]
        else:
            # Skip if we can't determine correctness
            bins[num_branches]["noanswer"] += 1
            continue

        if args.verbose:
            print(f"Problem: {problem_text[:80]}... -> branches: {num_branches}, correct: {is_correct}")

        bins[num_branches]["total"] += 1
        if is_correct:
            bins[num_branches]["correct"] += 1
            # Collect correct Q/A pair
            model_answer = sample.output.completion if sample.output else ""
            correct_pairs.append({
                "question": problem_text,
                "expected_answer": str(sample.target) if sample.target else "",
                "model_answer": model_answer,
                "num_branches": num_branches,
            })
        else:
            bins[num_branches]["incorrect"] += 1


    # Compute totals
    overall_correct = 0
    overall_total = 0
    overall_noanswer = 0

    for stats in bins.values():
        overall_correct += stats["correct"]
        overall_total += stats["total"]
        overall_noanswer += stats["noanswer"]

    # Print results
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"Total samples: {total_count}")
    print(f"No answer (noanswer/N): {overall_noanswer}")
    print(f"Analyzed: {overall_total}")

    print(f"\n{'='*70}")
    print(f"Accuracy by num_branches")
    print(f"{'='*70}")
    print(f"{'num_branches':<15} {'correct':<10} {'total':<10} {'noanswer':<10} {'accuracy':<10}")
    print(f"{'-'*55}")

    for num_branches in sorted(bins.keys()):
        stats = bins[num_branches]
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{num_branches:<15} {stats['correct']:<10} {stats['total']:<10} {stats['noanswer']:<10} {accuracy:.4f}")

    print(f"{'-'*55}")
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0
    print(f"{'Overall':<15} {overall_correct:<10} {overall_total:<10} {overall_noanswer:<10} {overall_acc:.4f}")

    # Print/save correct Q/A pairs
    print(f"\n{'='*70}")
    print(f"Correct Q/A Pairs: {len(correct_pairs)}")
    print(f"{'='*70}")

    if args.save_correct:
        with open(args.save_correct, "w") as f:
            json.dump(correct_pairs, f, indent=2)
        print(f"Saved {len(correct_pairs)} correct Q/A pairs to {args.save_correct}")
    else:
        # Print first few correct pairs
        for i, pair in enumerate(correct_pairs[:10]):
            print(f"\n[{i+1}] Question: {pair['question'][:100]}...")
            print(f"    Expected: {pair['expected_answer']}")
            print(f"    Model: {pair['model_answer'][:100]}..." if len(pair['model_answer']) > 100 else f"    Model: {pair['model_answer']}")
        if len(correct_pairs) > 10:
            print(f"\n... and {len(correct_pairs) - 10} more. Use --save-correct to save all.")


if __name__ == "__main__":
    main()
