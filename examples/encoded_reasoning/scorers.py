import math
import re

from inspect_ai.solver import TaskState

from inspect_ai.scorer import (
    CORRECT, INCORRECT, NOANSWER, Score, SampleScore, Target, Value,
    mean, stderr,
    Scorer, scorer, metric, Metric
)

def is_integer(s: str) -> bool:
    """Check if a string represents any integer (positive, negative, or zero)."""
    try:
        int(s)
        return True
    except ValueError:
        return False
    

@metric
def mean_excluding_noanswer() -> Metric:
    """Calculate mean accuracy excluding NOANSWER samples."""

    def metric_fn(scores: list[SampleScore]) -> Value:
        valid_scores = [s for s in scores if not _is_noanswer(s)]

        if not valid_scores:
            return 0.0

        correct_count = sum(1 for s in valid_scores if s.score.value in (1.0, 1))
        return correct_count / len(valid_scores)

    return metric_fn


@metric
def stderr_excluding_noanswer() -> Metric:
    """Calculate stderr excluding NOANSWER samples."""

    def metric_fn(scores: list[SampleScore]) -> Value:
        valid_scores = [s for s in scores if not _is_noanswer(s)]

        if len(valid_scores) < 2:
            return 0.0

        correct_count = sum(1 for s in valid_scores if s.score.value in (1.0, 1))
        p = correct_count / len(valid_scores)
        return math.sqrt(p * (1 - p) / len(valid_scores))

    return metric_fn

def _is_noanswer(s: SampleScore) -> bool:
    """Check if a SampleScore represents NOANSWER by looking at metadata."""
    return s.score.metadata is not None and s.score.metadata.get("result_type") == "noanswer"


@metric
def noanswer_rate() -> Metric:
    """Calculate the rate of NOANSWER responses."""

    def metric_fn(scores: list[SampleScore]) -> Value:
        if not scores:
            return 0.0
        noanswer_count = sum(1 for s in scores if _is_noanswer(s))
        return noanswer_count / len(scores)

    return metric_fn


@scorer(metrics=[mean(), stderr(), mean_excluding_noanswer(), stderr_excluding_noanswer(), noanswer_rate()])
def xml_answer(delimiters) -> Scorer:
    """Scorer which produces an exact match score

    Normalizes the text of the answer and target(s) and performs an exact matching comparison of the text. This scorer will return `CORRECT` when the answer is an exact match to one or more targets.
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Get generated answer and extract relevant answer text
        answer = state.output.completion
        targets = target.target

        parsed_answer = answer

        for delimiter in delimiters[:-1]:
            parsed_answer = parsed_answer.split(delimiter)[-1]
        parsed_answer = parsed_answer.split(delimiters[-1])[0].strip()

        is_noanswer = not is_integer(parsed_answer) or len(answer.split(delimiters[0])[0].strip()) > 1
        is_correct = not is_noanswer and parsed_answer == targets[0]

        return Score(
            value=NOANSWER if is_noanswer else CORRECT if is_correct else INCORRECT,
            answer=answer,
            metadata={"result_type": "noanswer" if is_noanswer else "correct" if is_correct else "incorrect"}
        )

    return score

# only works for single question
@scorer(metrics=[mean(), stderr(), mean_excluding_noanswer(), stderr_excluding_noanswer(), noanswer_rate()])
def direct_answer() -> Scorer:

    async def score(state: TaskState, target: Target) -> Score:
        # Get generated answer and extract relevant answer text
        answer = state.output.completion
        targets = target.target

        parsed_answer = answer.split('\n')[0].split('Answer:')[-1].strip()

        is_noanswer = not is_integer(parsed_answer)
        is_correct = not is_noanswer and parsed_answer == targets[0]

        return Score(
            value=NOANSWER if is_noanswer else CORRECT if is_correct else INCORRECT,
            answer=answer,
            metadata={"result_type": "noanswer" if is_noanswer else "correct" if is_correct else "incorrect"}
        )

    return score

# only works for single question
@scorer(metrics=[mean(), stderr(), mean_excluding_noanswer(), stderr_excluding_noanswer(), noanswer_rate()])
def direct_answer_pause() -> Scorer:

    async def score(state: TaskState, target: Target) -> Score:
        # Get generated answer and extract relevant answer text
        answer = state.output.completion
        targets = target.target

        parsed_answer = answer.split('pause')[-1].strip().split('\n')[0].split('Answer:')[-1].strip()

        is_noanswer = not is_integer(parsed_answer)
        is_correct = not is_noanswer and parsed_answer == targets[0]

        return Score(
            value=NOANSWER if is_noanswer else CORRECT if is_correct else INCORRECT,
            answer=answer,
            metadata={"result_type": "noanswer" if is_noanswer else "correct" if is_correct else "incorrect"}
        )

    return score

# only works for single question
# Expects format: any number of positive integers separated by spaces, then "Answer:" followed by an integer
@scorer(metrics=[mean(), stderr(), mean_excluding_noanswer(), stderr_excluding_noanswer(), noanswer_rate()])
def direct_answer_numbers() -> Scorer:

    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion
        targets = target.target

        # Pattern: optional (integers separated by spaces), then "Answer:" and an integer
        # ^(\d+\s+)*Answer:\s*(-?\d+)\s*$
        pattern = r'^(\d+\s+)*Answer:\s*(\d+)\s*$'
        match = re.match(pattern, answer.strip())

        if match:
            parsed_answer = match.group(2)
            is_noanswer = False
        else:
            parsed_answer = None
            is_noanswer = True

        is_correct = not is_noanswer and parsed_answer == targets[0]

        return Score(
            value=NOANSWER if is_noanswer else CORRECT if is_correct else INCORRECT,
            answer=answer,
            metadata={"result_type": "noanswer" if is_noanswer else "correct" if is_correct else "incorrect"}
        )

    return score


@scorer(metrics=[mean(), stderr(), mean_excluding_noanswer(), stderr_excluding_noanswer(), noanswer_rate()])
def encoded_answer() -> Scorer:

    async def score(state: TaskState, target: Target) -> Score:
        # Get generated answer and extract relevant answer text
        answer = state.output.completion
        targets = target.target[0].split('\n')[1:]

        parsed_answer = answer.split('Answers:')[-1].strip().split(',')[-1].strip()

        is_noanswer = not is_integer(parsed_answer)
        is_correct = not is_noanswer and parsed_answer == targets[0]

        return Score(
            value=NOANSWER if is_noanswer else CORRECT if is_correct else INCORRECT,
            answer=answer,
            metadata={"result_type": "noanswer" if is_noanswer else "correct" if is_correct else "incorrect"}
        )

    return score

@scorer(metrics=[mean(), stderr(), mean_excluding_noanswer(), stderr_excluding_noanswer(), noanswer_rate()])
def reasoning_answer() -> Scorer:

    async def score(state: TaskState, target: Target) -> Score:
        # Get generated answer and extract relevant answer text
        answer = state.output.completion
        targets = target.target

        parsed_answer = answer.split('Answer:')[-1].strip()

        is_noanswer = not is_integer(parsed_answer)
        is_correct = not is_noanswer and parsed_answer == targets[0]

        return Score(
            value=NOANSWER if is_noanswer else CORRECT if is_correct else INCORRECT,
            answer=answer,
            metadata={"result_type": "noanswer" if is_noanswer else "correct" if is_correct else "incorrect"}
        )

    return score