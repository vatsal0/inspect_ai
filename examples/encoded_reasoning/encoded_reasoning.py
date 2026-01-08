from inspect_ai import Task, task

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate, system_message, TaskState
from inspect_ai.scorer import (
    CORRECT, INCORRECT, NOANSWER, Score, Target,
    mean, stderr,
    Scorer, scorer
)

import random
import os

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

from .prompts import \
ZERO_SHOT_SYSTEM_PROMPT, DOUBLE_ZERO_SHOT_SYSTEM_PROMPT, SINGLE_COT_SYSTEM_PROMPT, \
DOUBLE_COT_SYSTEM_PROMPT, ENCODED_SYSTEM_PROMPT, ENCODED_SYSTEM_PROMPT_HINT, \
PAUSE_SYSTEM_PROMPT, PAUSE_SYSTEM_PROMPT_FIXED, PAUSE_SYSTEM_PROMPT_HINT, \
BLIND_SYSTEM_PROMPT_TRACE_BEFORE, BLIND_SYSTEM_PROMPT_TRACE_AFTER

from .dataset import custom_dataset

from .solvers import blind_solver, double_zero_solver, blind_general_solver, double_blind_solver


@scorer(metrics=[mean(), stderr()])
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

        return Score(
            value=NOANSWER if not parsed_answer.isdigit()
            else CORRECT if parsed_answer == targets[0]
            else INCORRECT, 
            answer=answer
        )

    return score

dataset_kwargs = {} # want to get these from command line
@task
def zero_shot():
    return Task(
        dataset=custom_dataset(**dataset_kwargs),
        solver=[system_message(ZERO_SHOT_SYSTEM_PROMPT), generate()],
        scorer=xml_answer(['</reasoning>', '<answer>', '</answer>']),
    )