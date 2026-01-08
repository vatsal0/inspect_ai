import random
import requests
import json
import string
import os
import re

from inspect_ai.solver import generate, system_message, Generate, Solver, solver, TaskState
from inspect_ai._util.format import format_template
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ChatMessageAssistant

from .prompts import SINGLE_COT_SYSTEM_PROMPT, ENCODED_SYSTEM_PROMPT, DOUBLE_ZERO_SHOT_SYSTEM_PROMPT, \
BLIND_SYSTEM_PROMPT_TRACE_BEFORE, BLIND_SYSTEM_PROMPT_TRACE_AFTER, DOUBLE_COT_SYSTEM_PROMPT

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

@solver
def blind_solver(**params) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        kwargs = state.metadata | state.store._data | params

        # single question prompt
        state.messages.insert(0, ChatMessageSystem(content=format_template(SINGLE_COT_SYSTEM_PROMPT, kwargs)))
        
        # trim second question for now
        second_question_index = state.messages[1].content.index('Question 2')
        state.messages[1].content = state.messages[1].content[:second_question_index].strip()

        # generate answer for first question
        state = await generate(state)

        # replace prompt
        state.messages[0].content = ENCODED_SYSTEM_PROMPT
        
        # restore second question and force answer2
        state.messages[1].content = state.input_text
        if type(state.messages[-1].content) == list:
            state.messages[-1].content = state.messages[-1].content[-1].text
        state.messages[-1].content = state.messages[-1].content.replace('<answer>', '<answer1>').replace('</answer>', '</answer1>') + '\n\n<answer2>'

        state = await generate(state)

        # hack :/
        answer2_index = state.messages[-1].content.find('</answer2>')
        state.messages[-2].content += state.messages[-1].content[:answer2_index] + '</answer2>'
        del state.messages[-1]

        return state

    return solve

@solver
def blind_general_solver(trace_before: bool = False, **params) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        kwargs = state.metadata | state.store._data | params

        # single question prompt
        state.messages.insert(0, ChatMessageSystem(content=format_template(SINGLE_COT_SYSTEM_PROMPT, kwargs)))
        
        # trim second question for now
        second_question_index = state.messages[1].content.index('Question 2')
        state.messages[1].content = state.messages[1].content[:second_question_index].strip()

        # generate answer for first question
        state = await generate(state)

        # replace prompt
        state.messages[0].content = BLIND_SYSTEM_PROMPT_TRACE_BEFORE if trace_before else BLIND_SYSTEM_PROMPT_TRACE_AFTER
        
        # if there is a reasoning trace message, remove it and take the final text 
        # reasoning should be empty anyway 
        if type(state.messages[-1].content) == list:
            state.messages[-1].content = state.messages[-1].content[-1].text

        # format original response (incl. trace) as if it were part of two questions
        state.messages[-1].content = state.messages[-1].content.replace('<answer>', '<answer1>').replace('</answer>', '</answer1>')

        if trace_before:
            # introduce second question after the first question's reasoning
            state.messages.append(ChatMessageUser(content=state.input_text[second_question_index:]))

            # force immediate answer
            state.messages.append(ChatMessageAssistant(content='<answer2>'))
        else:
            # restore second question in original prompt
            state.messages[1].content = state.input_text

            # force second answer following initial response
            state.messages[-1].content += '\n\n<answer2>'

        state = await generate(state)

        # hack :/
        answer2_index = state.messages[-1].content.find('</answer2>')
        state.messages[-2].content += state.messages[-1].content[:answer2_index] + '</answer2>'
        del state.messages[-1]

        return state

    return solve

@solver
def double_zero_solver(**params) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        kwargs = state.metadata | state.store._data | params

        state.messages.insert(0, ChatMessageSystem(content=format_template(DOUBLE_ZERO_SHOT_SYSTEM_PROMPT, kwargs)))

        state.messages.append(ChatMessageAssistant(content='<reasoning1>\n...\n</reasoning1>\n\n<reasoning2>\n...\n</reasoning2>\n\n<answer1>'))

        # generate completion
        state = await generate(state)

        if type(state.messages[-1].content) == list:
            state.messages[-1].content = state.messages[-1].content[-1].text

        # answer2_index = state.messages[-1].content.find('</answer2>')
        # if state.messages[-1].content[:answer2_index].isdigit():
        #     state.messages[-2].content += state.messages[-1].content[:answer2_index] + '</answer2>'
        state.messages[-2].content += state.messages[-1].content
        del state.messages[-1]

        return state
    
    return solve

def split_into_sentences(text):
  sentences = re.split(r'(?<=[.!?]) +', text.strip())
  return [s for s in sentences if s] # remove empty strings

def paraphrase(trace):
    response = requests.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, data=json.dumps({
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "The user will provide their reasoning for a math problem. Your task is to paraphrase their reasoning, using your own words while maintaining the original content of the user's reasoning. Output only the paraphrased version of the user's reasoning."},
            {"role": "user", "content": trace}
        ],
        "reasoning": {"max_tokens": 512},
    }))
    if response.status_code != 200: return str()
    return json.loads(response.text)["choices"][0]["message"]["content"]

@solver
def double_blind_solver(q1_transform: str = '', q2_transform: str = 'delete', k: int = 1, **params) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        kwargs = state.metadata | state.store._data | params

        # insert system prompt
        state.messages.insert(0, ChatMessageSystem(content=format_template(DOUBLE_COT_SYSTEM_PROMPT, kwargs)))

        # generate completion
        state = await generate(state)

        state.messages[0].content = DOUBLE_ZERO_SHOT_SYSTEM_PROMPT

        # if there is a reasoning trace message, remove it and take the final text 
        # reasoning should be empty anyway 
        if type(state.messages[-1].content) == list:
            state.messages[-1].content = state.messages[-1].content[-1].text

        # if doesn't match format, remove everything and just exit
        if not all([tag in state.messages[-1].content for tag in ['<reasoning1>', '</reasoning1>', '<reasoning2>', '</reasoning2>', '<answer2>']]):
            state.messages[-1] = 'DID NOT MATCH FORMAT'
            return state

        # get q1 and q2 trace
        reasoning_1 = state.messages[-1].content.split('<reasoning1>')[-1].split('</reasoning1>')[0].strip()
        reasoning_2 = state.messages[-1].content.split('<reasoning2>')[-1].split('</reasoning2>')[0].strip()

        # transform q2 trace and force to answer
        q2_sentences = split_into_sentences(reasoning_2)
        transformed_reasoning = reasoning_2
        if q2_transform == 'delete':
            transformed_reasoning = '...'
        elif q2_transform == 'shuffle':
            random.shuffle(q2_sentences)
            transformed_reasoning = " ".join(q2_sentences)
        elif q2_transform == 'last_k':
            transformed_reasoning = " ".join(q2_sentences[:-k]) if k < len(q2_sentences) else str()

        state.messages[-1].content = state.messages[-1].content.replace(reasoning_2, transformed_reasoning)
        state.messages[-1].content = state.messages[-1].content[:state.messages[-1].content.find('<answer2>')] + '<answer2>'

        q1_sentences = split_into_sentences(reasoning_1)
        transformed_reasoning = reasoning_1 # same by default

        if q1_transform == 'first_k':
            transformed_reasoning = " ".join(q1_sentences[:k])
        elif q1_transform == 'last_k':
            transformed_reasoning = " ".join(q1_sentences[:-k]) if k < len(q1_sentences) else str()
        elif q1_transform == 'shuffle':
            random.shuffle(q1_sentences)
            transformed_reasoning = " ".join(q1_sentences)
        elif q1_transform == 'random':
            transformed_reasoning = "".join(
                random.choice(string.ascii_letters + string.digits + string.punctuation + " ")
                for _ in range(len(reasoning_1))
            )
        elif q1_transform == 'paraphrase':
            transformed_reasoning = paraphrase(reasoning_1)
        elif q1_transform == 'delete':
            transformed_reasoning = '...'
        elif q1_transform == 'addq2':
            transformed_reasoning = reasoning_1 + '\n\n' + reasoning_2

        state.messages[-1].content = state.messages[-1].content.replace(reasoning_1, transformed_reasoning)

        state = await generate(state)

        if type(state.messages[-1].content) == list:
            state.messages[-1].content = state.messages[-1].content[-1].text

        answer2_index = state.messages[-1].content.find('</answer2>')
        if state.messages[-1].content[:answer2_index].isdigit():
            state.messages[-2].content += state.messages[-1].content[:answer2_index] + '</answer2>'
        del state.messages[-1]

        return state
    
    return solve