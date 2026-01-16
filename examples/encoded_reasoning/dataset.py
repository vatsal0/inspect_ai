import string
import requests
import os
import json
import random

from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

LOREM_IPSUM_WORDS = """lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum""".split()

src_dataset = load_dataset('openai/gsm8k', 'main', split='train')
chars = string.ascii_letters + string.digits + string.punctuation + " "

# Load arithmetic dataset from JSONL file
def load_arithmetic_dataset():
    arithmetic_path = "/nas/ucb/vatsalb/no_cot_math_public/arithmetic_problems.jsonl"
    with open(arithmetic_path, 'r') as f:
        return [json.loads(line) for line in f]

arithmetic_dataset = load_arithmetic_dataset()

def shorten(question, **kwargs):
    response = requests.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, data=json.dumps({
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You will be given a question. You need to rewrite the question concisely so that it is only half as long. Make sure the question still contains all relevant information needed to solve the question. Specifically, make sure all numbers remain in the question. Output only the rewritten, shortened question."},
            {"role": "user", "content": question}
        ],
        "reasoning": {"max_tokens": 1024},
    }))
    if response.status_code != 200: return str()
    return json.loads(response.text)["choices"][0]["message"]["content"]

def lengthen(question, **kwargs):
    try:
        response = requests.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, data=json.dumps({
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You will be given a question. You need to rewrite the question verbosely so that it is twice as long. Make sure the question still contains all relevant information needed to solve the question. Specifically, make sure all numbers remain in the question. Output only the rewritten, lengthened question."},
                {"role": "user", "content": question}
            ],
            "reasoning": {"max_tokens": 1024},
        }))
        if response.status_code != 200: return str()
        return json.loads(response.text)["choices"][0]["message"]["content"]
    except:
        print('lengthen error')
        return question

# from examples.encoded_reasoning.dataset import reason; reason('Evaluate this Python expression. (((-50 - 8) + 71) + -2) // (58 - (4 % 41))')
def reason(question, **kwargs):
    response = requests.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, data=json.dumps({
        "model": "anthropic/claude-opus-4.5",
        "messages": [
            {"role": "system", "content": "You will be given a math problem to solve. Answer using the format 'Answer: [ANSWER]' where [ANSWER] is just the numerical answer, nothing else."},
            # {"role": "system", "content": "You will be given a math problem. Answer immediately using the format 'Answer: [ANSWER]' where [ANSWER] is just the numerical answer, nothing else. No explanation, no words, no reasoning, just the number."},
            {"role": "user", "content": question}
        ],
        # "reasoning": {"effort": "none"},
    }))
    if response.status_code != 200: return str()
    return json.loads(response.text)["choices"][0]["message"]["content"]
    # return json.loads(response.text)["choices"][0]['message']['reasoning_details'][0]['text']

def add_random_tokens(question, **kwargs):
    random_tokens = "".join(random.choice(chars) for _ in range(kwargs.pop('filler_tokens', 50)))
    return question + "\n" + random_tokens

def add_blank_tokens(question, **kwargs):
    blank_tokens = " " * kwargs.pop('filler_tokens', 50)
    return question + "\n" + blank_tokens

def replace_random_tokens(question, **kwargs):
    n_tokens = kwargs.pop('n_tokens', len(question))
    random_tokens = "".join(random.choice(chars) for _ in range(n_tokens))
    return random_tokens

def replace_blank_tokens(question, **kwargs):
    n_tokens = kwargs.pop('n_tokens', len(question))
    blank_tokens = " " * n_tokens
    return blank_tokens

def add_number_tokens(question, **kwargs):
    filler_tokens = kwargs.pop('filler_tokens', 50)
    filler = " ".join(str(i) for i in range(1, filler_tokens + 1))
    return question + "\n" + filler

def add_ellipses(question, **kwargs):
    filler_tokens = kwargs.pop('filler_tokens', 50)
    filler = " ".join('...' for i in range(1, filler_tokens + 1))
    return question + "\n" + filler

def add_pause_tokens(question, **kwargs):
    filler_tokens = kwargs.pop('filler_tokens', 50)
    filler = " ".join('pause' for i in range(1, filler_tokens + 1))
    return question + "\n" + filler

def add_pause_tokens_before(question, **kwargs):
    filler_tokens = kwargs.pop('filler_tokens', 50)
    filler = " ".join('pause' for i in range(1, filler_tokens + 1))
    return filler + "\n" + question

def add_lorem_ipsum_tokens(question, **kwargs):
    filler_tokens = kwargs.pop('filler_tokens', 50)
    filler = " ".join(LOREM_IPSUM_WORDS[(i - 1) % len(LOREM_IPSUM_WORDS)] for i in range(1, filler_tokens + 1))
    return question + "\n" + filler

def add_fibonacci_tokens(question, **kwargs):
    filler_tokens = kwargs.pop('filler_tokens', 50)
    fibs = [0, 1]
    for _ in range(2, filler_tokens):
        fibs.append(fibs[-1] + fibs[-2])
    filler = " ".join(str(fibs[i-1]) for i in range(1, filler_tokens + 1))
    return question + "\n" + filler

question_transforms = {
    'shorten': shorten,
    'lengthen': lengthen,
    'add_random': add_random_tokens,
    'add_blank': add_blank_tokens,
    'replace_random': replace_random_tokens,
    'replace_blank': replace_blank_tokens,
    'add_numbers': add_number_tokens,
    'add_ellipses': add_ellipses,
    'add_pause': add_pause_tokens,
    'add_pause_before': add_pause_tokens_before,
    'add_lorem_ipsum': add_lorem_ipsum_tokens,
    'add_fibonacci': add_fibonacci_tokens,
}

def custom_dataset(task_name: str, q1_transform: str, q2_transform: str, two_questions: bool = False, N: int = 5000, **task_kwargs):
    random.seed(12345)
    if task_name == 'gsm8k':
        if two_questions:
            examples1 = src_dataset.select([i%N for i in range(0, N*2, 2)])
            examples2 = src_dataset.select([i%N for i in range(1, N*2, 2)])

            questions1 = [example['question'] for example in examples1]
            answers1 = [example['answer'].split('####')[-1].strip() for example in examples1]

            questions2 = [example['question'] for example in examples2]
            answers2 = [example['answer'].split('####')[-1].strip() for example in examples2]
        else:
            examples1 = src_dataset.select(range(0, N))
            questions1 = [example['question'] for example in examples1]
            answers1 = [example['answer'].split('####')[-1].strip() for example in examples1]
    if task_name == 'addition':
        n = task_kwargs.pop('n')

        examples1 = []
        for _ in range(N):
            a = random.randint(10 ** (n-1), 5 * 10 ** (n-1) - 1)
            b = random.randint(10 ** (n-1), 5 * 10 ** (n-1) - 1)
            examples1.append({'question': f'What is {a}+{b}?', 'answer': str(a+b)})
        questions1 = [example['question'] for example in examples1]
        answers1 = [example['answer'] for example in examples1]

        if two_questions:
            examples2 = []
            for _ in range(N):
                a = random.randint(10 ** (n-1), 5 * 10 ** (n-1) - 1)
                b = random.randint(10 ** (n-1), 5 * 10 ** (n-1) - 1)
                examples2.append({'question': f'What is {a}+{b}?', 'answer': str(a+b)})
            questions2 = [example['question'] for example in examples2]
            answers2 = [example['answer'] for example in examples2]

    if task_name == 'multiplication':
        n = task_kwargs.pop('n')

        examples1 = []
        for _ in range(N):
            a = random.randint(10 ** (n-1), 10 ** n - 1)
            b = random.randint(10 ** (n-1), 10 ** n - 1)
            examples1.append({'question': f'What is {a} times {b}?', 'answer': str(a*b)})
        questions1 = [example['question'] for example in examples1]
        answers1 = [example['answer'] for example in examples1]

        if two_questions:
            examples2 = []
            for _ in range(N):
                a = random.randint(10 ** (n-1), 10 ** n - 1)
                b = random.randint(10 ** (n-1), 10 ** n - 1)
                examples2.append({'question': f'What is {a} times {b}?', 'answer': str(a*b)})
            questions2 = [example['question'] for example in examples2]
            answers2 = [example['answer'] for example in examples2]

    if task_name == 'arithmetic':
        if two_questions:
            separator = 2
            indices1 = [i % len(arithmetic_dataset) for i in range(0, N * 2, separator)]
            indices2 = [i % len(arithmetic_dataset) for i in range(1, N * 2, separator)]

            questions1 = [arithmetic_dataset[i]['problem'] for i in indices1]
            answers1 = [str(arithmetic_dataset[i]['answer']) for i in indices1]

            questions2 = [arithmetic_dataset[i]['problem'] for i in indices2]
            answers2 = [str(arithmetic_dataset[i]['answer']) for i in indices2]
        else:
            separator = 1
            indices1 = [i % len(arithmetic_dataset) for i in range(0, N, separator)]

            questions1 = [arithmetic_dataset[i]['problem'] for i in indices1]
            answers1 = [str(arithmetic_dataset[i]['answer']) for i in indices1]

    if q1_transform in question_transforms:
        transform = question_transforms[q1_transform]
        with ThreadPoolExecutor() as executor:
            questions1 = list(tqdm(executor.map(lambda question: transform(question, **task_kwargs), questions1), total=len(questions1)))
    
    if two_questions and q2_transform in question_transforms:
        transform = question_transforms[q2_transform]
        with ThreadPoolExecutor() as executor:
            questions2 = list(tqdm(executor.map(lambda question: transform(question, **task_kwargs), questions2), total=len(questions2)))

    if two_questions:
        return MemoryDataset([
            Sample(
                input=f'''Question 1:\n{question1}\n\nQuestion 2:\n{question2}''',
                target=answer1 + '\n' + answer2,
            )
            for question1, answer1, question2, answer2 in zip(questions1, answers1, questions2, answers2)
        ])
    else:
        return MemoryDataset([
            Sample(
                input=question,
                target=answer,
            )
            for question, answer in zip(questions1, answers1)
        ])

def add_fewshot_examples(
    dataset: MemoryDataset,
    examples: list[tuple[str, str]],
    answer_format: str = "Answer: {answer}",
) -> MemoryDataset:
    """Add few-shot examples to each sample in a dataset as user/assistant message pairs.

    Args:
        dataset: The original dataset
        examples: List of (question, answer) tuples for few-shot demonstrations
        answer_format: Format string for the assistant's answer (use {answer} as placeholder)

    Returns:
        New dataset with few-shot examples prepended to each sample's input
    """
    new_samples = []
    for sample in dataset:
        # Build few-shot messages
        fewshot_messages = []
        for q, a in examples:
            fewshot_messages.append(ChatMessageUser(content=q))
            fewshot_messages.append(ChatMessageAssistant(content=answer_format.format(answer=a)))

        # Add the actual question
        if isinstance(sample.input, str):
            fewshot_messages.append(ChatMessageUser(content=sample.input))
        else:
            fewshot_messages.extend(sample.input)

        new_samples.append(Sample(
            input=fewshot_messages,
            target=sample.target,
            id=sample.id,
            metadata=sample.metadata,
        ))

    return MemoryDataset(new_samples, name=dataset.name)