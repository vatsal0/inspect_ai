from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def find_subsequence(seq, subseq):
    """Return list of start indices where subseq occurs in seq"""
    matches = []
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i:i+m] == subseq:
            matches.append(i)
    return matches

model_name = "vatsalb/length_penalty_4b"
model_name = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, 
    device_map="auto",
    attn_implementation="eager",
)
model.eval()

from datasets import load_dataset
src_dataset = load_dataset('openai/gsm8k', 'main', split='train')

from inspect_ai.dataset import MemoryDataset, Sample

N = len(src_dataset)

DOUBLE_COT_SYSTEM_PROMPT = '''/no_think
Answer the provided questions after providing a few sentences of reasoning for each question. Your answer must only contain a single integer with no additional symbols or phrases. Do not output anything other than a single integer between the <answer> and </answer> tags.
Respond in the following format:
<reasoning1>
...
</reasoning1>

<answer1>
...
</answer1>

<reasoning2>
...
</reasoning2>

<answer2>
...
</answer2>
'''

double_question_dataset = MemoryDataset([
    Sample(
        input=f'''Question 1:\n{example1['question']}\n\nQuestion 2:\n{example2['question']}''',
        target=example2['answer'].split('####')[-1].strip(),
    )
    for example1, example2 in zip(src_dataset.select([i%N for i in range(0, N*2, 2)]), src_dataset.select([i%N for i in range(1, N*2, 2)]))
])

def get_answer_prob(probs, target_tokens, answer_pos):
    answer_prob = 1
    target_pos = 0

    while target_pos < len(target_tokens):
        if answer_pos > probs.shape[1]:
            answer_prob = 0
            break

        token_prob = probs[0, answer_pos - 1, target_tokens[target_pos]]
        answer_prob *= token_prob

        target_pos += 1
        answer_pos += 1
    return answer_prob

deltas = []

for i in range(20):
    prompt = [
        {"role": "system", "content": DOUBLE_COT_SYSTEM_PROMPT},
        {"role": "user", "content": double_question_dataset[0].input},
    ]

    text = tokenizer.apply_chat_template(prompt, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generated_output = model.generate(**inputs, max_new_tokens=256)
    completion = generated_output[0][len(inputs['input_ids'][0]):]
    response = tokenizer.decode(completion, skip_special_tokens=True)

    prompt.append({"role": "assistant", "content": response.split('</think>')[-1].split('<answer2>')[0] + '<answer2>\n'})
    continue_text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True
    )
    continue_inputs = tokenizer(continue_text, return_tensors="pt").to(model.device)
    tokens = tokenizer.convert_ids_to_tokens(continue_inputs["input_ids"][0])
    q1_start = find_subsequence(tokens, ['<', 'reason', 'ing', '1'])[-1]
    q1_end = find_subsequence(tokens, ['</', 'reason', 'ing', '1'])[-1] + 5

    generated_answer = model.generate(**continue_inputs)
    orig_probs = model(generated_answer, use_cache=False).logits.softmax(dim=-1)
    attn_mask = torch.ones_like(generated_answer).to(model.device)
    attn_mask[:, q1_start:q1_end] = 0
    mask_probs = model(generated_answer, attention_mask=attn_mask, use_cache=False).logits.softmax(dim=-1)

    target_tokens = tokenizer.encode(double_question_dataset[0].target)

    answer_pos = len(continue_inputs['input_ids'][0])

    answer_completion = generated_answer[0][answer_pos:]
    answer = tokenizer.decode(answer_completion)

    deltas.append(get_answer_prob(orig_probs, target_tokens, answer_pos).item()
        - get_answer_prob(mask_probs, target_tokens, answer_pos).item())

print(deltas)
