
@task
def zero_shot_mul_1():
    return Task(
        dataset=multiplication_dataset(1),
        solver=[system_message(ZERO_SHOT_SYSTEM_PROMPT), generate()],
        scorer=xml_answer(['</reasoning>', '<answer>', '</answer>']),
    )

@task
def zero_shot_mul_2():
    return Task(
        dataset=multiplication_dataset(2),
        solver=[system_message(ZERO_SHOT_SYSTEM_PROMPT), generate()],
        scorer=xml_answer(['</reasoning>', '<answer>', '</answer>']),
    )

@task
def zero_shot_mul_4():
    return Task(
        dataset=multiplication_dataset(4),
        solver=[system_message(ZERO_SHOT_SYSTEM_PROMPT), generate()],
        scorer=xml_answer(['</reasoning>', '<answer>', '</answer>']),
    )

@task
def zero_shot_mul_5():
    return Task(
        dataset=multiplication_dataset(5),
        solver=[system_message(ZERO_SHOT_SYSTEM_PROMPT), generate()],
        scorer=xml_answer(['</reasoning>', '<answer>', '</answer>']),
    )

@task
def zero_shot_mul_6():
    return Task(
        dataset=multiplication_dataset(6),
        solver=[system_message(ZERO_SHOT_SYSTEM_PROMPT), generate()],
        scorer=xml_answer(['</reasoning>', '<answer>', '</answer>']),
    )

@task
def zero_shot_mul_7():
    return Task(
        dataset=multiplication_dataset(7),
        solver=[system_message(ZERO_SHOT_SYSTEM_PROMPT), generate()],
        scorer=xml_answer(['</reasoning>', '<answer>', '</answer>']),
    )

@task
def zero_shot_mul_8():
    return Task(
        dataset=multiplication_dataset(8),
        solver=[system_message(ZERO_SHOT_SYSTEM_PROMPT), generate()],
        scorer=xml_answer(['</reasoning>', '<answer>', '</answer>']),
    )

@task
def cot_mul_2():
    return Task(
        dataset=multiplication_dataset(2),
        solver=[system_message(SINGLE_COT_SYSTEM_PROMPT), generate()],
        scorer=xml_answer(['</reasoning>', '<answer>', '</answer>']),
    )

@task
def zero_shot():
    return Task(
        dataset=multiplication_dataset(4),
        solver=[system_message(ZERO_SHOT_SYSTEM_PROMPT), generate()],
        scorer=xml_answer(['</reasoning>', '<answer>', '</answer>']),
    )


@task
def double_zero_shot():
    return Task(
        dataset=double_question_dataset,
        solver=[double_zero_solver()],
        scorer=xml_answer(['</reasoning>', '</answer1>', '<answer2>', '</answer2>']),
    )

@task
def single_cot():
    return Task(
        dataset=single_question_dataset,
        solver=[system_message(SINGLE_COT_SYSTEM_PROMPT), generate()],
        scorer=xml_answer(['</reasoning>', '<answer>', '</answer>']),
    )

@task
def blind():
    return Task(
        dataset=double_question_dataset,
        solver=[
            blind_solver()
            ],
        scorer=xml_answer(['</reasoning>', '</answer1>', '<answer2>', '</answer2>']),
    )


@task
def blind_trace_before():
    return Task(
        dataset=double_question_dataset,
        solver=[blind_general_solver(trace_before=True)],
        scorer=xml_answer(['</reasoning>', '</answer1>', '<answer2>', '</answer2>']),
    )
@task
def blind_trace_after():
    return Task(
        dataset=double_question_dataset,
        solver=[blind_general_solver(trace_before=False)],
        scorer=xml_answer(['</reasoning>', '</answer1>', '<answer2>', '</answer2>']),
    )

@task
def encoded():
    return Task(
        dataset=multiplication_dataset(5),
        solver=[system_message(ENCODED_SYSTEM_PROMPT), generate()],
        scorer=xml_answer(['</reasoning>', '</answer1>', '<answer2>', '</answer2>']),
    )

@task
def encoded_hint():
    return Task(
        dataset=double_question_dataset,
        solver=[system_message(ENCODED_SYSTEM_PROMPT_HINT), generate()],
        scorer=xml_answer(['</reasoning>', '</answer1>', '<answer2>', '</answer2>']),
    )

@task
def pause():
    return Task(
        dataset=single_question_dataset,
        solver=[system_message(PAUSE_SYSTEM_PROMPT), generate()],
        scorer=xml_answer(['</reasoning>', '<answer>', '</answer>'])
    )

@task
def pause_fixed():
    return Task(
        dataset=single_question_dataset,
        solver=[system_message(PAUSE_SYSTEM_PROMPT_FIXED), generate()],
        scorer=xml_answer(['</reasoning>', '<answer>', '</answer>'])
    )

@task
def pause_hint():
    return Task(
        dataset=single_question_dataset,
        solver=[system_message(PAUSE_SYSTEM_PROMPT_HINT), generate()],
        scorer=xml_answer(['</reasoning>', '<answer>', '</answer>'])
    )

@task
def double_blind():
    return Task(
        dataset=double_question_dataset,
        solver=[double_blind_solver()],
        scorer=xml_answer(['</reasoning1>', '</reasoning2>', '</answer1>', '<answer2>', '</answer2>']),
    )

@task
def double_blind_first5():
    return Task(
        dataset=double_question_dataset,
        solver=[double_blind_solver(q1_transform='first_k', k=5)],
        scorer=xml_answer(['</reasoning1>', '</reasoning2>', '</answer1>', '<answer2>', '</answer2>']),
    )

@task
def double_blind_last2():
    return Task(
        dataset=double_question_dataset,
        solver=[double_blind_solver(q1_transform='last_k', k=2)],
        scorer=xml_answer(['</reasoning1>', '</reasoning2>', '</answer1>', '<answer2>', '</answer2>']),
    )

@task
def double_blind_shuffle():
    return Task(
        dataset=double_question_dataset,
        solver=[double_blind_solver(q1_transform='shuffle')],
        scorer=xml_answer(['</reasoning1>', '</reasoning2>', '</answer1>', '<answer2>', '</answer2>']),
    )

@task
def double_blind_random():
    return Task(
        dataset=double_question_dataset,
        solver=[double_blind_solver(q1_transform='random')],
        scorer=xml_answer(['</reasoning1>', '</reasoning2>', '</answer1>', '<answer2>', '</answer2>']),
    )

@task
def double_blind_paraphrase():
    return Task(
        dataset=double_question_dataset,
        solver=[double_blind_solver(q1_transform='paraphrase')],
        scorer=xml_answer(['</reasoning1>', '</reasoning2>', '</answer1>', '<answer2>', '</answer2>']),
    )

@task
def double_blind_delete():
    return Task(
        dataset=double_question_dataset,
        solver=[double_blind_solver(q1_transform='delete')],
        scorer=xml_answer(['</reasoning1>', '</reasoning2>', '</answer1>', '<answer2>', '</answer2>']),
    )

@task
def double_blind_addq2():
    return Task(
        dataset=double_question_dataset,
        solver=[double_blind_solver(q1_transform='addq2')],
        scorer=xml_answer(['</reasoning1>', '</reasoning2>', '</answer1>', '<answer2>', '</answer2>']),
    )

@task
def double_blind_last1_2():
    return Task(
        dataset=double_question_dataset,
        solver=[double_blind_solver(q2_transform='last_k', k=1)],
        scorer=xml_answer(['</reasoning1>', '</reasoning2>', '</answer1>', '<answer2>', '</answer2>']),
    )

@task
def double_blind_shuffle_2():
    return Task(
        dataset=double_question_dataset,
        solver=[double_blind_solver(q2_transform='shuffle')],
        scorer=xml_answer(['</reasoning1>', '</reasoning2>', '</answer1>', '<answer2>', '</answer2>']),
    )

@task
def double_cot():
    return Task(
        dataset=double_question_dataset,
        solver=[double_blind_solver(q1_transform='', q2_transform='')],
        scorer=xml_answer(['</reasoning1>', '</reasoning2>', '</answer1>', '<answer2>', '</answer2>']),
    )

@task
def double_cot_justq1():
    return Task(
        dataset=double_question_dataset,
        solver=[double_blind_solver(q1_transform='delete', q2_transform='')],
        scorer=xml_answer(['</reasoning1>', '</reasoning2>', '</answer1>', '<answer2>', '</answer2>']),
    )