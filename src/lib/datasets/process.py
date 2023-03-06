import torch
import random

def process_caption(tokenizer, tokens, grounded_attention, non_grounded_attention, train=True):
    output_tokens = []
    deleted_idx = []
    target_grounded_attention = [[] for idx in range(len(grounded_attention))]
    target_non_grounded_attention = []

    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()

        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% randomly change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
                    # target_grounded_attention.append(0)
                    for idx in range(len(grounded_attention)):
                        target_grounded_attention[idx].append(0)
                    target_non_grounded_attention.append(0)
            # 10% randomly change token to random token
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
                    # target_grounded_attention.append(0)
                    for idx in range(len(grounded_attention)):
                        target_grounded_attention[idx].append(0)
                    target_non_grounded_attention.append(0)
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    deleted_idx.append(len(output_tokens) - 1)
                    # target_grounded_attention.append(grounded_attention[i])
                    for idx in range(len(grounded_attention)):
                        target_grounded_attention[idx].append(grounded_attention[idx][i])
                    target_non_grounded_attention.append(non_grounded_attention[i])
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)
                # target_grounded_attention.append(grounded_attention[i])
                for idx in range(len(grounded_attention)):
                    target_grounded_attention[idx].append(grounded_attention[idx][i])
                target_non_grounded_attention.append(non_grounded_attention[i])

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]
        for idx in range(len(target_grounded_attention)):
            target_grounded_attention[idx] = [target_grounded_attention[idx][i] for i in range(len(target_grounded_attention[idx])) if i not in deleted_idx]
        target_non_grounded_attention = [target_non_grounded_attention[i] for i in
                                         range(len(target_non_grounded_attention)) if
                                         i not in deleted_idx]

    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    for idx in range(len(target_grounded_attention)):
        target_grounded_attention[idx] = [0] + target_grounded_attention[idx] + [0]
    target_non_grounded_attention = [0] + target_non_grounded_attention + [0]
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    target = torch.Tensor(target)
    target_grounded_attention = torch.Tensor(target_grounded_attention)
    target_non_grounded_attention = torch.Tensor(target_non_grounded_attention)
    return target, target_grounded_attention, target_non_grounded_attention


def process_goal(tokenizer, task_tokens, method_tokens, train=True):
    task_output_tokens = []
    task_deleted_idx = []

    method_output_tokens = []
    method_deleted_idx = []

    for i, token in enumerate(task_tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()

        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% randomly change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    task_output_tokens.append("[MASK]")
            # 10% randomly change token to random token
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    task_output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
            else:
                for sub_token in sub_tokens:
                    task_output_tokens.append(sub_token)
                    task_deleted_idx.append(len(task_output_tokens) - 1)
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                task_output_tokens.append(sub_token)

    if len(task_deleted_idx) != 0:
        task_output_tokens = [task_output_tokens[i] for i in range(len(task_output_tokens)) if i not in task_deleted_idx]

    for i, token in enumerate(method_tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()

        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% randomly change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    method_output_tokens.append("[MASK]")
            # 10% randomly change token to random token
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    method_output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
            else:
                for sub_token in sub_tokens:
                    method_output_tokens.append(sub_token)
                    method_deleted_idx.append(len(method_output_tokens) - 1)
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                method_output_tokens.append(sub_token)

    if len(method_deleted_idx) != 0:
        method_output_tokens = [method_output_tokens[i] for i in range(len(method_output_tokens)) if i not in method_deleted_idx]

    output_tokens = ['[CLS]'] + task_output_tokens + ["("] + method_output_tokens + [")"] + ['[SEP]']
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    target = torch.Tensor(target)
    return target