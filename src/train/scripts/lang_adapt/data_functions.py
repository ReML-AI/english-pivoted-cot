import ast
import os
import random

import datasets
import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from trl.trainer import ConstantLengthDataset

SYSTEM_PROMPT = ""


def chars_token_ratio(dataset, tokenizer, nb_examples=400, column="text"):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = example[column]
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def load_Bespoke_Stratos_17k(tokenizer):
    dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k")["train"]

    def apply_template(x):
        conv = x["conversations"]
        conv = [{"role": turn["from"], "content": turn["value"]} for turn in conv]
        x["text"] = tokenizer.apply_chat_template(
            [{"role": "system", "content": x["system"]}] + conv, tokenize=False
        )
        return x

    dataset = dataset.map(apply_template)
    dataset = dataset.remove_columns(["conversations", "system"])

    dataset = dataset.train_test_split(test_size=0.005, seed=None)
    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]

    return train_dataset, valid_dataset


def load_translated_bespoke_stratos_17k(tokenizer):
    df = pd.read_csv("data/bespoke_processed_merged.csv")
    texts = []

    for _, row in tqdm(df.iterrows()):
        formatted_conv = [{"role": "system", "content": SYSTEM_PROMPT}]
        formatted_conv.append({"role": "user", "content": row["user"]})
        formatted_conv.append({"role": "assistant", "content": row["assistant"]})
        text = tokenizer.apply_chat_template(formatted_conv, tokenize=False)
        texts.append(text)

    dataset = Dataset.from_dict({"text": texts}, split="train")
    dataset = dataset.train_test_split(test_size=0.005, seed=None)
    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]

    return train_dataset, valid_dataset


def load_translated_bespoke_stratos_17k_with_cot(tokenizer):
    df = pd.read_csv("data/bespoke/bespoke_processed_with_cot_merged.csv")
    texts = []

    for _, row in tqdm(df.iterrows()):
        formatted_conv = [{"role": "system", "content": SYSTEM_PROMPT}]
        formatted_conv.append({"role": "user", "content": row["user"]})
        formatted_conv.append({"role": "assistant", "content": row["assistant"]})
        if len(tokenizer.apply_chat_template(formatted_conv, tokenize=True)) <= 16_384:
            text = tokenizer.apply_chat_template(formatted_conv, tokenize=False)
            texts.append(text)

    dataset = Dataset.from_dict({"text": texts}, split="train")
    dataset = dataset.train_test_split(test_size=0.005, seed=None)
    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]

    return train_dataset, valid_dataset


def load_pensez(tokenizer):
    df = load_dataset("HoangHa/Pensez-v0.1-formatted")["train"].to_pandas()
    texts = []

    for _, row in tqdm(df.iterrows()):
        formatted_conv = [{"role": "system", "content": SYSTEM_PROMPT}]
        assistant_answer = row["messages"][1]["content"]
        if "\\boxed{" in assistant_answer:
            user_prompt = (
                "Please reason step by step, and put your final answer within \\boxed{}. "
                + row["messages"][0]["content"]
            )
        else:
            user_prompt = "Please reason step by step. " + row["messages"][0]["content"]
        formatted_conv.append({"role": "user", "content": user_prompt})
        formatted_conv.append({"role": "assistant", "content": assistant_answer})
        text = tokenizer.apply_chat_template(formatted_conv, tokenize=False)
        texts.append(text)

    dataset = Dataset.from_dict({"text": texts}, split="train")
    dataset = dataset.train_test_split(test_size=0.005, seed=None)
    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]

    return train_dataset, valid_dataset


def load_pensez_translated(tokenizer):
    df = pd.read_csv("data/pensez/pensez-v0.1-formatted-translated.csv")
    texts = []

    for _, row in tqdm(df.iterrows()):
        formatted_conv = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages = row["messages"]
        user_prompt = messages[0]["content"]
        assistant_answer = messages[1]["content"]
        if "\\boxed{" in assistant_answer:
            user_prompt = (
                "Please reason step by step, and put your final answer within \\boxed{}. "
                + user_prompt
            )
        else:
            user_prompt = "Please reason step by step. " + user_prompt
        formatted_conv.append({"role": "user", "content": user_prompt})
        formatted_conv.append({"role": "assistant", "content": assistant_answer})
        if len(tokenizer.apply_chat_template(formatted_conv, tokenize=True)) <= 16_384:
            text = tokenizer.apply_chat_template(formatted_conv, tokenize=False)
            texts.append(text)

    dataset = Dataset.from_dict({"text": texts}, split="train")
    dataset = dataset.train_test_split(test_size=0.005, seed=None)
    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]

    return train_dataset, valid_dataset


def load_congliu(tokenizer):
    dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k")["train"]

    def apply_template(x):
        conv = x["conversations"]
        if conv[0]["value"].startswith("Return your final response within \\boxed{}. "):
            conv[0]["value"] = (
                "Please reason step by step, and put your final answer within \\boxed{}. "
                + conv[0]["value"][
                    len("Return your final response within \\boxed{}. ") :
                ]
            )
        conv[1]["value"] = (
            conv[1]["value"]
            .replace("<|begin_of_thought|>", "<think>")
            .replace("<|end_of_thought|>", "</think>")
        )
        conv[1]["value"] = conv[1]["value"].replace("<|begin_of_solution|>", "")
        if "\n<|end_of_solution|>" in conv[1]["value"]:
            conv[1]["value"] = conv[1]["value"].replace("\n<|end_of_solution|>", "")
        else:
            conv[1]["value"] = conv[1]["value"].replace("<|end_of_solution|>", "")
        conv[1]["value"] = conv[1]["value"].strip()

        conv = [{"role": turn["from"], "content": turn["value"]} for turn in conv]
        x["text"] = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}] + conv, tokenize=False
        )
        return x

    dataset = dataset.map(apply_template)
    dataset = dataset.remove_columns(["conversations", "system"])

    df = load_dataset(
        "Congliu/Chinese-DeepSeek-R1-Distill-data-110k",
    )["train"].to_pandas()
    df = df[df["repo_name"] != "coig/neo"]
    curr_df_index = []
    for index in range(20):
        curr_df_index.extend([idx for idx in range(len(df)) if idx % 100 == index])
    df = df.iloc[curr_df_index, :]

    texts = []
    for _, row in tqdm(df.iterrows()):
        formatted_conv = [{"role": "system", "content": SYSTEM_PROMPT}]
        user_prompt = row["input"]
        assistant_answer = (
            "<think>\n\n"
            + row["reasoning_content"]
            + "\n\n</think>\n\n"
            + row["content"]
        )
        if "\\boxed{" in assistant_answer:
            user_prompt = (
                "Please reason step by step, and put your final answer within \\boxed{}. "
                + user_prompt
            )
        else:
            user_prompt = "Please reason step by step. " + user_prompt
        formatted_conv.append({"role": "user", "content": user_prompt})
        formatted_conv.append({"role": "assistant", "content": assistant_answer})
        text = tokenizer.apply_chat_template(formatted_conv, tokenize=False)
        texts.append(text)

    dataset_congliu = Dataset.from_dict({"text": texts}, split="train")
    dataset = datasets.concatenate_datasets([dataset, dataset_congliu])
    dataset = dataset.train_test_split(test_size=0.005, seed=None)
    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]

    return train_dataset, valid_dataset


def load_congliu_translated(tokenizer):
    dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k")["train"]

    def apply_template(x):
        conv = x["conversations"]
        if conv[0]["value"].startswith("Return your final response within \\boxed{}. "):
            conv[0]["value"] = (
                "Please reason step by step, and put your final answer within \\boxed{}. "
                + conv[0]["value"][
                    len("Return your final response within \\boxed{}. ") :
                ]
            )
        conv[1]["value"] = (
            conv[1]["value"]
            .replace("<|begin_of_thought|>", "<think>")
            .replace("<|end_of_thought|>", "</think>")
        )
        conv[1]["value"] = conv[1]["value"].replace("<|begin_of_solution|>", "")
        if "\n<|end_of_solution|>" in conv[1]["value"]:
            conv[1]["value"] = conv[1]["value"].replace("\n<|end_of_solution|>", "")
        else:
            conv[1]["value"] = conv[1]["value"].replace("<|end_of_solution|>", "")
        conv[1]["value"] = conv[1]["value"].strip()

        conv = [{"role": turn["from"], "content": turn["value"]} for turn in conv]
        x["text"] = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}] + conv, tokenize=False
        )
        return x

    dataset = dataset.map(apply_template)
    dataset = dataset.remove_columns(["conversations", "system"])

    df = pd.read_csv("data/congliu/translated_filtered_rep11_0_to_20_of_100.csv")

    texts = []
    for _, row in tqdm(df.iterrows()):
        formatted_conv = [{"role": "system", "content": SYSTEM_PROMPT}]
        user_prompt = row["input"]
        assistant_answer = (
            "<think>\n\n"
            + row["reasoning_content"]
            + "\n\n</think>\n\n"
            + row["content"]
        )
        if "\\boxed{" in assistant_answer:
            user_prompt = (
                "Please reason step by step, and put your final answer within \\boxed{}. "
                + user_prompt
            )
        else:
            user_prompt = "Please reason step by step. " + user_prompt
        formatted_conv.append({"role": "user", "content": user_prompt})
        formatted_conv.append({"role": "assistant", "content": assistant_answer})
        text = tokenizer.apply_chat_template(formatted_conv, tokenize=False)
        texts.append(text)

    dataset_congliu = Dataset.from_dict({"text": texts}, split="train")
    dataset = datasets.concatenate_datasets([dataset, dataset_congliu])
    dataset = dataset.train_test_split(test_size=0.005, seed=None)
    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]

    return train_dataset, valid_dataset


def create_datasets(tokenizer, conf):
    if conf.dataset_dir == "Bespoke-Stratos-17k":
        train_data, valid_data = load_Bespoke_Stratos_17k(tokenizer)
    elif conf.dataset_dir == "translated_bespoke_stratos_17k":
        train_data, valid_data = load_translated_bespoke_stratos_17k(tokenizer)
    elif conf.dataset_dir == "translated_bespoke_stratos_17k_with_cot":
        train_data, valid_data = load_translated_bespoke_stratos_17k_with_cot(tokenizer)
    elif conf.dataset_dir == "pensez":
        train_data, valid_data = load_pensez(tokenizer)
    elif conf.dataset_dir == "pensez_translated":
        train_data, valid_data = load_pensez_translated(tokenizer)
    elif conf.dataset_dir == "congliu":
        train_data, valid_data = load_congliu(tokenizer)
    elif conf.dataset_dir == "congliu_translated":
        train_data, valid_data = load_congliu_translated(tokenizer)

    train_data = train_data.shuffle(seed=None)
    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
    )

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    return train_data, valid_data


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    class Conf:
        dataset_dir: str
        block_size: int

    conf = Conf()
    conf.dataset_dir = "160625_mixture_v4"
    conf.block_size = 8192

    train_dataset, valid_dataset = create_datasets(tokenizer, conf)
    total_len = 0
    long_sample = 0
    print("Long samples: ", long_sample)
    print(total_len)
    chars_per_token = chars_token_ratio(train_dataset, tokenizer)
    print(total_len / (chars_per_token * 16384 * 2 * 8))
