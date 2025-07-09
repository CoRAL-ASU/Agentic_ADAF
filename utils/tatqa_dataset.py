import json
from typing import List

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from utils.data_loader import load_benchmark


class TatqaDataset(Dataset):
    """Dataset wrapper that tokenizes TAT-QA examples for language model fine-tuning."""

    def __init__(self, tokenizer: PreTrainedTokenizer, split: str = "train", max_length: int = 512) -> None:
        self.samples: List[dict] = load_benchmark("tatqa", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        example = self.samples[idx]
        question = example.get("question", "")
        table = example.get("table")
        paragraphs = example.get("paragraphs", [])
        answer = str(example.get("answer", ""))

        prompt = f"Question: {question}\n"
        if table:
            prompt += f"Table: {json.dumps(table)}\n"
        if paragraphs:
            prompt += f"Context: {' '.join(paragraphs)}\n"
        prompt += "Answer:"

        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_tokens = self.tokenizer.encode(answer + self.tokenizer.eos_token, add_special_tokens=False)

        input_ids = prompt_tokens + answer_tokens
        labels = [-100] * len(prompt_tokens) + answer_tokens

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len

        return {"input_ids": input_ids, "labels": labels}
