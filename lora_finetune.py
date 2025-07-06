"""LoRA fine-tuning script for the TAT-QA dataset using Mistral."""

from __future__ import annotations

import argparse

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from utils.tatqa_dataset import TatqaDataset


def finetune(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    output_dir: str = "lora_mistral",
    train_split: str = "train",
    eval_split: str = "dev",
) -> None:
    """Fine-tune `model_name` with LoRA parameters on the TAT-QA dataset."""

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
    )
    model = get_peft_model(model, config)

    train_dataset = TatqaDataset(tokenizer, split=train_split)
    eval_dataset = TatqaDataset(tokenizer, split=eval_split)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output", type=str, default="lora_mistral")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="dev")
    args = parser.parse_args()

    finetune(
        model_name=args.model,
        output_dir=args.output,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
