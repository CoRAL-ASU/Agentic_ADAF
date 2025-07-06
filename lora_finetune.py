"""Minimal LoRA fine-tuning script.

This file only contains a skeleton implementation of a LoRA fine-tuning
pipeline.  The actual data loading logic is left as a TODO because this
repository does not ship with training data.
"""

from peft import get_peft_model, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def finetune(model_name: str = "meta-llama/Llama-2-7b-hf") -> None:
    """Fine-tune `model_name` with LoRA parameters."""

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
    )
    model = get_peft_model(model, config)

    # TODO: replace this placeholder with an actual Dataset implementation
    train_dataset = []  # type: ignore[var-annotated]

    args = TrainingArguments(
        output_dir="lora_finetune",
        per_device_train_batch_size=1,
        num_train_epochs=1,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
    trainer.train()


if __name__ == "__main__":
    finetune()
