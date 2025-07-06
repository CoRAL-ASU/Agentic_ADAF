

# Adaptive Chain-of-Table Reasoning with Open-Source LLMs

## Directory structure:

```
adaptive-table-qa/
├── agents/
│   ├── table_agent.py
│   ├── context_agent.py
│   ├── calculation_agent.py
│   └── coordinator.py
├── data/
│   ├── tatqa/
│   ├── finqa/
│   └── tabfact/
├── prompts/
│   ├── chain_templates.md
│   └── demo_examples.json
├── scripts/
│   ├── lora_finetune.py
│   ├── evaluate.py
│   └── generate_synthetic_data.py
├── utils/
│   ├── table_ops.py
│   ├── metrics.py
│   └── data_loader.py
├── configs/
│   └── model_config.yaml
├── models/
│   └── align_llm.py
├── main.py
├── README.md
└── requirements.txt
```



# Adaptive Chain-of-Table QA

This repository implements a multi-agent reasoning framework to perform multi-hop question answering over tables (and optionally text) using open-source LLMs like LLaMA and Mistral.

## Features
- Modular agents: TableAgent, ContextAgent, CalculationAgent, Coordinator
- Chain-of-Table reasoning steps
- Few-shot prompt templates
- Finetuning with LoRA
- Evaluation on FinQA, TabFact, TAT-QA, WikiTQ, FeTaQA

## Setup
```bash
pip install -r requirements.txt
```

## Run Inference
```bash
python main.py --dataset tatqa --llm mistral-7b
```

## Fine-tune
```bash
python scripts/lora_finetune.py
```
