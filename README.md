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
├── lora_finetune.py
├── evaluate.py
├── scripts/
│   └── generate_synthetic_data.py
├── utils/
│   ├── table_ops.py
@@ -40,26 +39,26 @@ adaptive-table-qa/
```

# Adaptive Chain-of-Table QA
```
This repository implements a multi-agent reasoning framework to perform multi-hop question answering over tables (and optionally text) using open-source LLMs like LLaMA and Mistral.
```
## Features
```
- Modular agents: TableAgent, ContextAgent, CalculationAgent, Coordinator
- Chain-of-Table reasoning steps
- Few-shot prompt templates
- Finetuning with LoRA
- Evaluation on FinQA, TabFact, TAT-QA, WikiTQ, FeTaQA
```
## Setup
```bash
pip install -r requirements.txt
```

### Prepare Mistral model
Clone the official inference repository and install its Python package:
```bash
scripts/setup_mistral_inference.sh
```

### Use OpenAI API
Install the `openai` package (already listed in `requirements.txt`) and set
your API key:
```bash
export OPENAI_API_KEY=<your-key>
```
Run the pipeline with any supported OpenAI model, e.g. `gpt-3.5-turbo`:
```bash
python main.py --dataset tatqa --llm gpt-3.5-turbo
```


## Run Inference
```bash
python main.py --dataset tatqa --llm mistral-7b
```

## Fine-tune
```bash
python lora_finetune.py --model mistralai/Mistral-7B-v0.1
```

## Evaluate
```bash
python evaluate.py lora_mistral --split dev
```
