# Adaptive Chain-of-Table Reasoning with OpenAI LLMs
![Architecture](Decentralized_Agent_V2.png)
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
This repository implements a multi-agent reasoning framework to perform multi-hop question answering over tables (and optionally text) using OpenAI LLMs like `gpt-3.5-turbo`.
```
## Features
```
- Modular agents: TableAgent, ContextAgent, CalculationAgent, Coordinator
- Chain-of-Table reasoning steps
- Few-shot prompt templates
- Finetuning with LoRA
- Evaluation on FinQA, TabFact, TAT-QA, WikiTQ, FeTaQA
```

## Architecture Overview

The system follows a planner-free, log-mediated question answering workflow in which
multiple specialist agents collaborate through a shared append-only log. Each agent
reads prior entries, writes new observations, and hands off intermediate results to
other agents for synthesis and verification. A high-level coordinator ensures turn
taking while a summarizing verifier validates the final answer before it is returned
to the user.

For a detailed breakdown of the components and their interactions, see the
[architecture poster](docs/architecture_poster.md). The concrete log schema,
message types, and agent contracts live in the new
[shared log schema reference](docs/shared_log_schema.md).

## Datasets

- `tatqa` – Original Tabular And Text QA benchmark (default).
- `crtqa` / `crt-qa` – Compliance Readiness Tables QA. Compact dataset with curated CRT adoption tables and contextual passages.
- `multi_hop` / `multi-hop` – Synthetic 5–8 step operator chains for stressing arithmetic/multi-hop coordination.
- `finqa`, `mmqa_full`, `mmqa_text_table`, `wikitq`, `fetaqa` – drop JSON/JSONL splits under `data/<DatasetName>/<split>.json`. The loader ingests flat lists of QA samples, so you can preprocess HuggingFace exports or your own converters without changing code.
- For quick smoke tests, use `--limit 20` (or set `sample_limit` in `configs/planner_benchmarks.yaml`) to cap runs at twenty examples per dataset.
- For the FeTaQA factuality experiment (QAGS/BERTScore/Log-Groundedness + 100-example human eval), follow the playbook in [`docs/fetaqa_faithfulness.md`](docs/fetaqa_faithfulness.md). That note also contains the new table to insert after Table 2 in the paper.

Choose a dataset via `--dataset` when running `main.py`, `scripts/run_dealog.py`, or the benchmarking harness.

## Setup

### Requirements

- Python 3.10+.
- A CUDA-capable GPU is recommended for local model inference and for the larger evaluation runs.
- One of the following model backends:
  - `OPENAI_API_KEY` for OpenAI-hosted models.
  - `OPENROUTER_API_KEY` for OpenRouter-hosted models.
  - Local HuggingFace checkpoints for offline or self-hosted inference.

Install the Python dependencies in a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you plan to use the Mistral local backend, install the companion inference package as well:

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


```bash
OPENAI_API_KEY=<your-key>
PRIMARY_MODEL_NAME=<model-id>
DEALOG_SUMMARIZER_MODEL=<model-id>
```

Minimal OpenRouter configuration:

```bash
OPENROUTER_API_KEY=<your-key>
PRIMARY_MODEL_NAME=<model-id>
DEALOG_SUMMARIZER_MODEL=<model-id>
```

Minimal local-checkpoint configuration:

```bash
DEALOG_LLM_BACKEND=local
PRIMARY_MODEL_PATH=/path/to/models--org--name
DEALOG_SUMMARIZER_MODEL_PATH=/path/to/models--org--name
PRIMARY_MODEL_NAME=<label-for-outputs>
DEALOG_SUMMARIZER_MODEL=<label-for-outputs>
```

Common optional variables:

- `DEALOG_CUDA_VISIBLE_DEVICES=0,1` to pin evaluation jobs to selected GPUs.
- `VISUAL_CAPTION_MODEL` and `VISUAL_CAPTION_MODEL_PATH` for BLIP-2 captioning.
- `VISUAL_OCR_ENGINE` and `VISUAL_OCR_MODEL_DIR` for OCR-backed visual parsing.
- `HF_API_TOKEN` for gated HuggingFace downloads.
- `TMPDIR=/path/to/tmp` if your cluster requires a custom temporary directory.

If `PRIMARY_MODEL_PATH` or `DEALOG_SUMMARIZER_MODEL_PATH` points to a HuggingFace cache root of the form `models--org--name`, the loader automatically resolves the newest checkpoint under `snapshots/`.

### Data layout

The benchmark loader expects dataset files under `data/<DatasetName>/`. The repository already includes the CRT-QA and synthetic multi-hop resources used by the paper-style long-horizon experiments:

- `data/CRTQA/crtqa_{train,dev,test}.json`
- `data/multi_hop_synthetic/multi_hop_{train,dev,test}.json`

Additional datasets should be placed as flat JSON or JSONL splits under the corresponding directory, for example:

- `data/TATQA/tatqa_dataset_dev.json`
- `data/FinQA/dev.json`
- `data/WikiTQ/dev.json`
- `data/FeTaQA/dev.json`

## Running DeALoG

For a quick end-to-end smoke test, run the main entry point on a small slice of a dataset:

```bash
python main.py \
  --dataset crtqa \
  --split dev \
  --llm ${PRIMARY_MODEL_NAME} \
  --limit 20
```

This command prints per-example predictions to stdout and is useful for confirming that model credentials, dataset loading, and agent coordination are functioning correctly.

For a full evaluation run that writes machine-readable metrics and per-example traces, use:

```bash
python scripts/run_dealog.py \
  --dataset crtqa \
  --split dev \
  --llm ${PRIMARY_MODEL_NAME} \
  --summarizer-llm ${DEALOG_SUMMARIZER_MODEL} \
  --results-file benchmarks/results/crtqa_dev_dealog.json
```

Important optional flags:

- `--limit 20` for a smoke test.
- `--max-rounds 10` to match the long-horizon setting used in the paper-facing experiments.
- `--min-chain-len` and `--max-chain-len` to evaluate only selected multi-hop subsets.
- `--parallel-retrieval` to enable the retrieval micro-benchmark mode.

The JSON output contains aggregate metrics (`accuracy`, `latency_sec`, `num_examples`) and a `per_example` list with predictions, references, rationales, and shared-log traces.

## Evaluation

### Reproducing a single dataset result

The simplest paper-style evaluation path is `scripts/run_dealog.py`. For example, to evaluate the synthetic long-horizon subset with 7-8 operator steps:

```bash
python scripts/run_dealog.py \
  --dataset multi_hop \
  --split dev \
  --llm ${PRIMARY_MODEL_NAME} \
  --summarizer-llm ${DEALOG_SUMMARIZER_MODEL} \
  --min-chain-len 7 \
  --max-chain-len 8 \
  --max-rounds 10 \
  --results-file benchmarks/results/multihop_7_8_dev.json
```

### Reproducing the long-horizon DeALoG table

To reproduce the CRT-QA and Multi-Hop rows together, along with the `8192`-token summarizer ablation, run:

```bash
python scripts/run_table6_long_horizon.py \
  --llm ${PRIMARY_MODEL_NAME} \
  --summarizer-llm ${DEALOG_SUMMARIZER_MODEL} \
  --split dev \
  --base-max-tokens 256 \
  --ablation-max-tokens 8192 \
  --max-rounds 10 \
  --output-dir benchmarks/results/table6_long_horizon
```

This script writes:

- `benchmarks/results/table6_long_horizon/table6_long_horizon.json`
- `benchmarks/results/table6_long_horizon/table6_long_horizon.md`

The Markdown file is formatted as a paper-ready summary table; the JSON file contains the underlying row-level metrics for each task and ablation setting.

### Reproducing Table-6 aligned baselines

To compare DeALoG against CoT, ReAct, ReWOO, and planner-style baselines on the same CRT-QA and Multi-Hop slices:

```bash
python scripts/run_table6_baselines.py \
  --llm ${PRIMARY_MODEL_NAME} \
  --systems cot,react,rewoo,planner,planner_replan,dealog \
  --split dev \
  --max-rounds 10 \
  --summarizer-llm ${DEALOG_SUMMARIZER_MODEL} \
  --output-dir benchmarks/results/table6_baselines
```

Useful options:

- `--limit 20` for a smoke test.
- `--decoding '{"temperature":0.2,"max_new_tokens":512}'` to override decoding for non-CoT baselines.
- `--cot-max-new-tokens 256` and `--cot-temperature 0.2` to align baseline decoding settings across runs.

Outputs:

- `benchmarks/results/table6_baselines/table6_baselines.json`
- `benchmarks/results/table6_baselines/table6_baselines.md`
- Per-system raw metric files under `benchmarks/results/table6_baselines/raw/`

### Running the full benchmark matrix

The broader experimental matrix is configured through `configs/planner_benchmarks.yaml`. To inspect the commands without launching jobs:

```bash
python scripts/run_benchmark_matrix.py --dry-run
```

To execute the configured matrix:

```bash
python scripts/run_benchmark_matrix.py --max-workers 6
```

Each configured run is expected to emit a JSON metrics file containing at least:

- `accuracy`
- `per_example`
- `latency_sec`
- `calls`
- `tokens`
- `api_cost`

The runner stores execution logs under `benchmarks/results/<timestamp>/logs` and aggregates the metric file paths in `results.jsonl`.

To compute deltas, bootstrap confidence intervals, and paired permutation tests relative to a CoT baseline:

```bash
python scripts/analyze_benchmarks.py benchmarks/results/<run>/results.jsonl --baseline cot
```

### FeTaQA factuality protocol

For the FeTaQA factuality experiments, including QAGS, BERTScore, log-groundedness, and the associated reporting workflow, use the dedicated note:

- [docs/fetaqa_faithfulness.md](docs/fetaqa_faithfulness.md)

### Legacy fine-tuning path

The repository also includes an earlier LoRA fine-tuning and evaluation path:

```bash
python lora_finetune.py --model mistralai/Mistral-7B-v0.1
python evaluate.py /path/to/fine_tuned_checkpoint --split dev
```

This path evaluates a fine-tuned causal LM on TAT-QA and is separate from the multi-agent DeALoG evaluation scripts above.

## Visual models

If you run the visual pipeline, set the captioning and OCR checkpoints through `.env` or CLI flags:

- `VISUAL_CAPTION_MODEL` / `VISUAL_CAPTION_MODEL_PATH`
- `VISUAL_OCR_ENGINE` / `VISUAL_OCR_MODEL_DIR`

The current repository layout assumes BLIP-2 assets under `models/blip2_flan_t5_xl/` and PaddleOCR assets under `models/paddleocr_cache/`, but these paths are configurable.
