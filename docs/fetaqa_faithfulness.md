# FeTaQA Faithfulness & Human Evaluation Playbook

This note walks through the practical steps for reproducing the FeTaQA factuality
study referenced in Sec. 3 / App. B and for inserting the new table immediately
after Table 2 in the paper manuscript.

## 1. Produce model traces

```bash
python scripts/run_dealog.py \
  --dataset fetaqa --split dev \
  --llm ${PRIMARY_MODEL_NAME} \
  --results-file benchmarks/results/fetaqa_dev_run.json \
  --visual-caption-model ${VISUAL_CAPTION_MODEL} \
  --visual-caption-path ${VISUAL_CAPTION_MODEL_PATH} \
  --visual-ocr-engine ${VISUAL_OCR_ENGINE} \
  --visual-ocr-model-dir ${VISUAL_OCR_MODEL_DIR}
```

`scripts/run_dealog.py` now keeps the explanation rationale plus the entire
shared log inside each `per_example` record, which the downstream metric scripts
expect.

## 2. Automatic factuality metrics

After the run completes, compute the three metrics:

```bash
# QAGS-style faithfulness
python scripts/compute_qags.py \
  --dataset fetaqa --split dev \
  --results-file benchmarks/results/fetaqa_dev_run.json \
  --question-model valhalla/t5-base-qg-hl \
  --qa-model deepset/roberta-large-squad2 \
  --output-file benchmarks/results/fetaqa_dev_qags.json

# BERTScore (answer vs. reference)
python scripts/compute_bertscore.py \
  --results-file benchmarks/results/fetaqa_dev_run.json \
  --output-file benchmarks/results/fetaqa_dev_bertscore.json \
  --model-type microsoft/deberta-large-mnli

# Log-Groundedness (evidence coverage)
python scripts/compute_log_groundedness.py \
  --results-file benchmarks/results/fetaqa_dev_run.json \
  --output-file benchmarks/results/fetaqa_dev_grounded.json
```

Each JSON file contains a `mean_*` summary plus per-example details for error
analysis.

## 3. Human evaluation on 100 examples

1. **Sample selection** – Draw 100 FeTaQA dev samples with a fixed RNG seed
   (e.g., `python - <<'PY' ...`) and export a CSV containing question/table/text,
   model answer, explanation, and the subset of log entries tagged as
   `LOOKUP`/`QUOTE`/`VISUAL`.
2. **Annotator kit** – Provide the CSV plus the following rubric:
   - Faithfulness labels {Supported, Partially supported, Unsupported}
   - Completeness/Fluency (optional, keeps annotators calibrated)
   - Instructions: mark “Unsupported” if any number/entity in the explanation
     lacks evidence in the supplied table/text/log snippets.
3. **Tooling** – A lightweight Streamlit form makes labeling fast; store outputs
   as JSON/CSV with one row per annotator per example.
4. **Aggregation** – Majority vote the categorical Faithfulness label, then
   compute Krippendorff’s α (use the `krippendorff` Python package) over the raw
   labels. Report both the mean Supported rate and the α estimate (±95 % CI).

## 4. Reporting (Table directly after Table 2)

Use the metric summaries above to populate the new factuality table. Example
Markdown snippet for the paper/README:

> **Table 3**: FeTaQA dev factuality metrics for DeALoG. Reported values are
> averaged over the 100-example sample used for human evaluation. Insert this
> table immediately after Table 2 in the manuscript.

| Model | QAGS ↑ | BERTScore F1 ↑ | Log-Groundedness ↑ | Human Faithfulness ↑ | κ (α) ↑ |
|-------|--------|----------------|--------------------|----------------------|---------|
| DeALoG (`${PRIMARY_MODEL_NAME}`) | `<qags>` | `<bertscore>` | `<log_grounded>` | `<human_rate>` | `<alpha>` |

Replace the angle-bracket placeholders with the numeric aggregates from the JSON
reports and the human study. Cite the table in the surrounding text (e.g., “See
Table 3 for FeTaQA factuality metrics complementing the accuracy results in
Table 2.”).
