
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FeTaQA QAGS pipeline
--------------------
Computes QAGS-style factuality metrics for FeTaQA (free-form table QA) explanations.
Given a JSONL with fields:
 - id: unique id (string or int)
 - candidate_explanation: str (system output)
 - reference_explanation: str (gold explanation)
 - table_text: str (linearized table)  OR
   table (structured; list of dict rows or 2D) -> we will linearize
 - optional: extra_context: str (retrieved passage text, if any)
 - optional: log_entries: list[dict] (DeALOG shared-log entries as in the paper)
 - optional: bertscore_f1 (float) precomputed sentence/summary-level BERTScore
The script will:
 1) Sentence-segment the candidate explanation.
 2) Rank sentences by (1 - BERTScore vs reference) if reference provided, else by length.
 3) Use a question-generation model to produce 1..N questions per selected sentence.
 4) Answer those questions on (a) the source context (table_text + extra_context) and (b) the candidate explanation
    using a span-extractive QA model.
 5) Compare the two answers with token-F1 and/or BERTScore to get a per-question match score; average to yield QAGS.
 6) Optionally weight per-question scores by sentence weights derived from BERTScore (weighted QAGS_BS).
 7) Optionally compute a Log-Groundedness score by checking if entities/numerals in candidate are supported by log entries.
 8) Aggregate, bootstrap CIs, and write a CSV and JSON with details.

Dependencies (install before running):
    pip install pandas tqdm numpy torch transformers bert-score rapidfuzz

Usage:
    python fetaqa_qags_pipeline.py \
       --data /path/to/fetaqa_with_preds.jsonl \
       --out /path/to/out_dir \
       --qg-model valhalla/t5-base-qg-hl \
       --qa-model deepset/roberta-base-squad2 \
       --num-questions 5 --seed 42 --bootstrap 1000

Input JSONL schema example (one item per line):
{"id": "fetaqa_0001",
 "candidate_explanation": "The company's net income increased to $5M in 2019 due to higher sales.",
 "reference_explanation": "In 2019, net income rose to $5 million primarily driven by increased sales volume.",
 "table_text": "Year | Net income\n2018 | $3M\n2019 | $5M",
 "extra_context": "The report states: 'Increase due to higher sales volume.'",
 "log_entries": [{"agent": "TableAgent", "type": "LOOKUP", "content":"Net income 2019: $5M"}, ...]}

Outputs:
  out/qags_scores.csv (per-example metrics)
  out/qags_details.jsonl (per-question details per example)
  out/summary.txt (dataset-level summary with bootstrap CIs)
"""

import os, re, json, math, random, argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# HuggingFace + metrics
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
from bert_score import score as bertscore
from rapidfuzz import fuzz

# ----------------
# Utilities
# ----------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?;:])\s+')

def sent_segment(text: str) -> List[str]:
    if not text: 
        return []
    # Basic sentence segmentation (avoid external heavy deps)
    # Keep small sentences but trim spaces
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    return sents

def simple_table_linearize(table) -> str:
    """Accepts a list[dict] rows OR 2D list or TSV string; returns a textual linearization."""
    if table is None:
        return ""
    if isinstance(table, str):
        return table
    lines = []
    if isinstance(table, list):
        # list of dict rows or list of lists
        if len(table) == 0:
            return ""
        if isinstance(table[0], dict):
            headers = list({h for row in table for h in row.keys()})
            lines.append(" | ".join(headers))
            for row in table:
                line = " | ".join(str(row.get(h, "")) for h in headers)
                lines.append(line)
        elif isinstance(table[0], (list, tuple)):
            for row in table:
                lines.append(" | ".join(str(x) for x in row))
        else:
            # Fallback
            lines = [str(x) for x in table]
    else:
        lines = [str(table)]
    return "\n".join(lines)

def token_f1(a_pred: str, a_gold: str) -> float:
    def norm(s):
        return re.sub(r'\s+', ' ', s.lower().strip())
    pred = norm(a_pred)
    gold = norm(a_gold)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    p_toks = pred.split()
    g_toks = gold.split()
    common = {}
    for t in p_toks:
        common[t] = min(p_toks.count(t), g_toks.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    return 2 * precision * recall / (precision + recall + 1e-8)

def jaccard(a: str, b: str) -> float:
    A = set(a.lower().split())
    B = set(b.lower().split())
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def bertscore_pair(a: str, b: str, model_type: str = "microsoft/deberta-xlarge-mnli") -> float:
    # Returns average F1 BERTScore for a single pair
    P, R, F1 = bertscore([a], [b], model_type=model_type, verbose=False, rescale_with_baseline=True)
    return float(F1[0])

def extract_entities_and_numbers(text: str) -> List[str]:
    # Lightweight entity/number extraction for Log-Groundedness
    nums = re.findall(r'[-+]?\d[\d,]*\.?\d*%?', text)
    caps = re.findall(r'\b[A-Z][a-zA-Z0-9\-\_]{2,}\b', text)  # crude Proper-like tokens
    syms = re.findall(r'\$[0-9][\d,]*\.?\d*', text)
    tokens = list(dict.fromkeys(nums + syms + caps))
    return tokens

def log_groundedness(candidate: str, log_entries: List[Dict[str, Any]]) -> float:
    """Checks what fraction of entities/numbers in candidate appear in LOOKUP/QUOTE/VISUAL contents of the log."""
    if not candidate or not log_entries:
        return float('nan')
    probe = extract_entities_and_numbers(candidate)
    if not probe:
        return float('nan')
    corpus = " ".join([str(e.get("content","")) for e in log_entries if str(e.get("type","")).upper() in {"LOOKUP","QUOTE","VISUAL"}])
    corpus_low = corpus.lower()
    hits = 0
    for p in probe:
        if p.lower() in corpus_low:
            hits += 1
    return hits / max(1, len(probe))

# ----------------
# QG & QA
# ----------------

def build_qg(qg_model: str, device: int = -1):
    # For e.g. "valhalla/t5-base-qg-hl" or "iarfmoose/t5-base-question-generator"
    # We use text2text-generation pipeline
    return pipeline("text2text-generation", model=qg_model, device=device)

def build_qa(qa_model: str, device: int = -1):
    return pipeline("question-answering", model=qa_model, device=device)

def make_qg_prompt(sentence: str) -> str:
    # Works for highlight-based T5 QG models; fallback is simply "generate question" instruction
    # Many QG models expect "generate question: <hl> sentence </hl>"
    s = sentence.strip().replace("\n", " ")
    return f"generate question: {s}"

def generate_questions(qg_pipe, sentences: List[str], max_questions: int, num_beams: int = 4) -> List[str]:
    qs = []
    for s in sentences:
        prompt = make_qg_prompt(s)
        out = qg_pipe(prompt, max_length=64, num_beams=num_beams, do_sample=False, clean_up_tokenization_spaces=True)
        q = out[0]['generated_text'].strip()
        if q and q not in qs:
            qs.append(q)
        if len(qs) >= max_questions:
            break
    return qs

def answer_question(qa_pipe, question: str, context: str) -> str:
    if not context.strip():
        return ""
    try:
        ans = qa_pipe(question=question, context=context)
        if isinstance(ans, dict):
            return ans.get("answer", "").strip()
        return ""
    except Exception as e:
        return ""

# ----------------
# Main computation
# ----------------

def compute_qags_for_item(
    item: Dict[str, Any],
    qg_pipe,
    qa_pipe,
    max_questions: int,
    bertscore_model: str,
    sent_bs_override: list | None = None,
    sent_weighting: bool = True,
    use_bertscore_for_match: bool = False,
    disable_bertscore: bool = False,
) -> Dict[str, Any]:
    cid = item.get("id")
    cand = (
        item.get("candidate_explanation")
        or item.get("rationale")
        or item.get("prediction")
        or ""
    ).strip()
    ref = (item.get("reference_explanation") or item.get("reference") or "").strip()

    # Build source context: table_text + optional extra_context
    table_text = item.get("table_text")
    if table_text is None and item.get("table") is not None:
        table_text = simple_table_linearize(item.get("table"))
    table_text = (table_text or "").strip()
    extra = (item.get("extra_context") or "").strip()

    source_context = "\n".join([x for x in [table_text, extra] if x])

    # Sentence ranking by (1 - BERTScore)
    sents = sent_segment(cand)
    if not sents:
        return {"id": cid, "qags": float('nan'), "qags_bs": float('nan'), "num_q": 0}
    # Compute per-sentence BERTScore vs ref if available
    sent_scores = []
    if disable_bertscore:
        sent_scores = [0.5 for _ in sents]
    elif sent_bs_override and len(sent_bs_override) == len(sents):
        sent_scores = [float(x) for x in sent_bs_override]
    elif ref:
        P, R, F1 = bertscore(
            sents,
            [ref] * len(sents),
            model_type=bertscore_model,
            rescale_with_baseline=True,
            lang="en",
        )
        sent_scores = [float(f) for f in F1]
    else:
        sent_scores = [0.5 for _ in sents]  # neutral if no ref
    # Lower-scoring sentences (vs ref) are targeted to generate questions
    idx_sorted = list(sorted(range(len(sents)), key=lambda i: sent_scores[i]))
    picked = [sents[i] for i in idx_sorted[:max_questions]]

    # Generate questions
    questions = generate_questions(qg_pipe, picked, max_questions=max_questions)

    # Answer on source and on candidate explanation
    details = []
    per_q_scores = []
    per_q_weights = []
    for q in questions:
        a_src = answer_question(qa_pipe, q, source_context if source_context else ref)
        a_cand = answer_question(qa_pipe, q, cand)
        # Matching: token-F1 and optional BERTScore
        f1 = token_f1(a_src, a_cand)
        if use_bertscore_for_match:
            try:
                bs = bertscore_pair(a_src, a_cand, model_type=bertscore_model)
            except Exception:
                bs = float('nan')
        else:
            bs = float('nan')
        per_q_scores.append(f1 if not use_bertscore_for_match else (0.5*f1 + 0.5*(0.0 if math.isnan(bs) else bs)))
        # Weight by (1 - sent BERTScore) for the source sentence that produced this question (approximate)
        # Map question to its generating sentence by index (same order)
        sent_idx = questions.index(q) if q in questions else 0
        w = 1.0 - (sent_scores[idx_sorted[sent_idx]] if sent_scores else 0.5)
        per_q_weights.append(max(0.0, min(1.0, w)))
        details.append({
            "id": cid, "question": q, "ans_source": a_src, "ans_cand": a_cand,
            "token_f1": f1, "bertscore_pair": bs
        })

    if not per_q_scores:
        return {"id": cid, "qags": float('nan'), "qags_bs": float('nan'), "num_q": 0, "details": details}

    qags = float(np.nanmean(per_q_scores))
    if sent_weighting and sum(per_q_weights) > 0:
        qags_bs = float(np.nansum([s*w for s,w in zip(per_q_scores, per_q_weights)]) / (np.nansum(per_q_weights) + 1e-8))
    else:
        qags_bs = qags

    # Optional: Log-Groundedness
    lg = float('nan')
    if isinstance(item.get("log_entries"), list) and len(item.get("log_entries")) > 0:
        try:
            lg = log_groundedness(cand, item.get("log_entries"))
        except Exception:
            lg = float('nan')

    return {
        "id": cid,
        "qags": qags,
        "qags_bs": qags_bs,
        "num_q": len(questions),
        "log_groundedness": lg,
        "details": details
    }

def bootstrap_ci(values: List[float], iters: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    vals = np.array([v for v in values if not (v is None or (isinstance(v, float) and np.isnan(v)))])
    if len(vals) == 0:
        return (float('nan'), float('nan'))
    n = len(vals)
    boots = []
    for _ in range(iters):
        sample = np.random.choice(vals, size=n, replace=True)
        boots.append(np.nanmean(sample))
    lo = np.quantile(boots, alpha/2)
    hi = np.quantile(boots, 1 - alpha/2)
    return float(lo), float(hi)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to JSONL with fields described in header.")
    ap.add_argument("--out", type=str, required=True, help="Output directory.")
    ap.add_argument("--qg-model", type=str, default="valhalla/t5-base-qg-hl")
    ap.add_argument("--qa-model", type=str, default="deepset/roberta-base-squad2")
    ap.add_argument("--bertscore-model", type=str, default="microsoft/deberta-xlarge-mnli")

    ap.add_argument("--sent-bertscore-json", type=str, default=None,
                    help="Optional JSON mapping: id -> list[float] of sentence-level BERTScore (aligns with candidate sentence order).")

    ap.add_argument("--max-questions", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--use-bertscore-for-match", action="store_true", help="Blend token-F1 with BERTScore for answer-match")
    ap.add_argument("--disable-bertscore", action="store_true", help="Skip BERTScore sentence weighting (offline mode).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seed_everything(args.seed)

    # Load models
    print("Loading QG model:", args.qg_model, flush=True)
    qg_pipe = build_qg(args.qg_model, device=-1)
    print("Loading QA model:", args.qa_model, flush=True)
    qa_pipe = build_qa(args.qa_model, device=-1)

    # Read data
    items = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    # Optional: load precomputed sentence-level BERTScore weights
    sent_bs_map = None
    if args.sent_bertscore_json:
        with open(args.sent_bertscore_json, "r", encoding="utf-8") as jf:
            sent_bs_map = json.load(jf)

    results = []
    detail_fpath = os.path.join(args.out, "qags_details.jsonl")
    with open(detail_fpath, "w", encoding="utf-8") as df:
        for it in tqdm(items, desc="Computing QAGS"):
            sent_bs = sent_bs_map.get(str(it.get("id"))) if sent_bs_map else None
            r = compute_qags_for_item(
                it,
                qg_pipe=qg_pipe,
                qa_pipe=qa_pipe,
                max_questions=args.max_questions,
                bertscore_model=args.bertscore_model,
                sent_bs_override=sent_bs,
                sent_weighting=True,
                use_bertscore_for_match=args.use_bertscore_for_match,
                disable_bertscore=args.disable_bertscore,
            )
            results.append({k:v for k,v in r.items() if k != "details"})
            for d in r.get("details", []):
                df.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Dataframe and write CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.out, "qags_scores.csv")
    df.to_csv(csv_path, index=False)

    # Dataset-level summary + CIs
    qags_vals = [x.get("qags", float('nan')) for x in results]
    qags_bs_vals = [x.get("qags_bs", float('nan')) for x in results]

    lo1, hi1 = bootstrap_ci(qags_vals, iters=args.bootstrap)
    lo2, hi2 = bootstrap_ci(qags_bs_vals, iters=args.bootstrap)

    summary = {
        "N": len(results),
        "QAGS_mean": float(np.nanmean(qags_vals)),
        "QAGS_CI": [lo1, hi1],
        "QAGS_BS_mean": float(np.nanmean(qags_bs_vals)),
        "QAGS_BS_CI": [lo2, hi2],
        "Note": "QAGS_BS is weighted by (1 - sentence-level BERTScore) to focus on low-overlap, likely-factual segments."
    }
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as sf:
        sf.write(json.dumps(summary, indent=2))

    print("Done. Wrote:", csv_path, detail_fpath, os.path.join(args.out, "summary.txt"))

if __name__ == "__main__":
    main()
