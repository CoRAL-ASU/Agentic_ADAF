# Shared Log Schema & Agent Contracts

This document expands the repository's architecture notes with the concrete
schema, message types, and coordination rules referenced throughout the paper
excerpt. It serves as the missing Appendix-style reference for engineers who
need to extend the multi-agent system or plug in new specialists.

## Design Goals & Assumptions

1. **Modularity** – Specialists remain small, single-responsibility modules
   (`agents/*.py`) that can be swapped without retraining a monolithic model.
2. **Transparency** – Every decision is written to a provenance-aware,
   append-only log so downstream users can audit how the answer was produced.
3. **Robustness** – Specialists watch one another through the log, making it
   easier to catch arithmetic slips, retrieval omissions, or hallucinations.
4. **Inputs** – We assume access to tabular corpora, short unstructured text
   snippets, and lightweight visual metadata. Agents are prompt-only (zero/few
   shot) and do not maintain hidden state outside the log.

The `AdaptiveOrchestrator` acts as a scheduler, not a planner: it decides who
gets a turn, enforces guardrails, and knows when to stop, but it never spits out
pre-planned tool chains.

## Shared Log Structure

Each log record is a tuple `(agent, type, content, metadata)` with optional
coordination annotations (`needs`, `resolves`). The concrete implementation
lives in `utils/shared_log.py`, while this table captures the fixed vocabulary
used by the agents:

| Type      | Purpose                                                                  |
|-----------|--------------------------------------------------------------------------|
| `LOOKUP`  | Structured table extracts, derived values, or filtered row/column views. |
| `QUOTE`   | Direct text quotations or short paraphrases from retrieved passages.     |
| `VISUAL`  | OCR/caption interpretations of charts, infographics, or figures.         |
| `SUMMARY` | Progress reports or intermediate synthesis from the SummarizingAgent.    |
| `ANSWER`  | Candidate final answer proposed by the SummarizingAgent.                 |
| `FLAG`    | Verification failure reason emitted by VerificationAgent.                |
| `OK`      | Positive verification result when an answer is confirmed.                |

`metadata` preserves provenance (table id, row, column, document span, or image
ID) plus bookkeeping information such as `step`, `timestamp`, or references
back to dataset artefacts. The log exposes a deduplicated, globally visible view
to all agents—restricting visibility was empirically harmful, so no such
ablation is enabled by default.

## Agent Roles & I/O Contracts

Every specialist implements two primitives:

```python
def should_act(log: SharedLog) -> bool:
    ...

def act(log: SharedLog) -> LogEntry:
    ...
```

`should_act` is a lightweight, heuristic trigger (e.g., TableAgent checks for a
pending `LOOKUP` need, ContextAgent notices unanswered document requests).
`act` performs the expensive LLM/tool call and appends a typed entry. In this
repository the heuristics are encoded inside `AdaptiveOrchestrator.run`, which
uses `log.has_pending(...)` checks to decide when to call each agent. When
adding new specialists, follow the same contract so they can be slotted into the
controller loop with minimal changes.

Responsibilities:

- **TableAgent** (`agents/table_agent.py`) – Parses instructions and posts
  `LOOKUP` entries for cells, rows, aggregates, or derived values.
- **ContextAgent** (`agents/context_agent.py`) – Emits `QUOTE` entries with
  relevant textual spans.
- **VisualAgent** (`agents/visual_agent.py`) – Converts visual metadata to
  `VISUAL` entries via captioning/OCR.
- **SummarizingAgent** (`agents/summarizer_agent.py`) – Periodically writes
  `SUMMARY` updates and, when sufficient evidence exists, an `ANSWER`.
- **VerificationAgent** (`agents/verification_agent.py`) – Reads the proposed
  answer, recomputes sensitive steps, and emits either `OK` or `FLAG`.

## Controller Loop & Stopping

The control flow mirrors Algorithm&nbsp;1 from the paper:

1. **Initialization** – The controller logs the user question and seeds any
   outstanding `needs` (context, table, calculation, visual).
2. **Retrieval rounds** – In each round, the controller offers turns to the
   retrieval specialists `{Table, Context, Visual}`. Agents append new evidence
   or abstain; duplication filters prevent near-identical entries.
3. **Summarization** – Whenever new evidence appears (or patience expires), the
   SummarizingAgent runs. If it outputs `ANSWER`, execution proceeds to
   verification; otherwise it emits a `SUMMARY` describing missing items.
4. **Verification** – VerificationAgent inspects the latest reasoning trace. If
   the answer is sound it logs `OK` and the loop terminates. On `FLAG`, the
   controller may trigger one extra retrieval round targeted at the cited gap.

Guardrails enforced by the controller:

- Maximum rounds `R = 6`
- Per-agent action caps to avoid runaway loops
- Deduplication of near-identical entries

Empirically, most questions resolve within three to four agent calls even though
the theoretical bound is `O(R * A)` for `A` retrieval specialists.

## Verification & Re-engagement

Verification re-derives calculations, checks that cited spans actually support
claims, and ensures table references are consistent. If a discrepancy exists,
the verifier emits a `FLAG` entry explaining the missing fact (e.g., "CEO in
2020 missing"). The scheduler then allows a single re-engagement round so the
relevant retrieval agent(s) can target that gap. This "second chance" corrects
most arithmetic or retrieval omissions while preventing infinite loops.

## Memory Management & Truncation

To keep prompts within model context limits, the log maintains two tiers:

- **Recent entries** – Preserved verbatim for precise citations.
- **Summaries** – Older spans are compressed into `SUMMARY` stubs that retain
  citations and provenance but collapse verbose text.

Source-aware truncation rules apply before summarization: table excerpts are
trimmed to the touched rows/columns, textual quotes clip to ±k sentences, and
visual entries retain numeric strings surfaced by OCR. This keeps
SummarizingAgent's working window roughly constant without sacrificing
evidentiary support.

## Complexity & Parallelism

For `R` rounds and `A` retrieval agents the worst-case number of LLM/tool calls
is `O(R * A)`. In practice, instrumentation across the provided datasets shows
an average of 3.1 calls per query versus 3.8 for a planner-based baseline. While
the reference implementation executes sequentially (current API constraint),
timestamps in the log preserve a deterministic total order so retrieval steps
could be parallelised safely in future.
