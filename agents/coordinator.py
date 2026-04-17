"""Adaptive controller that coordinates specialist agents via a shared log."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, List

from .calculation_agent import CalculationAgent
from .context_agent import ContextAgent
from .summarizer_agent import SummarizingAgent
from .table_agent import TableAgent
from .verification_agent import VerificationAgent
from .visual_agent import VisualAgent
from utils.gated_scheduler import GatedScheduler
from utils.llm_client import LLMClient
from utils.shared_log import SharedLog


class AdaptiveOrchestrator:
    """Controller that manages agent turn-taking and verification."""

    def __init__(
        self,
        model_name: str | None = None,
        summarizer_model_name: str | None = None,
        summarizer_temperature: float = 0.2,
        summarizer_max_tokens: int = 256,
        visual_model_name: str | None = None,
        visual_caption_model: str | None = None,
        visual_caption_model_path: str | None = None,
        visual_ocr_engine: str | None = None,
        visual_ocr_model_dir: str | None = None,
        max_rounds: int = 6,
        entry_transform=None,
        scheduler_model_path: str | None = None,
        scheduler_threshold: float = 0.4,
        parallel_retrieval: bool = False,
    ) -> None:
        self.table_agent = TableAgent(model_name)
        self.context_agent = ContextAgent(model_name)
        self.calc_agent = CalculationAgent()
        self.visual_agent = VisualAgent(
            visual_model_name or model_name,
            caption_model_name=visual_caption_model or visual_model_name or model_name,
            caption_model_path=visual_caption_model_path,
            ocr_engine=visual_ocr_engine,
            ocr_model_dir=visual_ocr_model_dir,
        )
        self.summarizer = SummarizingAgent(
            llm_client=LLMClient(default_model=summarizer_model_name or model_name),
            temperature=summarizer_temperature,
            max_tokens=summarizer_max_tokens,
        )
        self.verifier = VerificationAgent(
            llm_client=LLMClient(default_model=summarizer_model_name or model_name),
            model=summarizer_model_name or model_name,
        )
        self._entry_transform = entry_transform
        self._max_rounds = max_rounds
        self._scheduler = GatedScheduler(
            model_path=scheduler_model_path, threshold=scheduler_threshold
        )
        self._parallel_retrieval = parallel_retrieval

    def _table_operation(self, sample: Dict[str, Any]) -> str:
        return (
            sample.get("table_operation")
            or sample.get("operation")
            or sample.get("instruction")
            or "noop"
        )

    def _table_op(self, sample: Dict[str, Any]):
        operation = self._table_operation(sample)
        return self.table_agent.apply_operation(sample.get("table") or [], operation)

    def _calc_expression(self, sample: Dict[str, Any]) -> Optional[str]:
        return (
            sample.get("calculation")
            or sample.get("expression")
            or sample.get("formula")
        )

    def _calc_op(self, sample: Dict[str, Any], question: str):
        expression = self._calc_expression(sample)
        return self.calc_agent.compute(
            expression if isinstance(expression, str) and expression.strip() else None,
            question=question,
            table=sample.get("table"),
            reasoning_steps=sample.get("reasoning_steps"),
            title=sample.get("title"),
            operator_chain=sample.get("operator_chain"),
            base_value=sample.get("base_value"),
        )

    def _append_retrieval(
        self,
        log: SharedLog,
        agent_name: str,
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if agent_name == "context_agent":
            log.append(
                "context_agent",
                "context",
                result,
                metadata=metadata or {"source": "paragraphs"},
                resolves=["context"],
            )
        elif agent_name == "table_agent":
            log.append(
                "table_agent",
                "table",
                result,
                metadata=metadata or {},
                resolves=["table"],
            )
        elif agent_name == "calculation_agent":
            log.append(
                "calculation_agent",
                "calculation",
                result,
                metadata=metadata or {},
                resolves=["calculation"],
            )
        elif agent_name == "visual_agent":
            log.append(
                "visual_agent",
                "visual",
                result,
                resolves=["visual"],
            )
    def run(
        self,
        sample: Dict[str, Any],
        initial_plan: Optional[str] = None,
        initial_needs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run a single QA sample through the coordinated agent pipeline.

        `initial_plan` optionally seeds the log with a planner-produced sketch,
        and `initial_needs` overrides the default inferred needs.
        """

        log = SharedLog(entry_transform=self._entry_transform)
        question = sample.get("question", "")

        needs = initial_needs or ["context", "table", "calculation"]
        if sample.get("visuals"):
            needs.append("visual")
        log.append(
            "controller",
            "question",
            question,
            metadata={"sample_id": sample.get("id")},
            needs=needs,
        )
        if initial_plan:
            log.append(
                "planner",
                "plan",
                initial_plan,
                metadata={"provided": True},
            )

        previous_entries = len(log.entries())
        previous_pending = len(log.pending_needs())

        for round_idx in range(max(1, self._max_rounds)):
            progress = False
            retrieval_start = time.perf_counter()

            if self._parallel_retrieval:
                futures: Tuple = ()
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = []
                    if log.has_pending("context"):
                        futures.append(
                            (
                                "context_agent",
                                {"source": "paragraphs"},
                                executor.submit(
                                    self.context_agent.fetch_relevant_text,
                                    question,
                                    sample.get("paragraphs"),
                                ),
                            )
                        )
                    if log.has_pending("table"):
                        futures.append(
                            (
                                "table_agent",
                                {"operation": self._table_operation(sample)},
                                executor.submit(self._table_op, sample),
                            )
                        )
                    if log.has_pending("calculation"):
                        futures.append(
                            (
                                "calculation_agent",
                                {"expression": self._calc_expression(sample)},
                                executor.submit(self._calc_op, sample, question),
                            )
                        )
                    if log.has_pending("visual"):
                        futures.append(
                            (
                                "visual_agent",
                                {},
                                executor.submit(self.visual_agent.describe, sample.get("visuals")),
                            )
                        )

                    for agent_name, meta, future in futures:
                        result = future.result()
                        self._append_retrieval(log, agent_name, result, meta)
                        progress = True
            else:
                if log.has_pending("context"):
                    context = self.context_agent.fetch_relevant_text(
                        question, sample.get("paragraphs")
                    )
                    log.append(
                        "context_agent",
                        "context",
                        context,
                        metadata={"source": "paragraphs"},
                        resolves=["context"],
                    )
                    progress = True

                if log.has_pending("table"):
                    table_result = self._table_op(sample)
                    log.append(
                        "table_agent",
                        "table",
                        table_result,
                        metadata={"operation": self._table_operation(sample)},
                        resolves=["table"],
                    )
                    progress = True

                if log.has_pending("calculation"):
                    calc_result = self._calc_op(sample, question)
                    log.append(
                        "calculation_agent",
                        "calculation",
                        calc_result,
                        metadata={"expression": self._calc_expression(sample)},
                        resolves=["calculation"],
                    )
                    progress = True

                if log.has_pending("visual"):
                    visual_summary = self.visual_agent.describe(sample.get("visuals"))
                    log.append(
                        "visual_agent",
                        "visual",
                        visual_summary,
                        resolves=["visual"],
                    )
                    progress = True

            retrieval_elapsed = time.perf_counter() - retrieval_start
            log.append(
                "controller",
                "retrieval_round",
                {
                    "round": round_idx,
                    "parallel": self._parallel_retrieval,
                    "retrieval_sec": round(retrieval_elapsed, 3),
                },
            )

            delta_entries = len(log.entries()) - previous_entries
            delta_pending = len(log.pending_needs()) - previous_pending
            summarizer_entry = log.latest("summary")
            summarizer_conf = (
                summarizer_entry.metadata.get("confidence")
                if summarizer_entry and summarizer_entry.metadata
                else 0.5
            )
            image_present = bool(sample.get("visuals"))

            score_features = [
                int(image_present),
                float(summarizer_conf or 0.5),
                float(delta_entries),
                float(delta_pending),
            ]
            continue_score = self._scheduler.score(score_features)

            log.append(
                "controller",
                "scheduler_decision",
                {
                    "round": round_idx,
                    "score": round(continue_score, 4),
                    "threshold": self._scheduler.threshold,
                    "features": {
                        "image_present": image_present,
                        "summarizer_conf": summarizer_conf,
                        "delta_entries": delta_entries,
                        "delta_pending": delta_pending,
                    },
                },
            )

            previous_entries = len(log.entries())
            previous_pending = len(log.pending_needs())

            if not progress or not any(log.pending_needs()):
                break
            if not self._scheduler.should_continue(score_features):
                break

        # Request a summary and final verification if not already satisfied
        log.append(
            "controller",
            "request_summary",
            "Synthesize final answer",
            needs=["summary"],
        )

        summary_payload = self.summarizer.synthesize(question, log)
        log.append(
            "summarizer_agent",
            "summary",
            summary_payload,
            metadata={
                "confidence": summary_payload.get("confidence"),
                "prompt_tokens": summary_payload.get("prompt_tokens"),
                "completion_tokens": summary_payload.get("completion_tokens"),
            },
            resolves=["summary"],
            needs=["verification"],
        )

        candidate_answer = summary_payload.get("answer", "")
        is_verified = self.verifier.verify(candidate_answer, sample.get("answer"))
        log.append(
            "verification_agent",
            "verification",
            {
                "verified": is_verified,
                "reference_available": sample.get("answer") is not None,
            },
            resolves=["verification"],
        )

        result = {
            "answer": candidate_answer,
            "rationale": summary_payload.get("rationale"),
            "verified": is_verified,
            "log": log.to_dict(),
        }
        return result
