"""Coordinator agent that distils log entries into a candidate answer."""

from __future__ import annotations

import re
from typing import Dict, Optional, List

from utils.llm_client import LLMClient

from utils.shared_log import SharedLog


class SummarizingAgent:
    """Synthesize an answer from accumulated log entries."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        *,
        temperature: float = 0.2,
        max_tokens: int = 256,
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._temperature = temperature
        self._max_tokens = max_tokens

    def synthesize(self, question: str, log: SharedLog) -> Dict[str, str]:
        """Return a candidate answer and supporting rationale."""

        context = log.latest("context")
        table = log.latest("table")
        calculation = log.latest("calculation")
        visual = log.latest("visual")

        evidence: List[str] = []
        if context and context.content:
            evidence.append(f"Context: {context.content}")
        if table and table.content not in (None, []):
            evidence.append(f"Table insight: {table.content}")
        if calculation and calculation.content not in (None, ""):
            evidence.append(f"Calculation: {calculation.content}")
        if visual and visual.content and visual.content != "No visual artefacts provided.":
            evidence.append(f"Visual cues: {visual.content}")

        rationale = " \n".join(["Question: " + question] + evidence)

        # For synthetic arithmetic chains, the calculator output is the most
        # faithful source of truth. Use it directly to avoid LLM formatting drift.
        if self._looks_like_multi_hop_arithmetic(question) and calculation and calculation.content not in (None, ""):
            calc_text = str(calculation.content).strip()
            calc_numeric = self._extract_numeric_span(calc_text)
            direct_answer = calc_numeric or calc_text
            return {
                "answer": direct_answer,
                "rationale": rationale,
                "confidence": 1.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        prompt = self._build_prompt(question, evidence)
        max_tokens = self._runtime_max_tokens(question)
        llm_result = (
            self._llm.complete(
                prompt,
                temperature=self._temperature,
                max_tokens=max_tokens,
            )
            if self._llm
            else None
        )

        normalized = None
        confidence = 0.5
        prompt_tokens = None
        completion_tokens = None

        if not llm_result or not isinstance(llm_result, dict):
            raise RuntimeError("Summarizer LLM call failed; heuristic fallback is disabled.")
        llm_answer = llm_result.get("content") or ""
        usage = llm_result.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        confidence = self._confidence_from_usage(usage)
        answer, normalized = self._parse_answer(llm_answer, question=question)

        return {
            "answer": normalized or answer,
            "rationale": rationale,
            "confidence": confidence,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def _build_prompt(self, question: str, evidence: List[str]) -> str:
        evidence_text = "\n".join(f"{idx+1}. {item}" for idx, item in enumerate(evidence)) or "1. (No evidence provided)"
        if self._looks_like_finqa(question):
            instructions = (
                "You are answering a financial report question. The tables list yearly metrics (e.g., revenue, profit, assets).\n"
                "Identify the year or row mentioned in the question, select the correct column (revenue/profit/etc.), and output only the requested number.\n"
                "Show intermediate arithmetic if necessary (e.g., 2020 value - 2019 value).\n"
                "Format:\n"
                "Reasoning: describe the lookup (year, column) and any arithmetic.\n"
                "Answer: <numeric value only>."
            )
        elif self._looks_like_crtqa(question):
            instructions = (
                "You are solving CRT-QA style table reasoning.\n"
                "Use only the provided evidence. Prioritize table cells and calculation outputs over free-text context.\n"
                "When evidence is insufficient, choose the closest grounded answer from evidence and keep output concise.\n"
                "If the question requests a constrained format, obey it exactly.\n"
                "Format:\n"
                "Reasoning: short evidence-grounded steps (row/column references when available).\n"
                "Answer: final answer only."
            )
        else:
            instructions = (
                "You are a financial/table reasoning assistant.\n"
                "1. Break the question down and note which row/column of the table or context provides each fact.\n"
                "2. When arithmetic is needed, show the intermediate calculation (e.g., 2020 value - 2019 value = X).\n"
                "3. End with a short final answer.\n"
                "Format:\n"
                "Reasoning: Step-by-step explanation referencing specific evidence items.\n"
                "Answer: <concise numeric or textual answer>."
            )
        q_lower = question.lower()
        if "answer with only 'yes' or 'no'" in q_lower or "answer with only \"yes\" or \"no\"" in q_lower:
            instructions += "\nReturn exactly one token in Answer: Yes or No."
        if "answer with only 'more', 'less' or 'equal'" in q_lower or "answer with only \"more\", \"less\" or \"equal\"" in q_lower:
            instructions += "\nReturn exactly one token in Answer: more, less, or equal."
        if "answer with only 'better', 'worse' or 'equal'" in q_lower or "answer with only \"better\", \"worse\" or \"equal\"" in q_lower:
            instructions += "\nReturn exactly one token in Answer: better, worse, or equal."
        return f"{instructions}\nQuestion: {question}\nEvidence:\n{evidence_text}"

    def _parse_answer(self, llm_response: str, *, question: str = "") -> tuple[str, Optional[str]]:
        lines = [line.strip() for line in llm_response.strip().splitlines() if line.strip()]
        candidate = None
        for line in lines:
            lower = line.lower()
            if lower.startswith("answer:"):
                candidate = line.split(":", 1)[1].strip()
                break
        if not candidate:
            candidate = lines[-1] if lines else llm_response.strip()

        constrained = self._extract_constrained_answer(question, candidate, llm_response)
        if constrained is not None:
            return constrained, self._extract_numeric_span(constrained)

        normalized = self._extract_numeric_span(candidate)
        return candidate, normalized

    def _extract_constrained_answer(
        self, question: str, candidate: str, raw_response: str
    ) -> Optional[str]:
        q = (question or "").lower()
        cand = (candidate or "").strip()
        raw = (raw_response or "").lower()
        merged = f"{cand.lower()} {raw}"

        if "answer with only 'yes' or 'no'" in q or "answer with only \"yes\" or \"no\"" in q:
            tokens = re.findall(r"\b(yes|no)\b", merged)
            if tokens:
                return tokens[0].capitalize()
            return None

        if "answer with only 'more', 'less' or 'equal'" in q or "answer with only \"more\", \"less\" or \"equal\"" in q:
            tokens = re.findall(r"\b(more|less|equal)\b", merged)
            if tokens:
                return tokens[0]
            return None

        if "answer with only 'better', 'worse' or 'equal'" in q or "answer with only \"better\", \"worse\" or \"equal\"" in q:
            tokens = re.findall(r"\b(better|worse|equal)\b", merged)
            if tokens:
                return tokens[0]
            return None

        # Generic constrained-option recovery: extract quoted options from question
        # and select the first option mentioned in the model output.
        if "answer with only" in q:
            options = self._extract_quoted_options(question)
            for option in options:
                pattern = r"\b" + re.escape(option).replace(r"\ ", r"\s+") + r"\b"
                if re.search(pattern, merged):
                    return option

        if "answer with only" in q and "nothing else" in q:
            # For constrained-format prompts, use first non-empty token span.
            cleaned = cand.strip().strip(".")
            if cleaned:
                return cleaned
        return None

    def _extract_quoted_options(self, question: str) -> List[str]:
        options: List[str] = []
        for raw in re.findall(r"'([^']+)'|\"([^\"]+)\"", question):
            opt = (raw[0] or raw[1] or "").strip().lower()
            if opt and opt not in options:
                options.append(opt)
        return options

    def _looks_like_finqa(self, question: str) -> bool:
        lowered = question.lower()
        has_year = bool(re.search(r"(19|20)\d{2}", lowered))
        financial_terms = ["revenue", "profit", "earnings", "reported", "income"]
        return has_year and any(term in lowered for term in financial_terms)

    def _looks_like_crtqa(self, question: str) -> bool:
        lowered = question.lower()
        crt_markers = [
            "answer with only",
            "standard deviation",
            "most common",
            "outlier",
            "top",
            "within one standard deviation",
            "versus",
        ]
        return any(marker in lowered for marker in crt_markers)

    def _looks_like_multi_hop_arithmetic(self, question: str) -> bool:
        lowered = question.lower()
        return (
            "start from" in lowered
            and "apply the following" in lowered
            and "operations in order" in lowered
            and "final value" in lowered
        )

    def _runtime_max_tokens(self, question: str) -> int:
        q = (question or "").lower()
        if self._looks_like_crtqa(question) and "answer with only" in q:
            return min(self._max_tokens, 32)
        if self._looks_like_crtqa(question):
            return min(self._max_tokens, 256)
        return self._max_tokens

    def _extract_numeric_span(self, text: str) -> Optional[str]:
        match = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", text.replace(",", ""))
        if match:
            return match.group(0)
        return None

    def _confidence_from_usage(self, usage: Dict[str, Optional[int]]) -> float:
        # Simple heuristic: presence of token counts -> high-ish confidence, else fallback.
        if usage.get("prompt_tokens") is not None or usage.get("completion_tokens") is not None:
            return 0.8
        return 0.5
