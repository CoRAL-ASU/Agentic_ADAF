from .table_agent import TableAgent
from .context_agent import ContextAgent
from .calculation_agent import CalculationAgent


class AdaptiveOrchestrator:
    """Simple coordinator that routes requests to agents."""

    def __init__(self, model_name: str | None = None) -> None:
        self.table_agent = TableAgent(model_name)
        self.context_agent = ContextAgent(model_name)
        self.calc_agent = CalculationAgent()

    def run(self, sample: dict) -> dict:
        """Run a single QA sample through the agent pipeline."""

        context = self.context_agent.fetch_relevant_text(
            sample.get("question", ""), sample.get("paragraphs")
        )
        _ = self.table_agent.apply_operation(sample.get("table"), "noop")
        answer = self.calc_agent.compute("0")        return {"answer": answer, "context": context}
