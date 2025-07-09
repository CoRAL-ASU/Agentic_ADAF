"""Agent for manipulating tables with simple natural language instructions."""

from typing import Any, Iterable, Tuple

from utils.table_ops import apply_tabular_op


class TableAgent:
    """Parse text instructions and apply tabular operations."""

    def __init__(self, model: Any | None = None) -> None:  # pragma: no cover - trivial
        self.model = model

    def _parse_operation(self, instruction: str) -> Tuple:
        """Convert a free-form instruction into a structured operation."""

        tokens = instruction.lower().split()

        if len(tokens) >= 3 and tokens[0] == "select" and tokens[1] == "row":
            try:
                return ("row", int(tokens[2]))
            except ValueError:
                pass

        if len(tokens) >= 3 and tokens[0] == "select" and tokens[1] == "column":
            try:
                return ("column", int(tokens[2]))
            except ValueError:
                pass

        if len(tokens) >= 4 and tokens[0] == "get" and tokens[1] == "cell":
            try:
                return ("cell", int(tokens[2]), int(tokens[3]))
            except ValueError:
                pass

        if len(tokens) >= 3 and tokens[0] == "sum" and tokens[1] == "column":
            try:
                return ("sum_column", int(tokens[2]))
            except ValueError:
                pass

        return ("noop", instruction)

    def apply_operation(self, table: Iterable[Iterable[Any]], operation_str: str):
        """Apply ``operation_str`` on ``table`` and return the result."""

        op = self._parse_operation(operation_str)
        return apply_tabular_op(table, op)
