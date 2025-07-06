"""Agent for manipulating tables with natural language instructions."""

from utils.table_ops import apply_tabular_op  # type: ignore


class TableAgent:
    def __init__(self, model):
        self.model = model

    def apply_operation(self, table, operation_str: str):
        """Apply `operation_str` on `table` and return the modified table."""
        # TODO: parse operation_str into a structured operation using the model
        return apply_tabular_op(table, operation_str)
