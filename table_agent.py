class TableAgent:
    def __init__(self, model):
        self.model = model

    def apply_operation(self, table, operation_str):
        # Parse operation_str like "filter rows where Revenue > 1000"
        # Use table_ops to apply it and return updated table
        from utils.table_ops import apply_tabular_op
        return apply_tabular_op(table, operation_str)