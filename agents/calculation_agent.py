class CalculationAgent:
    """Agent for performing numeric calculations."""

    def compute(self, expression):
        """Return the evaluated result of an expression."""
        try:
            return eval(expression)
        except Exception:
            return None