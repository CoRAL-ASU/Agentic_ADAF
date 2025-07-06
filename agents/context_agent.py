class ContextAgent:
    """Agent for retrieving textual context."""

    def __init__(self, model=None):
        self.model = model

    def fetch_relevant_text(self, question):
        """Return placeholder context for a question."""
        return ""