import os
from typing import Any, Dict


class AiraClient:
    """Very small mock AiraClient used for smoke tests.

    Methods mimic the minimal interface used by server.py: .from_env() and .answer(query=...)
    """

    def __init__(self, name: str = "local-mock"):
        self.name = name

    @classmethod
    def from_env(cls) -> "AiraClient":
        # Read optional env var to customize behavior in smoke tests
        name = os.environ.get("AIRA_CLIENT_NAME", "local-mock")
        return cls(name=name)

    def answer(self, query: str) -> Dict[str, Any]:
        # Return a structured response similar to what the real client might return
        # For smoke tests we include a 'sources' field to validate RAG-format responses
        answer_text = f"[aira-mock answer] processed query '{query[:120]}'"
        return {
            "query": query,
            "answer": answer_text,
            "sources": [
                {"file": "sample.pdf", "page": 1, "snippet": "Example snippet"}
            ],
            "client_name": self.name,
        }

    # Generic call support
    def __call__(self, q: str) -> Dict[str, Any]:
        return self.answer(q)
