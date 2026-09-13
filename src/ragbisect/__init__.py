"""ragbisect — stage-by-stage diagnostics for RAG pipelines.

The only thing you must implement to use ragbisect is the ``Retriever`` protocol:

    class MyRetriever:
        def retrieve(self, query: str, k: int) -> list[str]:
            '''Return chunk IDs, most relevant first.'''

Optionally, add ``generate(query, chunk_ids) -> str`` to have generation scored.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .corpus import Chunk

__version__ = "0.1.0"


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[str]:
        """Return chunk IDs, most relevant first."""
        ...


@runtime_checkable
class Generator(Protocol):
    def generate(self, query: str, chunk_ids: list[str]) -> str:
        """Answer the query using only the given chunks."""
        ...


__all__ = ["Chunk", "Retriever", "Generator", "__version__"]
