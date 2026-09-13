"""Example adapter. This is the whole integration surface:

    class MyRetriever:
        def retrieve(self, query: str, k: int) -> list[str]: ...

Run it with:

    ragbisect run --corpus ./docs --adapter examples/my_pipeline.py:MyRetriever

If the class constructor takes one positional argument, ragbisect passes the
chunk list (``list[ragbisect.Chunk]`` with ``.id``, ``.text``, ``.source``) so
you can build an index over exactly the chunks the eval set refers to. If it
takes none, ragbisect calls it with no arguments and you own the index.
"""

from __future__ import annotations

from ragbisect import Chunk
from ragbisect.retrievers import BM25Retriever


class MyRetriever:
    """A deliberately naive pipeline: BM25 only. Swap the body for your own
    vector store / LangChain / LlamaIndex call and return chunk IDs."""

    def __init__(self, chunks: list[Chunk]):
        self._chunks = {c.id: c.text for c in chunks}
        self._index = BM25Retriever(chunks)

    def retrieve(self, query: str, k: int) -> list[str]:
        return self._index.retrieve(query, k)

    # Optional. Only add this if you want generation (faithfulness) scored.
    # def generate(self, query: str, chunk_ids: list[str]) -> str:
    #     context = "\n\n".join(self._chunks[c] for c in chunk_ids)
    #     return my_llm(f"Answer from the context only.\n\n{context}\n\nQ: {query}")


# --- Sketch for a LangChain vector store -------------------------------------
# class LangChainRetriever:
#     def __init__(self):
#         self.store = Chroma(persist_directory="./index", embedding_function=OpenAIEmbeddings())
#     def retrieve(self, query, k):
#         docs = self.store.similarity_search(query, k=k)
#         return [d.metadata["chunk_id"] for d in docs]   # IDs must match your chunks.jsonl
