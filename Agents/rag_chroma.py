from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path
from typing import List

import chromadb
import ollama


class OllamaEmbeddingFunction:
    def __init__(self, *, model: str) -> None:
        self._model = model

    def name(self) -> str:
        return f"ollama:{self._model}"

    def __call__(self, input: List[str]) -> List[List[float]]:  # Chroma expects this signature
        out: List[List[float]] = []
        for t in input:
            r = ollama.embeddings(model=self._model, prompt=t)
            out.append(r["embedding"])
        return out


class ChromaChatMemory:
    """
    Persistent, local chat-memory RAG using ChromaDB + Ollama embeddings.
    Stores each (user, assistant) turn and can retrieve top-k relevant turns.
    """

    def __init__(
        self,
        *,
        persist_dir: str | None = None,
        collection_name: str = "chat_memory",
        embedding_model: str | None = None,
    ) -> None:
        base = Path(persist_dir) if persist_dir else Path(__file__).resolve().parent.parent / "chroma_db"
        base.mkdir(parents=True, exist_ok=True)

        self._persist_dir = base
        self._client = chromadb.PersistentClient(path=str(base))

        # Default to a common local embedding model; override via env var if desired.
        self._embed_model = (
            embedding_model
            or os.environ.get("OLLAMA_EMBED_MODEL")
            or "nomic-embed-text"
        )

        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=OllamaEmbeddingFunction(model=self._embed_model),
            )
        except KeyError:
            # Chroma occasionally fails to load older persisted configs (e.g. missing "_type").
            # Auto-recover by backing up the old directory and creating a fresh store.
            backup = base.with_name(f"{base.name}_backup_{int(time.time())}")
            try:
                shutil.move(str(base), str(backup))
            except Exception:
                pass
            base.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(base))
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=OllamaEmbeddingFunction(model=self._embed_model),
            )

    def add_turn(self, *, user_text: str, assistant_text: str) -> None:
        doc = f"User: {user_text}\nAssistant: {assistant_text}"
        self._collection.add(
            ids=[uuid.uuid4().hex],
            documents=[doc],
            metadatas=[
                {
                    "ts": time.time(),
                    "user_text": user_text or "",
                    "assistant_text": assistant_text or "",
                }
            ],
        )

    def retrieve(self, *, query: str, k: int = 5) -> List[dict]:
        if not query:
            return []
        res = self._collection.query(query_texts=[query], n_results=max(1, int(k)))
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: List[dict] = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append(
                {
                    "distance": dist,
                    "document": doc,
                    "user_text": (meta or {}).get("user_text", ""),
                    "assistant_text": (meta or {}).get("assistant_text", ""),
                    "ts": (meta or {}).get("ts"),
                }
            )
        return out

    def format_retrieval(self, *, query: str, k: int = 5) -> str:
        hits = self.retrieve(query=query, k=k)
        if not hits:
            return "No relevant prior conversation found."

        # Note: Chroma returns distances; lower is better (for cosine distance).
        parts = ["Relevant prior conversation snippets:"]
        for i, h in enumerate(hits, 1):
            parts.append(f"\n[{i}] (distance={h['distance']:.4f})")
            parts.append(h["document"])
        return "\n".join(parts)


_DEFAULT_MEMORY: ChromaChatMemory | None = None


def get_memory() -> ChromaChatMemory:
    global _DEFAULT_MEMORY
    if _DEFAULT_MEMORY is None:
        _DEFAULT_MEMORY = ChromaChatMemory()
    return _DEFAULT_MEMORY

