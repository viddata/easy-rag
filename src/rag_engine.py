"""
easy-rag — Core RAG Engine
Works with OpenAI, Anthropic, Groq, or any OpenAI-compatible API.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

# ── Vector store (in-memory, no external DB needed to start) ──────────────────
class SimpleVectorStore:
    """
    Lightweight in-memory vector store.
    Swap this with Chroma / Pinecone / Weaviate in production.
    """
    def __init__(self):
        self.documents: List[Dict] = []   # [{text, embedding, metadata}]

    def add(self, text: str, embedding: List[float], metadata: dict = {}):
        self.documents.append({
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
        })

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Cosine similarity search."""
        if not self.documents:
            return []

        import numpy as np
        q = np.array(query_embedding)
        scores = []
        for doc in self.documents:
            d = np.array(doc["embedding"])
            score = float(np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d) + 1e-10))
            scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scores[:top_k]]

    def save(self, path: str):
        """Persist to disk as JSON."""
        with open(path, "w") as f:
            json.dump(self.documents, f)
        print(f"✅ Vector store saved → {path}")

    def load(self, path: str):
        """Load from disk."""
        if Path(path).exists():
            with open(path) as f:
                self.documents = json.load(f)
            print(f"✅ Loaded {len(self.documents)} chunks from {path}")
        else:
            print(f"⚠️  No existing store at {path}. Starting fresh.")


# ── Text Chunker ──────────────────────────────────────────────────────────────
class TextChunker:
    """Split long documents into overlapping chunks."""

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict = {}) -> List[Dict]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append({
                "text": chunk_text,
                "metadata": {**metadata, "chunk_index": len(chunks)},
            })
            start += self.chunk_size - self.overlap
        return chunks


# ── RAG Pipeline ──────────────────────────────────────────────────────────────
class RAGPipeline:
    """
    End-to-end RAG: ingest → embed → retrieve → generate.

    Supports:
      - provider="openai"     (gpt-4o, text-embedding-3-small)
      - provider="anthropic"  (claude-3-5-haiku, voyage-3 embeddings)
      - provider="groq"       (llama-3.3-70b-versatile, FREE + fast)
    """

    def __init__(
        self,
        provider: str = "groq",
        llm_model: str = "llama-3.3-70b-versatile",
        embed_model: str = "text-embedding-3-small",
        store_path: str = "vector_store.json",
        chunk_size: int = 500,
        overlap: int = 100,
        top_k: int = 5,
    ):
        self.provider = provider
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.store_path = store_path
        self.top_k = top_k

        self.store = SimpleVectorStore()
        self.store.load(store_path)
        self.chunker = TextChunker(chunk_size, overlap)

        self._setup_client()

    # ── Client setup ──────────────────────────────────────────────────────────
    def _setup_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "groq":
            # Groq uses OpenAI-compatible SDK — FREE tier available
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            )
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Use 'openai', 'groq', or 'anthropic'.")

    # ── Embedding ─────────────────────────────────────────────────────────────
    def embed(self, text: str) -> List[float]:
        if self.provider in ("openai", "groq"):
            # Groq doesn't have embeddings yet — use a free local alternative
            if self.provider == "groq":
                return self._embed_local(text)
            resp = self.client.embeddings.create(input=text, model=self.embed_model)
            return resp.data[0].embedding
        elif self.provider == "anthropic":
            # Voyage embeddings via Anthropic SDK
            import voyageai
            vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
            result = vo.embed([text], model=self.embed_model)
            return result.embeddings[0]

    def _embed_local(self, text: str) -> List[float]:
        """
        Free local embeddings using sentence-transformers.
        Used automatically when provider='groq' (Groq has no embedding API).
        Install: pip install sentence-transformers
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Groq provider ke liye sentence-transformers chahiye.\n"
                "Run karo: pip install sentence-transformers"
            )
        if not hasattr(self, "_st_model"):
            print("⏳ Embedding model load ho raha hai (pehli baar thoda time lagega)...")
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._st_model.encode(text).tolist()

    # ── Ingest ────────────────────────────────────────────────────────────────
    def ingest_text(self, text: str, source: str = "unknown"):
        """Ingest a raw string."""
        chunks = self.chunker.chunk(text, metadata={"source": source})
        for chunk in chunks:
            emb = self.embed(chunk["text"])
            self.store.add(chunk["text"], emb, chunk["metadata"])
        self.store.save(self.store_path)
        print(f"✅ Ingested {len(chunks)} chunks from '{source}'")

    def ingest_file(self, filepath: str):
        """Ingest a .txt or .md file."""
        path = Path(filepath)
        text = path.read_text(encoding="utf-8")
        self.ingest_text(text, source=path.name)

    def ingest_folder(self, folder: str, extensions: List[str] = [".txt", ".md"]):
        """Ingest all matching files in a folder."""
        folder_path = Path(folder)
        files = [f for ext in extensions for f in folder_path.glob(f"*{ext}")]
        print(f"📂 Found {len(files)} files in '{folder}'")
        for f in files:
            self.ingest_file(str(f))

    def ingest_pdf(self, filepath: str):
        """Ingest a PDF file (requires pypdf)."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("Install pypdf: pip install pypdf")
        reader = PdfReader(filepath)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        self.ingest_text(text, source=Path(filepath).name)

    # ── Retrieve ──────────────────────────────────────────────────────────────
    def retrieve(self, query: str) -> List[Dict]:
        """Return top-k relevant chunks for a query."""
        q_emb = self.embed(query)
        return self.store.search(q_emb, top_k=self.top_k)

    # ── Generate ──────────────────────────────────────────────────────────────
    def _build_context(self, chunks: List[Dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            src = c["metadata"].get("source", "?")
            parts.append(f"[{i}] (source: {src})\n{c['text']}")
        return "\n\n---\n\n".join(parts)

    def ask(self, question: str, system_prompt: Optional[str] = None) -> Dict:
        """
        Full RAG pipeline: retrieve + generate.
        Returns: {answer, sources, chunks}
        """
        chunks = self.retrieve(question)

        if not chunks:
            return {
                "answer": "Mujhe koi relevant information nahi mili. Pehle kuch documents ingest karo.",
                "sources": [],
                "chunks": [],
            }

        context = self._build_context(chunks)
        sys_prompt = system_prompt or (
            "You are a helpful assistant. Answer the user's question ONLY using the provided context. "
            "If the answer is not in the context, say 'I don't know based on the provided documents.' "
            "Always cite which source [1], [2], etc. you used."
        )

        user_msg = f"Context:\n{context}\n\nQuestion: {question}"

        if self.provider in ("openai", "groq"):
            resp = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            answer = resp.choices[0].message.content

        elif self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.llm_model,
                max_tokens=1024,
                system=sys_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            answer = resp.content[0].text

        sources = list({c["metadata"].get("source", "?") for c in chunks})
        return {"answer": answer, "sources": sources, "chunks": chunks}
