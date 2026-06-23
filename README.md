# 🔍 easy-rag

> **Apne documents pe AI se poochho — no database setup, no boilerplate, bas 4 lines.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange)](https://openai.com)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-purple)](https://anthropic.com)

**easy-rag** ek zero-friction RAG (Retrieval-Augmented Generation) starter kit hai.  
Kisi bhi document ko ingest karo, questions poochho — hallucination-free answers pao.

---

🌐 **Language / भाषा चुनें:**  
**[🇮🇳 हिंदी में पढ़ें](#-hindi-documentation-हिंदी-दस्तावेज़)** &nbsp;|&nbsp; [🇬🇧 English](#-english-documentation)

---

---

# 🇬🇧 English Documentation

## ✨ Features

| Feature | Description |
|---|---|
| 🚀 **4-line quickstart** | Ingest → Ask — that's it |
| 📄 **Multi-format** | `.txt`, `.md`, `.pdf` support |
| 🤖 **Multi-provider** | OpenAI **or** Anthropic Claude |
| 💾 **No DB needed** | JSON-based vector store (swap-ready) |
| 🖥️ **Web UI** | Gradio app included |
| 💬 **CLI** | Interactive chat + one-shot Q&A |
| 📓 **Notebook** | Step-by-step Jupyter tutorial |
| 🔌 **Extensible** | Easy to plug in Chroma, Pinecone, FAISS |

---

## ⚡ Quickstart (5 minutes)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/easy-rag.git
cd easy-rag
pip install -r requirements.txt
```

### 2. Set API Key

```bash
cp .env.example .env
# Add your OPENAI_API_KEY in the .env file
```

### 3. Run!

```python
from src.rag_engine import RAGPipeline

rag = RAGPipeline(provider='openai')
rag.ingest_file('data/sample_docs/what_is_rag.txt')

result = rag.ask('What are the main use cases of RAG?')
print(result['answer'])
print('Sources:', result['sources'])
```

---

## 🏗️ Repo Structure

```
easy-rag/
├── src/
│   ├── rag_engine.py        ← Core RAG logic (embed, store, retrieve, generate)
│   ├── cli.py               ← Command-line interface
│   └── app.py               ← Gradio web UI
├── notebooks/
│   └── 01_quickstart.ipynb  ← Interactive tutorial
├── data/
│   └── sample_docs/         ← Drop your documents here
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔧 Usage

### Python API

```python
from src.rag_engine import RAGPipeline

rag = RAGPipeline(provider='openai')  # or 'anthropic'

# Ingest
rag.ingest_text("Your content here...", source="my-notes")
rag.ingest_file("report.pdf")
rag.ingest_folder("data/docs/")

# Ask
result = rag.ask("Your question here?")
print(result['answer'])
print(result['sources'])

# Custom system prompt
result = rag.ask("Tell me about refund policy",
    system_prompt="You are a customer support agent. Be concise.")

# Retrieve only (no LLM call)
chunks = rag.retrieve("search query")
```

### CLI

```bash
python src/cli.py ingest --file data/sample_docs/what_is_rag.txt
python src/cli.py ingest --folder data/my_docs --extensions .txt,.md,.pdf
python src/cli.py ask "What is RAG?"
python src/cli.py chat       # Interactive mode
python src/cli.py stats      # View store statistics
```

### Web UI (Gradio)

```bash
python src/app.py
# Open in browser: http://localhost:7860
```

---

## ⚙️ Configuration (.env)

```env
RAG_PROVIDER=openai          # openai OR anthropic
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...        # Required for Anthropic provider (Voyage embeddings)
RAG_LLM_MODEL=gpt-4o-mini
RAG_EMBED_MODEL=text-embedding-3-small
RAG_STORE_PATH=vector_store.json
```

---

## 🧠 RAG Architecture

```
Documents → Chunk → Embed → Vector Store
                                │
Question → Embed → Similarity Search → Top-K Chunks
                                              │
                                    LLM (Context + Question)
                                              │
                                        Final Answer ✅
```

---

## 🔄 Production Upgrade — Chroma (easy swap)

```python
# pip install chromadb
import chromadb, hashlib

class ChromaVectorStore:
    def __init__(self):
        self.col = chromadb.Client().get_or_create_collection("easy-rag")

    def add(self, text, embedding, metadata={}):
        uid = hashlib.md5(text.encode()).hexdigest()
        self.col.add(ids=[uid], embeddings=[embedding],
                     documents=[text], metadatas=[metadata])

    def search(self, query_embedding, top_k=5):
        r = self.col.query(query_embeddings=[query_embedding], n_results=top_k)
        return [{"text": d, "metadata": m}
                for d, m in zip(r["documents"][0], r["metadatas"][0])]

# Use it:
rag = RAGPipeline(provider='openai')
rag.store = ChromaVectorStore()
```

---

## 🤝 Contributing

PRs welcome! Open issues for:
- FAISS / Pinecone adapters
- URL / web scraping ingestion
- Streaming responses
- Docker support

---

## 📄 License

MIT — Free to use, modify, and distribute.

**Built with ❤️ for anyone who wants RAG without the headache.**

---
---

# 🇮🇳 Hindi Documentation (हिंदी दस्तावेज़)

> **अपने documents पर AI से पूछो — कोई database setup नहीं, कोई boilerplate नहीं, बस 4 lines।**

**easy-rag** एक simple RAG (Retrieval-Augmented Generation) starter kit है।  
किसी भी document को load करो, सवाल पूछो — सटीक जवाब पाओ बिना hallucination के।

---

## ✨ क्या-क्या मिलेगा

| Feature | विवरण |
|---|---|
| 🚀 **4-line quickstart** | Ingest करो → पूछो — बस इतना |
| 📄 **Multi-format support** | `.txt`, `.md`, `.pdf` सब चलता है |
| 🤖 **Multi-provider** | OpenAI **या** Anthropic Claude — दोनों |
| 💾 **कोई DB नहीं चाहिए** | JSON-based vector store (production में swap करो) |
| 🖥️ **Web UI** | Gradio app included है |
| 💬 **CLI tool** | Terminal से chat करो |
| 📓 **Notebook** | Step-by-step Jupyter tutorial |
| 🔌 **Extensible** | Chroma, Pinecone, FAISS आसानी से लगाओ |

---

## ⚡ शुरुआत करो (5 मिनट में)

### Step 1: Clone और Install करो

```bash
git clone https://github.com/YOUR_USERNAME/easy-rag.git
cd easy-rag
pip install -r requirements.txt
```

### Step 2: API Key डालो

```bash
cp .env.example .env
# .env file खोलो और OPENAI_API_KEY डालो
```

### Step 3: चलाओ!

```python
from src.rag_engine import RAGPipeline

# Pipeline बनाओ
rag = RAGPipeline(provider='openai')

# अपना document load करो
rag.ingest_file('data/sample_docs/what_is_rag.txt')

# सवाल पूछो
result = rag.ask('RAG के main use cases क्या हैं?')
print(result['answer'])
print('Sources:', result['sources'])
```

**Output कुछ ऐसा आएगा:**
```
🤖 RAG के common use cases में customer support chatbots, HR Q&A,
   legal document search, और medical literature retrieval शामिल हैं।

📚 Sources: what_is_rag.txt
```

---

## 🏗️ Folder Structure

```
easy-rag/
├── src/
│   ├── rag_engine.py        ← मुख्य RAG logic (embed, store, retrieve, generate)
│   ├── cli.py               ← Command-line tool
│   └── app.py               ← Gradio web UI
├── notebooks/
│   └── 01_quickstart.ipynb  ← Interactive tutorial (यहाँ से शुरू करो)
├── data/
│   └── sample_docs/         ← अपने documents यहाँ डालो
├── requirements.txt          ← सभी dependencies
├── .env.example              ← API keys का template
└── README.md
```

---

## 🔧 उपयोग कैसे करें

### Python API (Code में use करो)

```python
from src.rag_engine import RAGPipeline

# OpenAI या Anthropic — जो चाहो
rag = RAGPipeline(provider='openai')   # या provider='anthropic'

# — Documents Load करो —
rag.ingest_text("यहाँ अपना text डालो...", source="meri-notes")
rag.ingest_file("report.pdf")           # .txt / .md / .pdf
rag.ingest_folder("data/docs/")         # पूरा folder एक बार में

# — सवाल पूछो —
result = rag.ask("अपना सवाल यहाँ?")
print(result['answer'])     # AI का जवाब
print(result['sources'])    # किस file से आया
print(result['chunks'])     # Retrieved chunks (debugging के लिए)

# — Custom System Prompt —
result = rag.ask(
    "Refund policy क्या है?",
    system_prompt="तुम एक helpful customer support agent हो। Hindi में जवाब दो।"
)

# — सिर्फ Search (LLM call नहीं) —
chunks = rag.retrieve("search query")
```

### CLI (Terminal से)

```bash
# File ingest करो
python src/cli.py ingest --file data/sample_docs/what_is_rag.txt

# पूरा folder ingest करो
python src/cli.py ingest --folder data/my_docs --extensions .txt,.md,.pdf

# एक सवाल पूछो
python src/cli.py ask "RAG क्या होता है?"

# Interactive chat mode (बातचीत करो)
python src/cli.py chat

# Store में कितने documents हैं देखो
python src/cli.py stats
```

### Web UI (Browser में)

```bash
python src/app.py
# Browser में खोलो: http://localhost:7860
```

Web UI में तीन tabs मिलेंगे:
- **📂 Ingest** — Files upload करो या text paste करो
- **💬 Ask** — Chat interface से सवाल पूछो
- **📊 Stats** — कितने documents load हैं देखो

---

## ⚙️ Configuration (.env file)

`.env.example` को copy करो और अपनी keys डालो:

```env
# Provider चुनो: openai या anthropic
RAG_PROVIDER=openai

# OpenAI के लिए
OPENAI_API_KEY=sk-...

# Anthropic के लिए (दोनों keys चाहिए)
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...        # Voyage AI embeddings के लिए

# Model override (optional — default अच्छे हैं)
# RAG_LLM_MODEL=gpt-4o-mini
# RAG_EMBED_MODEL=text-embedding-3-small

# Vector store कहाँ save हो
RAG_STORE_PATH=vector_store.json
```

---

## 🧠 RAG कैसे काम करता है

```
तुम्हारे Documents
       │
       ▼
┌──────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Chunker    │───▶│  Embedding Model│───▶│  Vector Store   │
│ (500 words)  │    │ (OpenAI/Voyage) │    │ (JSON/Chroma/   │
└──────────────┘    └─────────────────┘    │  Pinecone)      │
                                           └─────────────────┘
                                                   │
User का सवाल ──▶ Embed ──▶ Similarity Search
                                                   │
                                          Top-K Chunks मिले
                                                   │
                                                   ▼
                                     ┌─────────────────────┐
                                     │  LLM (GPT / Claude) │
                                     │  Context + Question │
                                     └─────────────────────┘
                                                   │
                                            Final Answer ✅
```

**3 Simple Steps:**

**Step 1 — Indexing** (एक बार होता है)
- Documents को छोटे chunks में तोड़ो
- हर chunk को numbers (embedding) में convert करो
- उन numbers को vector store में save करो

**Step 2 — Retrieval** (हर सवाल पर)
- User का सवाल embedding में convert करो
- Vector store में search करो — similar chunks ढूंढो
- Top-K relevant chunks निकालो

**Step 3 — Generation** (हर सवाल पर)
- Retrieved chunks को LLM को context दो
- LLM सवाल + context देखकर accurate जवाब देता है
- Hallucination नहीं होती क्योंकि real documents पर based है

---

## 🔄 Production के लिए Upgrade करो

### Chroma (Local, Free)

```python
# pip install chromadb
import chromadb, hashlib

class ChromaVectorStore:
    def __init__(self):
        self.col = chromadb.Client().get_or_create_collection("easy-rag")

    def add(self, text, embedding, metadata={}):
        uid = hashlib.md5(text.encode()).hexdigest()
        self.col.add(ids=[uid], embeddings=[embedding],
                     documents=[text], metadatas=[metadata])

    def search(self, query_embedding, top_k=5):
        r = self.col.query(query_embeddings=[query_embedding], n_results=top_k)
        return [{"text": d, "metadata": m}
                for d, m in zip(r["documents"][0], r["metadatas"][0])]

# Use करो:
rag = RAGPipeline(provider='openai')
rag.store = ChromaVectorStore()   # बस यह line बदलो!
```

### Provider Comparison

| | OpenAI | Anthropic |
|---|---|---|
| **LLM** | gpt-4o-mini / gpt-4o | claude-3-5-haiku / claude-sonnet |
| **Embeddings** | text-embedding-3-small | voyage-3 (Voyage AI) |
| **Setup** | 1 key | 2 keys (Anthropic + Voyage) |
| **Cost** | कम | कम |

---

## 🤝 Contribute करो

PRs welcome हैं! इन पर काम हो सकता है:
- [ ] FAISS / Pinecone adapters
- [ ] URL / web scraping से ingest
- [ ] Streaming responses
- [ ] Docker container
- [ ] Hindi/multilingual support

---

## 📄 License

MIT License — Free है, use करो, modify करो, distribute करो।

**किसी भी RAG headache के बिना बनाया गया — ❤️ के साथ।**

---

[⬆️ ऊपर जाओ](#-easy-rag)
