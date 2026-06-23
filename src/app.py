"""
easy-rag — Gradio Web UI
Run: python src/app.py
Then open http://localhost:7860
"""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from rag_engine import RAGPipeline

# ── Initialize pipeline (reads .env automatically if python-dotenv installed) ─
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROVIDER    = os.getenv("RAG_PROVIDER", "openai")
LLM_MODEL   = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")
STORE_PATH  = os.getenv("RAG_STORE_PATH", "vector_store.json")

rag = RAGPipeline(
    provider=PROVIDER,
    llm_model=LLM_MODEL,
    embed_model=EMBED_MODEL,
    store_path=STORE_PATH,
)

# ── Gradio handlers ───────────────────────────────────────────────────────────
def upload_and_ingest(files):
    if not files:
        return "⚠️ Koi file select nahi ki."
    msgs = []
    for f in files:
        path = Path(f.name)
        if path.suffix.lower() == ".pdf":
            rag.ingest_pdf(str(path))
        else:
            rag.ingest_file(str(path))
        msgs.append(f"✅ {path.name}")
    total = len(rag.store.documents)
    return "\n".join(msgs) + f"\n\n📊 Total chunks in store: {total}"


def ingest_text_block(text, source_name):
    if not text.strip():
        return "⚠️ Text empty hai."
    src = source_name.strip() or "manual-input"
    rag.ingest_text(text, source=src)
    return f"✅ Ingested as '{src}' | Total chunks: {len(rag.store.documents)}"


def answer_question(question, history):
    if not question.strip():
        return history, ""
    result = rag.ask(question)
    ans = result["answer"]
    srcs = ", ".join(result["sources"]) if result["sources"] else "none"
    reply = f"{ans}\n\n📚 *Sources: {srcs}*"
    history = history or []
    history.append((question, reply))
    return history, ""


def show_stats():
    docs = rag.store.documents
    if not docs:
        return "📭 Vector store empty hai. Pehle kuch documents ingest karo."
    from collections import Counter
    sources = Counter(d["metadata"].get("source", "?") for d in docs)
    lines = [f"**Total chunks:** {len(docs)}\n\n**Sources:**"]
    for src, cnt in sources.most_common():
        lines.append(f"- `{src}` → {cnt} chunks")
    return "\n".join(lines)


# ── UI Layout ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="🔍 easy-rag",
    theme=gr.themes.Soft(primary_hue="indigo"),
) as demo:

    gr.Markdown("""
# 🔍 easy-rag
**Apne documents pe AI se poochho — no database, no complexity.**
    """)

    with gr.Tab("📂 Ingest Documents"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Files Upload karo")
                file_input = gr.File(
                    label="Files (.txt, .md, .pdf)",
                    file_types=[".txt", ".md", ".pdf"],
                    file_count="multiple",
                )
                upload_btn = gr.Button("⬆️ Ingest Files", variant="primary")
                upload_status = gr.Textbox(label="Status", lines=4, interactive=False)
                upload_btn.click(upload_and_ingest, inputs=file_input, outputs=upload_status)

            with gr.Column():
                gr.Markdown("### Ya text directly paste karo")
                raw_text = gr.Textbox(label="Text", lines=8, placeholder="Yahan apna text paste karo...")
                source_name = gr.Textbox(label="Source name (optional)", placeholder="e.g. my-notes")
                text_btn = gr.Button("⬆️ Ingest Text", variant="secondary")
                text_status = gr.Textbox(label="Status", interactive=False)
                text_btn.click(ingest_text_block, inputs=[raw_text, source_name], outputs=text_status)

    with gr.Tab("💬 Ask Questions"):
        chatbot = gr.Chatbot(height=420, label="RAG Chat")
        with gr.Row():
            q_input = gr.Textbox(
                placeholder="Apna sawaal yahan likho...",
                label="",
                scale=5,
            )
            ask_btn = gr.Button("Ask 🚀", variant="primary", scale=1)

        ask_btn.click(answer_question, inputs=[q_input, chatbot], outputs=[chatbot, q_input])
        q_input.submit(answer_question, inputs=[q_input, chatbot], outputs=[chatbot, q_input])

    with gr.Tab("📊 Stats"):
        stats_btn = gr.Button("Refresh Stats")
        stats_out = gr.Markdown()
        stats_btn.click(show_stats, outputs=stats_out)
        demo.load(show_stats, outputs=stats_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
