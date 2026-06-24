"""
easy-rag — Beautiful Web UI
Run: python src/app.py
"""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import gradio as gr
from rag_engine import RAGPipeline

PROVIDER    = os.getenv("RAG_PROVIDER", "groq")
LLM_MODEL   = os.getenv("RAG_LLM_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")
STORE_PATH  = os.getenv("RAG_STORE_PATH", "vector_store.json")

rag = RAGPipeline(
    provider=PROVIDER,
    llm_model=LLM_MODEL,
    embed_model=EMBED_MODEL,
    store_path=STORE_PATH,
)

def upload_and_ingest(files):
    if not files:
        return "⚠️ Koi file select nahi ki."
    msgs = []
    for f in files:
        path = Path(f.name)
        try:
            if path.suffix.lower() == ".pdf":
                rag.ingest_pdf(str(path))
            else:
                rag.ingest_file(str(path))
            msgs.append(f"✅ {path.name}")
        except Exception as e:
            msgs.append(f"❌ {path.name}: {str(e)}")
    total = len(rag.store.documents)
    return "\n".join(msgs) + f"\n\n📊 Total chunks in knowledge base: {total}"

def ingest_text_block(text, source_name):
    if not text.strip():
        return "⚠️ Text empty hai."
    src = source_name.strip() or "manual-input"
    rag.ingest_text(text, source=src)
    return f"✅ '{src}' se {len(rag.store.documents)} chunks load hue"

def answer_question(question, history):
    if not question.strip():
        return history, ""
    if not rag.store.documents:
        history = history or []
        history.append((question, "⚠️ Pehle koi document upload karo — phir sawaal poochho!"))
        return history, ""
    result = rag.ask(question)
    srcs = ", ".join(f"`{s}`" for s in result["sources"]) if result["sources"] else "none"
    reply = f"{result['answer']}\n\n---\n📚 **Sources:** {srcs}"
    history = history or []
    history.append((question, reply))
    return history, ""

def show_stats():
    docs = rag.store.documents
    if not docs:
        return "📭 Knowledge base empty hai.\nPehle **Documents** tab mein kuch upload karo!"
    from collections import Counter
    sources = Counter(d["metadata"].get("source", "?") for d in docs)
    lines = [f"**🗂️ Total Chunks:** {len(docs)}\n**📁 Loaded Sources:**\n"]
    for src, cnt in sources.most_common():
        lines.append(f"- 📄 `{src}` — {cnt} chunks")
    lines.append(f"\n**🤖 Provider:** `{PROVIDER}` | **Model:** `{LLM_MODEL}`")
    return "\n".join(lines)

# ── Custom CSS ────────────────────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #0a0e1a !important;
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 16px !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 40px 20px 20px;
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1729 100%);
    border-bottom: 1px solid #1e2d4a;
    margin-bottom: 8px;
}

.app-header h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 0 8px !important;
    letter-spacing: -0.5px;
}

.app-header p {
    color: #64748b !important;
    font-size: 1rem !important;
    margin: 0 !important;
}

.provider-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1e2d4a, #162032);
    border: 1px solid #6366f1;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #818cf8;
    margin-top: 10px;
    font-family: 'Space Grotesk', sans-serif;
}

/* Tabs */
.tab-nav {
    background: #0d1424 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 12px !important;
    padding: 4px !important;
    margin-bottom: 16px !important;
}

.tab-nav button {
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: #64748b !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    border: none !important;
    background: transparent !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
}

/* Cards */
.card {
    background: #0d1424 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

/* Inputs */
textarea, input[type="text"] {
    background: #0d1424 !important;
    border: 1px solid #1e2d4a !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    outline: none !important;
}

/* Buttons */
button.primary {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 12px 24px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}

button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
}

button.secondary {
    background: #1e2d4a !important;
    border: 1px solid #2d3f5e !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s !important;
}

button.secondary:hover {
    border-color: #6366f1 !important;
    color: #818cf8 !important;
}

/* Chatbot */
.chatbot {
    background: #0d1424 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 16px !important;
}

.chatbot .message.user {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 12px 16px !important;
    margin: 4px 0 !important;
}

.chatbot .message.bot {
    background: #162032 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 12px 16px !important;
    margin: 4px 0 !important;
}

/* File upload */
.upload-btn {
    background: #0d1424 !important;
    border: 2px dashed #2d3f5e !important;
    border-radius: 12px !important;
    color: #64748b !important;
    transition: all 0.2s !important;
}

.upload-btn:hover {
    border-color: #6366f1 !important;
    color: #818cf8 !important;
    background: #0f1729 !important;
}

/* Labels */
label span {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Status box */
.status-box textarea {
    background: #060b14 !important;
    border-color: #1e2d4a !important;
    color: #4ade80 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* Section titles */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Divider */
hr { border-color: #1e2d4a !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }

/* Stats markdown */
.stats-area p, .stats-area li { color: #94a3b8 !important; }
.stats-area strong { color: #818cf8 !important; }
.stats-area code {
    background: #1e2d4a !important;
    color: #06b6d4 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
}
"""

# ── Header HTML ───────────────────────────────────────────────────────────────
header_html = f"""
<div class="app-header">
    <h1>🔍 easy-rag</h1>
    <p>Apne documents pe AI se poochho — instant, accurate, hallucination-free</p>
    <div class="provider-badge">⚡ {PROVIDER.upper()} · {LLM_MODEL}</div>
</div>
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="🔍 easy-rag") as demo:

    gr.HTML(header_html)

    with gr.Tabs() as tabs:

        # ── TAB 1: Documents ──────────────────────────────────────────────────
        with gr.Tab("📂  Documents"):
            gr.HTML("<div style='height:12px'></div>")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-title'>📎 Files Upload karo</div>")
                    file_input = gr.File(
                        label="Drag & Drop karo ya click karo (.txt .md .pdf)",
                        file_types=[".txt", ".md", ".pdf"],
                        file_count="multiple",
                        elem_classes=["upload-btn"],
                    )
                    upload_btn = gr.Button("⬆️  Ingest Files", variant="primary")
                    upload_status = gr.Textbox(
                        label="Status",
                        lines=5,
                        interactive=False,
                        placeholder="Files upload karoge to yahan status dikhega...",
                        elem_classes=["status-box"],
                    )
                    upload_btn.click(upload_and_ingest, inputs=file_input, outputs=upload_status)

                with gr.Column(scale=1):
                    gr.HTML("<div class='section-title'>✏️  Ya Text Directly Paste karo</div>")
                    raw_text = gr.Textbox(
                        label="Text Content",
                        lines=7,
                        placeholder="Yahan apna content paste karo — notes, FAQs, policies, kuch bhi...",
                    )
                    source_name = gr.Textbox(
                        label="Source Name (optional)",
                        placeholder="e.g. physics-notes, hr-policy",
                    )
                    text_btn = gr.Button("⬆️  Ingest Text", variant="secondary")
                    text_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        placeholder="Status yahan dikhega...",
                        elem_classes=["status-box"],
                    )
                    text_btn.click(ingest_text_block, inputs=[raw_text, source_name], outputs=text_status)

        # ── TAB 2: Chat ───────────────────────────────────────────────────────
        with gr.Tab("💬  Ask AI"):
            gr.HTML("<div style='height:12px'></div>")
            chatbot = gr.Chatbot(
                height=460,
                label="",
                placeholder="<div style='text-align:center;color:#334155;padding:60px 20px'><div style='font-size:2.5rem'>🔍</div><div style='font-size:1.1rem;margin-top:8px;font-family:Space Grotesk'>Documents upload karo aur sawaal poochho</div><div style='font-size:0.85rem;margin-top:6px;color:#475569'>AI tumhare apne documents se jawab dega</div></div>",
                show_label=False,
            )
            with gr.Row():
                q_input = gr.Textbox(
                    placeholder="💭 Apna sawaal yahan likho... (Enter dabao ya button click karo)",
                    label="",
                    scale=5,
                    lines=1,
                )
                ask_btn = gr.Button("Ask  🚀", variant="primary", scale=1, min_width=100)

            ask_btn.click(answer_question, inputs=[q_input, chatbot], outputs=[chatbot, q_input])
            q_input.submit(answer_question, inputs=[q_input, chatbot], outputs=[chatbot, q_input])

            gr.HTML("""
            <div style='margin-top:12px;padding:12px 16px;background:#0d1424;border:1px solid #1e2d4a;border-radius:10px;'>
                <span style='color:#475569;font-size:0.8rem'>💡 <strong style='color:#6366f1'>Tips:</strong>
                Specific sawaal poochho &nbsp;·&nbsp;
                Document ka naam mention karo &nbsp;·&nbsp;
                Hindi ya English dono mein poochh sakte ho</span>
            </div>
            """)

        # ── TAB 3: Stats ──────────────────────────────────────────────────────
        with gr.Tab("📊  Stats"):
            gr.HTML("<div style='height:12px'></div>")
            stats_out = gr.Markdown(elem_classes=["stats-area"])
            refresh_btn = gr.Button("🔄  Refresh", variant="secondary")
            refresh_btn.click(show_stats, outputs=stats_out)
            demo.load(show_stats, outputs=stats_out)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=custom_css,
    )
