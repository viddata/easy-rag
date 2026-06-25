"""
easy-rag — Single Page UI
Upload + Chat ek hi page pe
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
        return "⚠️ Koi file select nahi ki.", get_stats()
    msgs = []
    for f in files:
        path = Path(f.name)
        try:
            if path.suffix.lower() == ".pdf":
                rag.ingest_pdf(str(path))
            else:
                rag.ingest_file(str(path))
            msgs.append(f"✅ {path.name} — successfully loaded!")
        except Exception as e:
            msgs.append(f"❌ {path.name}: {str(e)}")
    status = "\n".join(msgs)
    status += f"\n\n📊 Total chunks in knowledge base: {len(rag.store.documents)}"
    status += "\n\n💬 Ab neeche sawaal poochho!"
    return status, get_stats()

def get_stats():
    docs = rag.store.documents
    if not docs:
        return "📭 Koi document load nahi hua abhi"
    from collections import Counter
    sources = Counter(d["metadata"].get("source", "?") for d in docs)
    lines = [f"📊 **{len(docs)} chunks** loaded\n"]
    for src, cnt in sources.most_common():
        lines.append(f"📄 `{src}` — {cnt} chunks")
    return "\n".join(lines)

def answer_question(question, history):
    if not question.strip():
        return history, ""
    if not rag.store.documents:
        history = history or []
        history.append({
            "role": "user", "content": question
        })
        history.append({
            "role": "assistant",
            "content": "⚠️ Pehle upar koi document upload karo — phir sawaal poochho!"
        })
        return history, ""
    result = rag.ask(question)
    srcs = ", ".join(f"`{s}`" for s in result["sources"]) if result["sources"] else "none"
    reply = f"{result['answer']}\n\n---\n📚 **Sources:** {srcs}"
    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": reply})
    return history, ""

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #0a0e1a !important;
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 16px 40px !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 36px 20px 24px;
    border-bottom: 1px solid #1e2d4a;
    margin-bottom: 24px;
}
.app-header h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 0 6px !important;
}
.app-header p {
    color: #64748b !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}
.provider-badge {
    display: inline-block;
    background: #1e2d4a;
    border: 1px solid #6366f1;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #818cf8;
    margin-top: 10px;
}

/* Panel headings */
.panel-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2d4a;
}

/* Inputs */
textarea, input[type="text"] {
    background: #0d1424 !important;
    border: 1px solid #1e2d4a !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
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
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
    transition: all 0.2s !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
}
button.secondary {
    background: #1e2d4a !important;
    border: 1px solid #2d3f5e !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Status output */
.status-out textarea {
    background: #060b14 !important;
    border: 1px solid #1e2d4a !important;
    color: #4ade80 !important;
    font-size: 0.83rem !important;
    font-family: monospace !important;
}

/* Stats markdown */
.stats-out p, .stats-out li { color: #64748b !important; font-size:0.85rem !important; }
.stats-out strong { color: #818cf8 !important; }
.stats-out code {
    background: #1e2d4a !important;
    color: #06b6d4 !important;
    padding: 1px 5px !important;
    border-radius: 4px !important;
    font-size: 0.8rem !important;
}

/* Chatbot */
.chatbot {
    background: #0d1424 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 14px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }

label span { color: #64748b !important; font-size: 0.82rem !important; font-weight: 500 !important; }

.divider { border: none; border-top: 1px solid #1e2d4a; margin: 0; }
"""

header_html = f"""
<div class="app-header">
    <h1>🔍 easy-rag</h1>
    <p>Apne documents pe AI se poochho — instant, accurate, hallucination-free</p>
    <div class="provider-badge">⚡ {PROVIDER.upper()} · {LLM_MODEL}</div>
</div>
"""

with gr.Blocks(title="🔍 easy-rag") as demo:

    gr.HTML(header_html)

    with gr.Row(equal_height=False):

        # ── LEFT: Upload Panel ────────────────────────────────────────────────
        with gr.Column(scale=4, min_width=300):
            gr.HTML("<div class='panel-title'>📂 Documents</div>")

            file_input = gr.File(
                label="PDF, TXT ya MD file upload karo",
                file_types=[".txt", ".md", ".pdf"],
                file_count="multiple",
            )
            upload_btn = gr.Button("⬆️  Ingest karo", variant="primary")

            upload_status = gr.Textbox(
                label="Upload Status",
                lines=4,
                interactive=False,
                placeholder="Files choose karo aur 'Ingest karo' dabao...",
                elem_classes=["status-out"],
            )

            gr.HTML("<hr class='divider' style='margin:16px 0'>")

            stats_out = gr.Markdown(
                value=get_stats(),
                elem_classes=["stats-out"],
            )

        # ── RIGHT: Chat Panel ─────────────────────────────────────────────────
        with gr.Column(scale=8, min_width=500):
            gr.HTML("<div class='panel-title'>💬 AI se Poochho</div>")

            chatbot = gr.Chatbot(
                height=480,
                show_label=False,
                placeholder="<div style='text-align:center;color:#1e2d4a;padding:80px 20px'><div style='font-size:3rem'>🔍</div><div style='font-size:1rem;margin-top:10px;color:#334155'>Pehle document upload karo → phir sawaal poochho</div></div>",
                type="messages",
            )

            with gr.Row():
                q_input = gr.Textbox(
                    placeholder="💭 Apna sawaal yahan likho...",
                    label="",
                    scale=6,
                    lines=1,
                )
                ask_btn = gr.Button("Ask 🚀", variant="primary", scale=1, min_width=90)
                clear_btn = gr.Button("🗑️", variant="secondary", scale=0, min_width=48)

            gr.HTML("""
            <div style='margin-top:8px;padding:10px 14px;background:#0d1424;border:1px solid #1e2d4a;border-radius:8px;'>
                <span style='color:#334155;font-size:0.78rem'>
                💡 <strong style='color:#6366f1'>Tips:</strong>
                Specific sawaal poochho &nbsp;·&nbsp;
                Hindi ya English dono chalega &nbsp;·&nbsp;
                Enter dabao ya button click karo
                </span>
            </div>
            """)

    # ── Events ────────────────────────────────────────────────────────────────
    upload_btn.click(
        upload_and_ingest,
        inputs=file_input,
        outputs=[upload_status, stats_out],
    )
    ask_btn.click(
        answer_question,
        inputs=[q_input, chatbot],
        outputs=[chatbot, q_input],
    )
    q_input.submit(
        answer_question,
        inputs=[q_input, chatbot],
        outputs=[chatbot, q_input],
    )
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, q_input])

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        css=custom_css,
    )
