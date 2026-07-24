"""
UOS Hostel Assistant — Streamlit app (single file)
====================================================
This is a Streamlit port of the original Flask + HTML/CSS/JS app. It keeps
the same visual theme ("Registry Desk": ink navy / brick maroon / brass /
warm parchment), the same sidebar (seal, quick-lookup chips, reset button),
the same chat bubble layout with collapsible source passages, and the same
Parent-Document RAG pipeline (Chroma child-chunk index + filesystem-backed
parent docstore, Mistral embeddings + Mistral chat model).

Run:
    1. Unzip your storage.zip into this folder so you have:
         storage/chroma_db/...
         storage/parent_docs/...
    2. Copy .env.example -> .env and fill in MISTRAL_API_KEY
    3. pip install streamlit python-dotenv langchain-chroma langchain-classic \
         langchain-core langchain-mistralai langchain-text-splitters
    4. streamlit run app.py
"""

import html as html_lib
import os
import pickle
import re
import time
from typing import Iterator, List, Optional, Sequence, Tuple

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CHAT_MODEL = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")
EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "split_parents")

PERSIST_DIR = os.getenv("PERSIST_DIR", "./storage")
CHROMA_DIR = os.path.join(PERSIST_DIR, "chroma_db")
DOCSTORE_DIR = os.path.join(PERSIST_DIR, "parent_docs")

TOP_K = int(os.getenv("TOP_K", "4"))                # parent docs retrieved per turn
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))  # remembered turns per session

st.set_page_config(
    page_title="UOS Hostel Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Same custom classes used when the index was built (must match exactly so
# pickled parent Documents load back correctly).
# ---------------------------------------------------------------------------
class FileSystemDocStore(BaseStore[str, Document]):
    def __init__(self, root_path: str):
        self.root_path = root_path
        os.makedirs(root_path, exist_ok=True)

    def _path(self, key: str) -> str:
        safe_key = key.replace("/", "_").replace("\\", "_")
        return os.path.join(self.root_path, f"{safe_key}.pkl")

    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        results = []
        for key in keys:
            path = self._path(key)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    results.append(pickle.load(f))
            else:
                results.append(None)
        return results

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        for key, value in key_value_pairs:
            with open(self._path(key), "wb") as f:
                pickle.dump(value, f)

    def mdelete(self, keys: Sequence[str]) -> None:
        for key in keys:
            path = self._path(key)
            if os.path.exists(path):
                os.remove(path)

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        for filename in os.listdir(self.root_path):
            if filename.endswith(".pkl"):
                key = filename[:-4]
                if prefix is None or key.startswith(prefix):
                    yield key


class SafeMistralEmbeddings(MistralAIEmbeddings):
    """Batches embedding calls so we never exceed Mistral's request limits."""

    def embed_documents(self, texts, max_batch_size: int = 16):
        all_embeddings = []
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i: i + max_batch_size]
            all_embeddings.extend(super().embed_documents(batch))
        return all_embeddings


# ---------------------------------------------------------------------------
# Build the retriever + LLM once, cached across reruns/sessions
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading hostel handbook knowledge base…")
def load_pipeline():
    if not MISTRAL_API_KEY:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    if not os.path.isdir(CHROMA_DIR) or not os.path.isdir(DOCSTORE_DIR):
        raise RuntimeError(
            f"Could not find '{CHROMA_DIR}' and '{DOCSTORE_DIR}'.\n"
            f"Unzip your storage.zip into '{PERSIST_DIR}' before starting the app."
        )

    embeddings = SafeMistralEmbeddings(model=EMBED_MODEL, api_key=MISTRAL_API_KEY)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    docstore = FileSystemDocStore(root_path=DOCSTORE_DIR)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""], length_function=len,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""], length_function=len,
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.search_kwargs = {"k": TOP_K}

    doc_count = vectorstore._collection.count()
    llm = ChatMistralAI(model=CHAT_MODEL, temperature=0.1, api_key=MISTRAL_API_KEY)
    return retriever, llm, doc_count


try:
    retriever, llm, doc_count = load_pipeline()
except Exception as exc:
    st.error(str(exc))
    st.stop()

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the UOS Hostel Assistant, a helpful information desk assistant \
for the University of Sargodha's hostel system.

Answer the student's question using ONLY the information in the "Retrieved context" \
section below. This context comes from the official UOS Hostel Handbook.

Rules you must follow:
1. If the answer is fully or partially contained in the context, answer clearly and \
   directly. Use short paragraphs or bullet points when it improves readability.
2. If the context does not contain the answer, say plainly that the handbook does not \
   cover that, and suggest the student contact the hostel superintendent's office or \
   student affairs for confirmation. Never invent facts, names, numbers, or fees.
3. For anything involving money (fees, fines, deposits) or official contacts (names, \
   phone numbers, room counts), only state figures that literally appear in the context. \
   Add a brief note to double check with the hostel office before acting on it.
4. Keep a warm, respectful, student-friendly tone, like a knowledgeable senior helping a \
   junior — but stay concise. Avoid filler like "As an AI" or repeating the question back.
5. If the student greets you or makes small talk, respond briefly and naturally without \
   forcing in hostel information.
6. Use the conversation history to resolve follow-up questions (e.g. "what about Iqbal \
   hall?" after discussing fees), but always ground factual claims in the retrieved context.

Retrieved context:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

output_parser = StrOutputParser()


def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(no matching passages were found in the handbook)"
    blocks = []
    for i, d in enumerate(docs, start=1):
        page = d.metadata.get("page")
        page_str = f" (page {page + 1})" if isinstance(page, int) else ""
        blocks.append(f"[Passage {i}{page_str}]\n{d.page_content.strip()}")
    return "\n\n".join(blocks)


def answer_question(question: str, chat_history: List) -> Tuple[str, List[Document]]:
    docs = retriever.invoke(question)
    context = format_docs(docs)
    chain = prompt | llm | output_parser
    answer = chain.invoke(
        {
            "question": question,
            "context": context,
            "chat_history": chat_history,
        }
    )
    return answer, docs


# ---------------------------------------------------------------------------
# Session state (per browser session, mirrors Flask's SESSION_HISTORY)
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # langchain BaseMessage list, fed to the chain

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Assalam-o-Alaikum! I'm the hostel assistant — ask me anything about "
                "hostel fees, facilities, rules, or the application process, and I'll "
                "answer from the official handbook."
            ),
            "sources": None,
            "error": False,
        }
    ]

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def trim_history() -> None:
    max_messages = MAX_HISTORY_TURNS * 2
    if len(st.session_state.chat_history) > max_messages:
        del st.session_state.chat_history[: len(st.session_state.chat_history) - max_messages]


def handle_query(text: str) -> None:
    text = (text or "").strip()
    if not text:
        return
    if len(text) > 2000:
        st.session_state.messages.append(
            {"role": "user", "content": text[:2000], "sources": None, "error": False}
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Message is too long (max 2000 characters).",
                "sources": None,
                "error": True,
            }
        )
        return

    st.session_state.messages.append({"role": "user", "content": text, "sources": None, "error": False})

    try:
        answer, docs = answer_question(text, st.session_state.chat_history)
        sources = []
        for d in docs:
            sources.append(
                {
                    "excerpt": d.page_content.strip()[:400],
                    "page": (d.metadata.get("page") + 1) if isinstance(d.metadata.get("page"), int) else None,
                    "source": os.path.basename(d.metadata.get("source", "") or ""),
                }
            )
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources, "error": False}
        )
        st.session_state.chat_history.append(HumanMessage(content=text))
        st.session_state.chat_history.append(AIMessage(content=answer))
        trim_history()
    except Exception as exc:  # surfaces API errors, rate limits, etc. to the UI
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"The assistant hit an error: {exc}",
                "sources": None,
                "error": True,
            }
        )


def format_answer_html(text: str) -> str:
    """Mirrors the original script.js formatAnswer(): **bold**, "- " bullet lists."""
    safe = html_lib.escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)

    lines = safe.split("\n")
    out = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^[-•]\s+", stripped):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{re.sub(r'^[-•]\\s+', '', stripped)}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            if stripped:
                out.append(f"<p>{stripped}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out) if out else f"<p>{safe}</p>"


# ---------------------------------------------------------------------------
# Theme CSS (adapted from style.css — "Registry Desk" theme)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --ink:        #17213A;
  --ink-2:      #202D4C;
  --maroon:     #7A2436;
  --maroon-2:   #932C41;
  --brass:      #B7924B;
  --brass-soft: #D8C08C;
  --paper:      #FAF8F2;
  --paper-2:    #F1ECDF;
  --line:       #E3DCC9;
  --text:       #232323;
  --text-soft:  #5B5747;
  --green:      #4C7A5B;
  --radius:     10px;
  --font-display: "Source Serif 4", Georgia, serif;
  --font-body: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: "IBM Plex Mono", "SFMono-Regular", monospace;
}

html, body, .stApp { background: var(--paper) !important; color: var(--text); font-family: var(--font-body); }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; height: 3rem; }
.block-container { padding-top: 4.5rem; max-width: 900px; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
  background: linear-gradient(175deg, var(--ink) 0%, var(--ink-2) 100%) !important;
}
section[data-testid="stSidebar"] * { color: #EDE9DC; }

.seal { display: flex; justify-content: center; margin-bottom: 14px; }
.seal-ring {
  position: relative; width: 88px; height: 88px; border-radius: 50%;
  border: 1px solid rgba(216, 192, 140, 0.55);
  display: flex; align-items: center; justify-content: center;
}
.seal-ring::before {
  content: ""; position: absolute; inset: 6px; border-radius: 50%;
  border: 1px dashed rgba(216, 192, 140, 0.35);
}
.seal-svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.seal-svg text { fill: var(--brass-soft); font-family: var(--font-mono); letter-spacing: 0.5px; }
.seal-center { font-family: var(--font-display); font-weight: 700; font-size: 21px; color: var(--brass-soft); letter-spacing: 1px; }

.brand-title {
  font-family: var(--font-display); font-weight: 600; font-size: 27px; line-height: 1.1;
  text-align: center; margin: 4px 0 8px; color: #FBF9F2;
}
.brand-sub { font-size: 13px; line-height: 1.55; color: #B9C0D4; text-align: center; margin: 0 4px 16px; }

.status-row {
  display: flex; align-items: center; gap: 8px; font-family: var(--font-mono); font-size: 11.5px;
  color: #A9C7B4; background: rgba(76, 122, 91, 0.12); border: 1px solid rgba(76, 122, 91, 0.35);
  border-radius: 999px; padding: 7px 12px; justify-content: center; margin-bottom: 14px;
}
.status-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--green);
  box-shadow: 0 0 0 3px rgba(76, 122, 91, 0.25); display: inline-block;
}

.divider { height: 1px; background: rgba(216, 192, 140, 0.2); margin: 6px 0 14px; }
.section-label {
  font-family: var(--font-mono); text-transform: uppercase; letter-spacing: 1.2px;
  font-size: 10.5px; color: var(--brass-soft); margin: 0 0 8px;
}

section[data-testid="stSidebar"] .stButton button {
  width: 100%; text-align: left; background: rgba(255, 255, 255, 0.04) !important;
  border: 1px solid rgba(216, 192, 140, 0.22) !important; color: #E8E4D6 !important;
  border-radius: 8px !important; padding: 8px 12px !important; font-size: 13px !important;
  font-family: var(--font-body) !important; box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
  background: rgba(216, 192, 140, 0.12) !important; border-color: rgba(216, 192, 140, 0.5) !important;
}
section[data-testid="stSidebar"] .stButton button p { color: #E8E4D6 !important; }

.model-tag { font-family: var(--font-mono); font-size: 10.5px; color: #7D869E; text-align: center; margin: 12px 0 0; }
.credit-line {
  font-family: var(--font-mono); font-size: 10.5px; letter-spacing: 0.3px; color: #8891A6;
  text-align: center; margin: 6px 0 0; padding-top: 10px; border-top: 1px solid rgba(216, 192, 140, 0.15);
}
.credit-line span { color: var(--brass-soft); font-weight: 500; }

/* ---- Main chat header ---- */
.chat-header { padding: 0 0 14px; border-bottom: 1px solid var(--line); margin-bottom: 18px; }
.eyebrow {
  font-family: var(--font-mono); text-transform: uppercase; letter-spacing: 1.4px;
  font-size: 10.5px; color: var(--maroon-2); margin: 0 0 6px;
}
.chat-header h2 { font-family: var(--font-display); font-weight: 600; font-size: 24px; margin: 0; color: var(--ink); }

/* ---- Messages ---- */
.msg { display: flex; gap: 12px; max-width: 760px; margin-bottom: 16px; }
.msg.user { margin-left: auto; flex-direction: row-reverse; }
.avatar {
  flex-shrink: 0; width: 34px; height: 34px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--font-mono); font-size: 10px; font-weight: 600; letter-spacing: 0.5px;
  background: var(--ink); color: var(--brass-soft); border: 1px solid var(--brass);
}
.msg.user .avatar { background: var(--maroon); color: #F3E3D8; border-color: var(--maroon-2); }
.bubble {
  background: #fff; border: 1px solid var(--line); border-radius: var(--radius);
  padding: 13px 16px; font-size: 14.5px; line-height: 1.6; color: var(--text);
  box-shadow: 0 1px 2px rgba(23, 33, 58, 0.04);
}
.msg.user .bubble { background: var(--ink); border-color: var(--ink-2); color: #F1EEE4; }
.bubble p { margin: 0 0 8px; }
.bubble p:last-child { margin-bottom: 0; }
.bubble ul { margin: 6px 0; padding-left: 20px; }
.bubble strong { color: var(--maroon-2); }
.msg.user .bubble strong { color: var(--brass-soft); }
.error-bubble { background: #FBEDEA !important; border-color: #E3B4AC !important; color: #7A2A1D !important; }

.source-card {
  background: var(--paper-2); border: 1px solid var(--line); border-left: 3px solid var(--brass);
  border-radius: 6px; padding: 9px 11px; font-size: 12.5px; color: var(--text-soft);
  line-height: 1.5; margin-bottom: 6px;
}
.source-card .tag {
  font-family: var(--font-mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: 0.5px; color: var(--maroon-2); display: block; margin-bottom: 4px;
}

.disclaimer { margin: 16px 0 0; font-size: 11px; color: var(--text-soft); text-align: center; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
SEAL_SVG = """
<div class="seal">
  <div class="seal-ring">
    <svg viewBox="0 0 100 100" class="seal-svg">
      <path id="seal-arc-top" d="M 12,50 A 38,38 0 1 1 88,50" fill="none"/>
      <path id="seal-arc-bottom" d="M 88,50 A 38,38 0 1 1 12,50" fill="none"/>
      <text font-size="6.3"><textPath href="#seal-arc-top" startOffset="50%" text-anchor="middle">UNIVERSITY&#8202;OF&#8202;SARGODHA</textPath></text>
      <text font-size="6.3"><textPath href="#seal-arc-bottom" startOffset="50%" text-anchor="middle">HOSTEL&#8202;DESK&#8202;·&#8202;EST.&#8202;RECORDS</textPath></text>
    </svg>
    <span class="seal-center">UOS</span>
  </div>
</div>
"""

QUICK_LOOKUPS = [
    ("Application documents", "What documents are required to apply for a hostel seat?"),
    ("Fee & mess charges", "What is the hostel fee and mess charges?"),
    ("Facilities", "What facilities are available in the hostels?"),
    ("Warm water in winter", "Is warm water available in hostels during winter?"),
    ("Attendance & timing", "What is the attendance and curfew time?"),
    ("Bringing appliances", "Can I bring my own air cooler or other appliances?"),
]

with st.sidebar:
    st.markdown(SEAL_SVG, unsafe_allow_html=True)
    st.markdown('<h1 class="brand-title">Hostel<br>Assistant</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="brand-sub">Answers drawn from the official UOS Hostel Handbook — '
        "fees, facilities, rules &amp; applications.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="status-row"><span class="status-dot"></span>'
        f"<span>Knowledge base loaded · {doc_count} indexed passages</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Quick lookups</p>', unsafe_allow_html=True)

    for label, question in QUICK_LOOKUPS:
        if st.button(label, key=f"chip_{label}", use_container_width=True):
            st.session_state.pending_query = question

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if st.button("↺  New conversation", key="reset_btn", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Conversation cleared. Ask me anything about hostel fees, facilities, rules, or applications.",
                "sources": None,
                "error": False,
            }
        ]
        st.session_state.pending_query = None
        st.rerun()

    st.markdown(f'<p class="model-tag">Model: {CHAT_MODEL}</p>', unsafe_allow_html=True)
    st.markdown('<p class="credit-line">Made by <span>Munib Ahmad</span></p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main chat header
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="chat-header">
  <p class="eyebrow">Student Information Desk</p>
  <h2>Ask about hostel life at UOS</h2>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Handle a pending chip click before rendering, so it shows up immediately
# ---------------------------------------------------------------------------
if st.session_state.pending_query:
    q = st.session_state.pending_query
    st.session_state.pending_query = None
    with st.spinner("Looking through the handbook…"):
        handle_query(q)

# ---------------------------------------------------------------------------
# Render chat messages
# ---------------------------------------------------------------------------
for m in st.session_state.messages:
    role = m["role"]
    avatar_label = "YOU" if role == "user" else "UOS"
    row_class = "msg user" if role == "user" else "msg assistant"
    bubble_class = "bubble error-bubble" if m.get("error") else "bubble"
    body_html = format_answer_html(m["content"]) if role == "assistant" else f"<p>{html_lib.escape(m['content'])}</p>"

    st.markdown(
        f"""
        <div class="{row_class}">
          <div class="avatar">{avatar_label}</div>
          <div class="{bubble_class}">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if role == "assistant" and m.get("sources"):
        sources = m["sources"]
        with st.expander(f"View {len(sources)} source passage{'s' if len(sources) > 1 else ''}"):
            for s in sources:
                tag_parts = [p for p in [s.get("source"), f"p.{s['page']}" if s.get("page") else None] if p]
                tag = " · ".join(tag_parts) or "Handbook"
                excerpt = html_lib.escape(s["excerpt"]) + ("…" if len(s["excerpt"]) >= 400 else "")
                st.markdown(
                    f'<div class="source-card"><span class="tag">{html_lib.escape(tag)}</span>{excerpt}</div>',
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# Composer (Enter to send, like the original textarea)
# ---------------------------------------------------------------------------
user_input = st.chat_input("Ask about hostel fees, rooms, mess, rules…", max_chars=2000)

if user_input:
    with st.spinner("Looking through the handbook…"):
        handle_query(user_input)
    st.rerun()

st.markdown(
    '<p class="disclaimer">Answers are generated from the hostel handbook and may be '
    "incomplete — confirm fees and official details with the hostel office.</p>",
    unsafe_allow_html=True,
)
