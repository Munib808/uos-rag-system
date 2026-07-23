"""
UOS Hostel Assistant — Flask backend
=====================================
Serves a chat UI in front of the Parent-Document RAG pipeline you built in
the notebook (Chroma child-chunk index + filesystem-backed parent docstore,
Mistral embeddings + Mistral chat model).

Run:
    1. Unzip your storage.zip into this folder so you have:
         storage/chroma_db/...
         storage/parent_docs/...
    2. Copy .env.example -> .env and fill in MISTRAL_API_KEY
    3. pip install -r requirements.txt
    4. python app.py
    5. Open http://localhost:5000
"""

import os
import pickle
import time
import uuid
from typing import Iterator, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session

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
if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY is not set. Copy .env.example to .env and add your key."
    )

CHAT_MODEL = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")
EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "split_parents")

PERSIST_DIR = os.getenv("PERSIST_DIR", "./storage")
CHROMA_DIR = os.path.join(PERSIST_DIR, "chroma_db")
DOCSTORE_DIR = os.path.join(PERSIST_DIR, "parent_docs")

TOP_K = int(os.getenv("TOP_K", "4"))                # parent docs to retrieve per turn
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))  # remembered turns per session

if not os.path.isdir(CHROMA_DIR) or not os.path.isdir(DOCSTORE_DIR):
    raise RuntimeError(
        f"Could not find '{CHROMA_DIR}' and '{DOCSTORE_DIR}'.\n"
        f"Unzip your storage.zip into '{PERSIST_DIR}' before starting the app."
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
            batch = texts[i : i + max_batch_size]
            all_embeddings.extend(super().embed_documents(batch))
        return all_embeddings


# ---------------------------------------------------------------------------
# Build the retriever + LLM once, at startup
# ---------------------------------------------------------------------------
print("Loading embeddings + vector store ...")
embeddings = SafeMistralEmbeddings(model=EMBED_MODEL, api_key=MISTRAL_API_KEY)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)
docstore = FileSystemDocStore(root_path=DOCSTORE_DIR)

# These splitters are only used if new documents are ever added again; for
# pure retrieval against an already-built index they're inert but required
# by the ParentDocumentRetriever constructor.
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
print(f"Vector store ready — {doc_count} child chunks indexed.")

llm = ChatMistralAI(model=CHAT_MODEL, temperature=0.1, api_key=MISTRAL_API_KEY)

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
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", str(uuid.uuid4()))

# In-memory per-session chat history: {session_id: [BaseMessage, ...]}
SESSION_HISTORY = {}


def get_history(session_id: str) -> List:
    return SESSION_HISTORY.setdefault(session_id, [])


def trim_history(history: List) -> None:
    max_messages = MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        del history[: len(history) - max_messages]


@app.route("/")
def index():
    return render_template("index.html", doc_count=doc_count, chat_model=CHAT_MODEL)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400
    if len(message) > 2000:
        return jsonify({"error": "Message is too long (max 2000 characters)."}), 400

    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    history = get_history(session["sid"])

    start = time.time()
    try:
        answer, docs = answer_question(message, history)
    except Exception as exc:  # surfaces API errors, rate limits, etc. to the UI
        return jsonify({"error": f"The assistant hit an error: {exc}"}), 502
    elapsed = round(time.time() - start, 2)

    history.append(HumanMessage(content=message))
    history.append(AIMessage(content=answer))
    trim_history(history)

    sources = []
    for d in docs:
        sources.append(
            {
                "excerpt": d.page_content.strip()[:400],
                "page": (d.metadata.get("page") + 1) if isinstance(d.metadata.get("page"), int) else None,
                "source": os.path.basename(d.metadata.get("source", "") or ""),
            }
        )

    return jsonify(
        {
            "answer": answer,
            "sources": sources,
            "elapsed": elapsed,
        }
    )


@app.route("/api/reset", methods=["POST"])
def reset():
    session_id = session.get("sid")
    if session_id and session_id in SESSION_HISTORY:
        SESSION_HISTORY[session_id] = []
    return jsonify({"status": "ok"})


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "chat_model": CHAT_MODEL,
            "embed_model": EMBED_MODEL,
            "indexed_chunks": doc_count,
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
