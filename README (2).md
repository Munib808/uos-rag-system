# UOS Hostel Assistant

A RAG-powered chat assistant that answers student questions about University of Sargodha
hostel life — fees, facilities, rules, and applications — grounded entirely in the official
UOS Hostel Handbook. Built with a Parent-Document Retriever over Chroma, Mistral embeddings
and chat models, and shipped as **two interchangeable front-ends**: a Flask web app and a
single-file Streamlit app.

Both apps share the exact same retrieval pipeline, prompt template, and chat behavior — pick
whichever is easier for you to deploy.

## Highlights

- **Grounded answers only** — the assistant answers strictly from retrieved handbook
  passages, and plainly says when something isn't covered instead of guessing.
- **Source transparency** — every answer includes a "View sources" toggle showing the exact
  passages it used.
- **Multi-turn memory** — follow-up questions like "what about Iqbal Hall?" are resolved
  using recent chat history.
- **Pre-built embeddings** — the handbook was chunked and embedded once and the resulting
  Chroma index + parent docstore are shipped in `storage/`, so the app loads instantly and
  never re-embeds the handbook on startup or per request.
- **Data curation** — the handbook content that powers this assistant was sourced and
  prepared by **Munib Ahmad**.

## Project layout

```
uos_rag_app/
├── flaskapp.py             # Flask backend + RAG chain + prompt template
├── app.py                  # Streamlit app (single file, same pipeline & theme)
├── requirements.txt
├── .env.example            # copy to .env and fill in your key
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/script.js
└── storage/                 # <- put your unzipped storage.zip contents here
    ├── chroma_db/
    └── parent_docs/
```

## 1. Set up your index

Unzip the pre-built `storage.zip` (the embeddings, already generated — you do not need
to re-run the embedding step) so it sits at `uos_rag_app/storage/`:

```bash
cd uos_rag_app
unzip /path/to/storage.zip -d storage
```

You should end up with `storage/chroma_db/` and `storage/parent_docs/` directly inside the
app folder (not nested one level deeper — if `unzip` creates `storage/storage/...`, move the
contents up one level).

## 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Configure your API key

```bash
cp .env.example .env
```

Edit `.env` and set `MISTRAL_API_KEY`. **Rotate any key that was previously hardcoded** in
a notebook or shared file — treat it as compromised.

## 4. Run it

**Flask version:**
```bash
python flaskapp.py
```
Open **http://localhost:5000**.

**Streamlit version:**
```bash
streamlit run app.py
```
Opens automatically in your browser (default **http://localhost:8501**).

## What changed vs. the original notebook

- **Prompt template added** (`SYSTEM_PROMPT`): the assistant now answers strictly from
  retrieved context, admits when the handbook doesn't cover something, is careful about
  fees/names/numbers, and keeps a friendly, concise tone — instead of relying on
  `RetrievalQA`'s generic default prompt.
- **Multi-turn memory**: each session keeps its own short chat history (last 5 exchanges by
  default) so follow-up questions work naturally.
- **Source transparency**: every answer includes a "View sources" toggle showing the exact
  handbook passages used.
- **No API key in code**: reads `MISTRAL_API_KEY` from environment/`.env`.
- **Embeddings computed once, reused forever**: the Chroma index and parent docstore are
  built ahead of time and persisted to `storage/` — both apps just load them at startup
  instead of embedding the handbook on every run.

## Customizing

- Change retrieved passage count: `TOP_K` in `.env`.
- Change how much history is remembered: `MAX_HISTORY_TURNS` in `.env`.
- Edit the assistant's tone/rules: `SYSTEM_PROMPT` (in `flaskapp.py` or `app.py`).
- Edit quick-lookup buttons: the `.chip` buttons in `templates/index.html` (Flask) or the
  `QUICK_LOOKUPS` list in `app.py` (Streamlit).

## Production notes

- Chat history is currently stored **in memory** — it resets if the server restarts, and
  won't work correctly across multiple worker processes. For production, swap the in-memory
  store for Redis or a database.
- Run the Flask app with a real WSGI server (e.g. `gunicorn flaskapp:app`) instead of the
  dev server, and put it behind HTTPS.

## Credits

- Handbook data collection & preparation: **Munib Ahmad**
- Retrieval pipeline: Parent-Document Retriever (LangChain) over Chroma
- Embeddings & chat model: Mistral AI
