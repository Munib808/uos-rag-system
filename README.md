# UOS Hostel Assistant — Flask App

A Flask front-end for your Parent-Document RAG pipeline (Chroma + Mistral),
with a proper prompt template, multi-turn memory, and a chat UI that shows
the retrieved source passages.

## 1. Project layout

```
uos_rag_app/
├── app.py                 # Flask backend + RAG chain + prompt template
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

## 2. Set up your index

Unzip **your** `storage.zip` (the one with the embeddings you already built)
so it sits at `uos_rag_app/storage/`:

```bash
cd uos_rag_app
unzip /path/to/storage.zip -d storage
```

You should end up with `storage/chroma_db/` and `storage/parent_docs/`
directly inside the app folder (not nested one level deeper — if `unzip`
creates `storage/storage/...`, move the contents up one level).

## 3. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Configure your API key

```bash
cp .env.example .env
```

Edit `.env` and set `MISTRAL_API_KEY`. **Rotate the key that was hardcoded
in your notebook** — treat it as compromised since it's now been shared in
a shared file.

## 5. Run it

```bash
python app.py
```

Open **http://localhost:5000**.

## What changed vs. the notebook

- **Prompt template added** (`SYSTEM_PROMPT` in `app.py`): the assistant now
  answers strictly from retrieved context, admits when the handbook doesn't
  cover something, is careful about fees/names/numbers, and keeps a
  friendly, concise tone — instead of relying on `RetrievalQA`'s generic
  default prompt.
- **Multi-turn memory**: each browser session keeps its own short chat
  history (last 5 exchanges) so follow-up questions like "what about Iqbal
  Hall?" work.
- **Source transparency**: every answer includes a "View sources" toggle
  showing the exact handbook passages used.
- **No API key in code**: reads `MISTRAL_API_KEY` from environment/`.env`.

## Customizing

- Change retrieved passage count: `TOP_K` in `.env`.
- Change how much history is remembered: `MAX_HISTORY_TURNS` in `.env`.
- Edit the assistant's tone/rules: `SYSTEM_PROMPT` in `app.py`.
- Edit quick-lookup buttons: the `.chip` buttons in `templates/index.html`.

## Production notes

- Chat history is currently stored **in memory** (a Python dict) — it resets
  if the server restarts, and won't work correctly if you run multiple
  worker processes. For production, swap `SESSION_HISTORY` for Redis or a
  database.
- Run with a real WSGI server (e.g. `gunicorn app:app`) instead of the Flask
  dev server, and put it behind HTTPS.
