# PolicyPilot

A RAG-based assistant that answers questions about policy documents — grounded in the actual source text, cited by page number, and built to refuse rather than guess when an answer isn't in the document.

**Live demo:** https://policy-pilot.streamlit.app/
**Backend API docs:** https://policypilot-api-c3yw.onrender.com/docs

> Note: the backend runs on a free-tier instance that sleeps after inactivity. The first question after a period of inactivity may take up to ~60 seconds while it wakes up; subsequent questions respond in 1-2 seconds.

---

## The problem

Policy documents — insurance terms, HR handbooks, compliance manuals — are long, dense, and rarely read in full. Employees and customers end up asking someone else the same handful of questions repeatedly, or worse, guessing. A wrong answer about a legal/financial policy isn't just inconvenient — it's a real liability.

PolicyPilot lets someone ask a plain-English question and get an answer that is:
- **Grounded** — generated only from the actual document text, never the model's general knowledge
- **Cited** — every claim points to a real source file and page number
- **Honest about its limits** — explicitly refuses to answer when the document doesn't contain the information, rather than guessing

## Architecture


**Two deployed services, fully decoupled:**
- **Backend** (FastAPI, on Render) — owns ingestion, retrieval, and the RAG chain. Exposes `/health`, `/upload`, `/ask`.
- **Frontend** (Streamlit, on Streamlit Community Cloud) — a chat UI that calls the backend over HTTP, exactly as an independent client would.

## Key design decisions

**Multi-tenant document isolation.** Beyond the default pre-loaded policy corpus, users can upload their own PDF. Each upload gets a unique `session_id` tagged onto every chunk's metadata; retrieval for that session is filtered (via a Qdrant payload index + filter) to only search chunks matching that ID. The default corpus is queried with the inverse filter (chunks with *no* `session_id`), so uploaded documents can never leak into the shared knowledge base, and vice versa.

**Embeddings via API, not a local model.** Rather than loading `sentence-transformers`/PyTorch locally (a multi-GB dependency), embeddings are generated through HuggingFace's hosted Inference API. This kept the project runnable on a disk-constrained machine and mirrors how many production systems treat embedding as a network call, not an in-process model.

**Grounding + refusal as a first-class prompt requirement**, not an afterthought — the system prompt explicitly instructs the model to answer only from retrieved context and to return an exact, checkable refusal string when the context doesn't contain the answer. This was tested directly (see "Known limitations & what I'd improve" below).

**Separate, minimal dependency files per deployable service** (`requirements-api.txt` for the backend, `frontend/requirements.txt` for the frontend) rather than one shared `pip freeze` dump — the backend doesn't need Streamlit's dependency tree, and mixing them caused real deployment failures during development (see below).

## Tech stack

| Layer | Choice |
|---|---|
| LLM | Groq (`openai/gpt-oss-120b`) |
| Embeddings | HuggingFace Inference API (`sentence-transformers/all-MiniLM-L6-v2`) |
| Vector store | Qdrant Cloud |
| Orchestration | LangChain (LCEL) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Observability | LangSmith |
| Backend hosting | Render |
| Frontend hosting | Streamlit Community Cloud |

## Real bugs found and fixed during development

These were genuine issues discovered through testing, not hypothetical "future work" — I think they're more informative than a list of features that just worked on the first try:

- **Retrieval ambiguity presented as a complete answer.** A question like *"what happens if the policyholder dies during the loan tenure"* initially returned only one of several distinct, differently-worded death-benefit clauses in the source document, and the model answered confidently as if it were the whole picture. Fixed by increasing `top_k` and explicitly instructing the prompt to surface multiple distinct provisions separately rather than silently picking one.
- **Multi-tenancy isolation bug.** The default (non-uploaded) corpus wasn't actually filtered to exclude session-tagged chunks, so repeated test uploads of the same document silently polluted the shared knowledge base with duplicates, crowding out genuinely different chunks from retrieval. Fixed with an explicit "is-empty" filter on `metadata.session_id` for the default-corpus path.
- **Free-tier infrastructure isn't permanent.** An inactive Qdrant Cloud cluster was automatically deleted between sessions, taking its data with it — a good reminder that "the cloud" still has real operational limits, not just an abstraction to assume is always there.
- **Model/library naming drift.** Hit three separate instances of documentation/memory going stale within one project: a Groq model tier change, a LangChain env var rename (`LANGCHAIN_*` → `LANGSMITH_*`), and Render using `PYTHON_VERSION` rather than the Heroku-style `runtime.txt` convention. The fix each time was the same: check current official docs rather than trust memorized specifics.
- **A near-miss with committed secrets.** `.env` was committed in an early commit before `.gitignore` existed. GitHub's push protection caught and blocked the push before it became public. Resolved by resetting local git history and re-committing cleanly, plus rotating the affected keys as a precaution.

## What I'd improve next

- **Automated evaluation (RAGAS)** — replace manual spot-checking with faithfulness/relevancy scores across a real test set
- **Re-ranking / deduplication** — retrieval occasionally surfaces near-duplicate chunks; a cross-encoder rerank step would make better use of the same top-k budget
- **Cleaner citations** — currently shows the full (sometimes messy, temp-path) source filename; would strip this down to a clean display name
- **Session cleanup** — uploaded documents currently persist indefinitely in Qdrant; a real product would need a TTL/cleanup job
- **Tests + CI** — no automated test suite yet

## Running locally

```bash
git clone https://github.com/Lohith0409/Policypilot.git
cd Policypilot
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-api.txt

# Create a .env file with:
# GROQ_API_KEY=...
# QDRANT_URL=...
# QDRANT_API_KEY=...
# HUGGINGFACEHUB_API_TOKEN=...

python scripts\ingest.py          # one-time: ingest the default document set
python -m uvicorn src.api.main:app --reload   # backend, in one terminal

pip install -r frontend/requirements.txt
python -m streamlit run frontend/app.py       # frontend, in a second terminal
```