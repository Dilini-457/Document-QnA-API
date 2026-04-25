# Document Q&A API

A RAG-based (Retrieval-Augmented Generation) API that answers questions strictly based on uploaded document context.

## How it works

1. Upload a document (text, .txt file, or .pdf file) via `/ingest`
2. Ask questions about it via `/ask`
3. The app finds relevant parts of the document and uses Groq LLM to answer
4. The LLM only answers based on the document — not from its own knowledge

## Tech Stack

- **FastAPI** — API framework
- **Groq LLM** (llama-3.1-8b-instant) — AI brain
- **TF-IDF + Cosine Similarity** — document search
- **Docker** — containerization

## Run locally with Docker

1. Clone this repo:
```bash
git clone https://github.com/Dilini-457/Document-QnA-API.git
cd Document-QnA-API
```

2. Create a `.env` file with your Groq API key:
GROQ_API_KEY=your_groq_api_key_here

3. Build and run with Docker:
```bash
docker build -t rag-app .
docker run -p 8000:8000 --env-file .env rag-app
```

4. Visit http://localhost:8000/docs to test the API

## Run locally without Docker

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with your Groq API key

3. Run the app:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### POST /ingest
Upload a document. Accepts either:
- A `.txt` or `.pdf` file
- Raw text typed directly

File example:
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.pdf"
```

Text example:
```bash
curl -X POST http://localhost:8000/ingest \
  -F "text=Your paragraph text here"
```

### POST /ask
Ask a question about the uploaded document:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'
```

### GET /
Health check — confirms the app is running.

## Live Demo
Coming soon — will be updated after deployment.