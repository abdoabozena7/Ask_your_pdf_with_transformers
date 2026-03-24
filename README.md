# Ask Your PDF with Transformers

A document question-answering app built with `FastAPI`, `Transformers`, and a custom archival-style HTML interface. Upload a PDF or TXT file, browse simple chapter-like sections from the left rail, ask questions against the whole document or the selected section, and inspect notes from the right annotations panel.

## Current UI

The app now uses a three-column workspace instead of Streamlit:

- Left rail: `Chapters`, upload button, and quick navigation for `Introduction`, `Methodology`, `Key Findings`, and `Conclusion`
- Center workspace: document card, question composer, AI answer panel, and related context blocks
- Right rail: annotations cards and supporting-file drop area
- Top bar: workspace tabs, document/status actions, and profile avatar

The current visual direction matches the new `Lexicon Archival` layout shown in the latest UI update.

## Features

- Upload PDF and TXT files from the browser
- Extract text from PDFs with `PyPDF2`
- Ask questions with `distilbert-base-cased-distilled-squad`
- Restrict QA to the selected section from the left sidebar
- Detect simple section headings when present, with fallback section splitting when headings are missing
- Preview the uploaded source file directly from the interface
- Use a static HTML + Tailwind frontend instead of Streamlit

## How It Works

1. Upload a document from the `Upload New` button.
2. The backend reads and normalizes the text.
3. The app builds lightweight sections such as `Introduction`, `Methodology`, `Key Findings`, and `Conclusion`.
4. When you ask a question, the transformer searches the selected section first.
5. The UI renders the answer, supporting snippets, related context, and annotations.

## Run

Install dependencies:

```bash
pip install fastapi uvicorn python-multipart transformers torch PyPDF2 jinja2
```

Start the server:

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## API

- `GET /`: main HTML interface
- `POST /api/upload`: upload a PDF or TXT document
- `POST /api/ask`: ask a question, optionally scoped to a selected section
- `GET /api/document/{document_id}/file`: preview the uploaded original file

## Project Structure

- `app.py`: FastAPI backend, document parsing, section detection, and QA pipeline
- `templates/index.html`: main HTML layout and styling
- `static/app.js`: frontend behavior for uploads, section switching, and question flow

## Tech Stack

- Python 3.10+
- FastAPI
- Jinja2 Templates
- Transformers
- PyTorch
- PyPDF2
- Tailwind CSS CDN

## Notes

- Section detection is intentionally simple and heuristic-based.
- Uploaded documents are currently stored in memory, not persisted.
- The QA model works best for short factual answers rather than long-form summarization.

## Next Improvements

- Add true heading extraction from PDF structure
- Persist uploaded documents and session state
- Highlight answer spans inside the document preview
- Improve retrieval for larger files with vector search
