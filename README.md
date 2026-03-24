# Document QA System
This project is now a FastAPI-based document question answering app with a static HTML frontend. Users can upload PDF or TXT files, preview the processed document, and ask transformer-powered questions against its contents.

## Features
- Upload PDF or TXT documents from the browser.
- Parse document text with `PyPDF2` for PDFs and plain decoding for text files.
- Ask questions with `distilbert-base-cased-distilled-squad` using a FastAPI backend.
- Use the provided archival-style HTML interface instead of Streamlit.
- Preview the uploaded source file directly from the UI.

## Run
Install the required packages first:

```bash
pip install fastapi uvicorn python-multipart transformers torch PyPDF2 jinja2
```

Start the app:

```bash
uvicorn app:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Project Structure
- `app.py`: FastAPI backend and transformer QA pipeline
- `templates/index.html`: frontend markup and Tailwind-based layout
- `static/app.js`: client-side upload and question flow

## Technologies Used
- Python 3.10+
- FastAPI
- Jinja2 templates
- Transformers
- PyTorch
- PyPDF2
- Tailwind CSS CDN

## Future Improvements
- Persist uploaded documents instead of storing them in memory.
- Add citation highlighting and page anchoring.
- Add retrieval indexing for larger documents.
- Replace placeholder chapter labels with real structural extraction.
