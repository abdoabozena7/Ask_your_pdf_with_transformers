import io
import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import PyPDF2
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODEL_NAME = "distilbert-base-cased-distilled-squad"
MAX_LENGTH = 384
DOC_STRIDE = 128
MAX_ANSWER_TOKENS = 50
TOP_K = 8
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP = 180
TOP_RETRIEVAL_CHUNKS = 5
SUMMARY_SENTENCES = 3
KEYWORD_LIMIT = 6
FALLBACK_SECTION_COUNT = 4

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

QUESTION_WORDS = {"what", "when", "where", "which", "who", "why", "how", "define", "explain"}

SECTION_DEFINITIONS = [
    {
        "id": "introduction",
        "title": "Introduction",
        "icon": "book",
        "patterns": [r"\bintroduction\b", r"\boverview\b", r"\babstract\b"],
    },
    {
        "id": "methodology",
        "title": "Methodology",
        "icon": "science",
        "patterns": [r"\bmethodology\b", r"\bmethods?\b", r"\bapproach\b"],
    },
    {
        "id": "key-findings",
        "title": "Key Findings",
        "icon": "insights",
        "patterns": [r"\bkey findings\b", r"\bfindings\b", r"\bresults?\b", r"\bdiscussion\b"],
    },
    {
        "id": "conclusion",
        "title": "Conclusion",
        "icon": "analytics",
        "patterns": [r"\bconclusion\b", r"\bsummary\b", r"\bclosing\b", r"\brecommendations?\b"],
    },
]

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


@dataclass
class DocumentSection:
    section_id: str
    title: str
    icon: str
    text: str
    excerpt: str
    keywords: list[str]
    detected: bool


@dataclass
class StoredDocument:
    document_id: str
    name: str
    media_type: str
    file_bytes: bytes
    text: str
    page_count: int
    word_count: int
    read_minutes: int
    doc_type: str
    snapshot_sentences: list[str]
    keywords: list[str]
    sections: list[DocumentSection]


class AskRequest(BaseModel):
    document_id: str
    question: str
    section_id: str | None = None


documents: dict[str, StoredDocument] = {}

app = FastAPI(title="Ask Your PDF with Transformers")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@lru_cache(maxsize=1)
def load_qa_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_terms(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if token not in STOPWORDS]


def normalize_question(question: str) -> str:
    question = question.strip()
    lowered = question.lower()
    terms = tokenize_terms(question)

    if not question:
        return question
    if question.endswith("?"):
        return question
    if any(word in lowered.split() for word in QUESTION_WORDS):
        return question
    if len(terms) <= 6:
        return f"What is {question}?"
    return question


def parse_document(file_name: str, file_type: str, file_bytes: bytes) -> dict:
    is_pdf = file_type == "application/pdf" or file_name.lower().endswith(".pdf")

    if is_pdf:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
        text = normalize_text("\n\n".join(pages))
        page_count = len(reader.pages)
        doc_type = "PDF"
    else:
        text = normalize_text(file_bytes.decode("utf-8", errors="ignore"))
        page_count = max(1, math.ceil(len(text) / 2500))
        doc_type = "TXT"

    word_count = len(text.split())
    read_minutes = max(1, math.ceil(word_count / 220)) if text else 1

    return {
        "name": file_name,
        "type": doc_type,
        "text": text,
        "page_count": page_count,
        "word_count": word_count,
        "read_minutes": read_minutes,
    }


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return [text] if text else []

    chunks = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        start = 0
        while start < len(paragraph):
            end = min(start + max_chars, len(paragraph))
            chunks.append(paragraph[start:end].strip())
            if end >= len(paragraph):
                break
            start = max(end - overlap, start + 1)
        current = ""

    if current:
        chunks.append(current)

    return chunks


def score_text_match(text: str, question: str, question_terms: list[str]) -> float:
    lowered_text = text.lower()
    text_terms = tokenize_terms(text)
    if not text_terms:
        return 0.0

    text_counts = Counter(text_terms)
    unique_question_terms = set(question_terms)
    matched_terms = sum(1 for term in unique_question_terms if term in text_counts)
    match_frequency = sum(text_counts[term] for term in unique_question_terms)
    exact_phrase_bonus = 4.0 if question.lower().strip("?") in lowered_text else 0.0

    return (
        matched_terms * 2.5
        + match_frequency * 0.35
        + exact_phrase_bonus
        + min(len(text) / 600.0, 2.0)
    )


def select_relevant_chunks(context: str, question: str) -> list[str]:
    question_terms = tokenize_terms(question)
    chunks = chunk_text(context)
    scored_chunks = []

    for chunk in chunks:
        score = score_text_match(chunk, question, question_terms)
        if score > 0:
            scored_chunks.append((score, chunk))

    if not scored_chunks:
        return chunks[:TOP_RETRIEVAL_CHUNKS]

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:TOP_RETRIEVAL_CHUNKS]]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def sentence_with_answer(chunk: str, answer: str) -> str:
    for sentence in split_sentences(chunk):
        if answer.lower() in sentence.lower():
            return sentence
    return answer


def run_qa_on_chunk(chunk: str, question: str):
    tokenizer, model = load_qa_model()
    encoded = tokenizer(
        question,
        chunk,
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )

    best_candidate = None

    for feature_index in range(len(encoded["input_ids"])):
        offsets = encoded["offset_mapping"][feature_index].tolist()
        sequence_ids = encoded.sequence_ids(feature_index)
        start_logits = outputs.start_logits[feature_index]
        end_logits = outputs.end_logits[feature_index]
        null_score = (start_logits[0] + end_logits[0]).item()

        top_starts = torch.topk(start_logits, k=TOP_K).indices.tolist()
        top_ends = torch.topk(end_logits, k=TOP_K).indices.tolist()

        for start_index in top_starts:
            if sequence_ids[start_index] != 1:
                continue

            for end_index in top_ends:
                if sequence_ids[end_index] != 1:
                    continue
                if end_index < start_index:
                    continue
                if end_index - start_index + 1 > MAX_ANSWER_TOKENS:
                    continue

                start_char = offsets[start_index][0]
                end_char = offsets[end_index][1]
                answer = chunk[start_char:end_char].strip()
                if not answer:
                    continue

                answer_score = start_logits[start_index].item() + end_logits[end_index].item()
                score_margin = answer_score - null_score
                candidate = {
                    "answer": answer,
                    "score": answer_score,
                    "margin": score_margin,
                    "snippet": sentence_with_answer(chunk, answer),
                }

                if best_candidate is None or candidate["margin"] > best_candidate["margin"]:
                    best_candidate = candidate

    return best_candidate


def confidence_label(margin: float) -> str:
    if margin >= 8:
        return "Strong"
    if margin >= 4:
        return "Moderate"
    return "Tentative"


def ask_question(context: str, raw_question: str) -> dict:
    normalized = normalize_question(raw_question)
    normalized_terms = tokenize_terms(normalized)
    relevant_chunks = select_relevant_chunks(context, normalized)

    best_candidate = None

    for chunk in relevant_chunks:
        retrieval_score = score_text_match(chunk, normalized, normalized_terms)
        candidate = run_qa_on_chunk(chunk, normalized)
        if candidate is None:
            continue

        candidate["combined_score"] = candidate["margin"] + retrieval_score
        candidate["retrieval_score"] = retrieval_score

        if best_candidate is None or candidate["combined_score"] > best_candidate["combined_score"]:
            best_candidate = candidate

    if best_candidate is None:
        return {
            "answer": "No confident transformer answer found for that question.",
            "snippet": "",
            "confidence": "Unavailable",
            "margin": None,
            "found": False,
        }

    return {
        "answer": best_candidate["answer"],
        "snippet": best_candidate["snippet"],
        "confidence": confidence_label(best_candidate["margin"]),
        "margin": best_candidate["margin"],
        "found": True,
    }


def extract_keywords(text: str, limit: int = KEYWORD_LIMIT) -> list[str]:
    counts = Counter(token for token in tokenize_terms(text) if len(token) > 3)
    return [token for token, _ in counts.most_common(limit)]


def build_extractive_snapshot(text: str, sentence_limit: int = SUMMARY_SENTENCES) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    focus_window = sentences[: min(len(sentences), 40)]
    term_counts = Counter(tokenize_terms(" ".join(focus_window)))
    scored = []

    for index, sentence in enumerate(focus_window):
        terms = tokenize_terms(sentence)
        if not terms:
            continue

        density = sum(term_counts[term] for term in terms)
        length_penalty = abs(len(sentence) - 170) / 170
        position_bonus = max(0, 6 - index) * 0.5
        score = density + position_bonus - length_penalty
        scored.append((score, index, sentence))

    top_sentences = sorted(scored, reverse=True)[:sentence_limit]
    return [sentence for _, _, sentence in sorted(top_sentences, key=lambda item: item[1])]


def fallback_section_texts(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return [text] * FALLBACK_SECTION_COUNT if text else [""] * FALLBACK_SECTION_COUNT

    chunk_size = max(1, math.ceil(len(paragraphs) / FALLBACK_SECTION_COUNT))
    grouped = []
    for index in range(FALLBACK_SECTION_COUNT):
        start = index * chunk_size
        end = start + chunk_size
        grouped.append("\n\n".join(paragraphs[start:end]).strip())

    filled = [group for group in grouped if group]
    if not filled:
        return [""] * FALLBACK_SECTION_COUNT

    while len(filled) < FALLBACK_SECTION_COUNT:
        filled.append(filled[-1])
    return filled[:FALLBACK_SECTION_COUNT]


def find_section_span(text: str, patterns: list[str]):
    candidates = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            candidates.append(match.start())
    if not candidates:
        return None
    return min(candidates)


def strip_section_heading(section_text: str, title: str) -> str:
    lines = [line.strip() for line in section_text.splitlines()]
    if not lines:
        return section_text

    title_pattern = re.compile(rf"^{re.escape(title)}\b[:\-\s]*$", flags=re.IGNORECASE)
    first_line = lines[0]
    if title_pattern.match(first_line):
        cleaned = "\n".join(lines[1:]).strip()
        return cleaned or section_text.strip()
    return section_text.strip()


def detect_document_sections(text: str) -> list[DocumentSection]:
    fallback_texts = fallback_section_texts(text)
    positions = []

    for definition in SECTION_DEFINITIONS:
        position = find_section_span(text, definition["patterns"])
        if position is not None:
            positions.append((position, definition))

    positions.sort(key=lambda item: item[0])
    spans: dict[str, tuple[int, int]] = {}
    for index, (start, definition) in enumerate(positions):
        end = len(text)
        if index + 1 < len(positions):
            end = positions[index + 1][0]
        spans[definition["id"]] = (start, end)

    sections = []
    for fallback_index, definition in enumerate(SECTION_DEFINITIONS):
        span = spans.get(definition["id"])
        detected = span is not None
        if detected:
            section_text = text[span[0]:span[1]].strip()
        else:
            section_text = fallback_texts[fallback_index].strip()
        if not section_text:
            section_text = text[:MAX_CHUNK_CHARS].strip()
        section_text = strip_section_heading(section_text, definition["title"])

        snapshot = build_extractive_snapshot(section_text, sentence_limit=1)
        sections.append(
            DocumentSection(
                section_id=definition["id"],
                title=definition["title"],
                icon=definition["icon"],
                text=section_text,
                excerpt=snapshot[0] if snapshot else section_text[:220].strip(),
                keywords=extract_keywords(section_text, limit=3),
                detected=detected,
            )
        )

    return sections


def get_document_section(document: StoredDocument, section_id: str | None):
    if not section_id:
        return None
    for section in document.sections:
        if section.section_id == section_id:
            return section
    return None


def serialize_document(document: StoredDocument) -> dict:
    return {
        "document_id": document.document_id,
        "name": document.name,
        "type": document.doc_type,
        "page_count": document.page_count,
        "word_count": document.word_count,
        "read_minutes": document.read_minutes,
        "snapshot_sentences": document.snapshot_sentences,
        "keywords": document.keywords,
        "sections": [
            {
                "section_id": section.section_id,
                "title": section.title,
                "icon": section.icon,
                "excerpt": section.excerpt,
                "keywords": section.keywords,
                "detected": section.detected,
            }
            for section in document.sections
        ],
        "preview_url": f"/api/document/{document.document_id}/file",
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    parsed = parse_document(file.filename or "document", file.content_type or "", file_bytes)
    if not parsed["text"]:
        raise HTTPException(status_code=400, detail="The uploaded file did not produce readable text.")

    document_id = str(uuid.uuid4())
    stored = StoredDocument(
        document_id=document_id,
        name=parsed["name"],
        media_type=file.content_type or "application/octet-stream",
        file_bytes=file_bytes,
        text=parsed["text"],
        page_count=parsed["page_count"],
        word_count=parsed["word_count"],
        read_minutes=parsed["read_minutes"],
        doc_type=parsed["type"],
        snapshot_sentences=build_extractive_snapshot(parsed["text"]),
        keywords=extract_keywords(parsed["text"]),
        sections=detect_document_sections(parsed["text"]),
    )
    documents[document_id] = stored
    return JSONResponse(serialize_document(stored))


@app.post("/api/ask")
async def ask_document_question(payload: AskRequest):
    document = documents.get(payload.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found. Upload the file again.")
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    active_section = get_document_section(document, payload.section_id)
    question_context = active_section.text if active_section and active_section.text.strip() else document.text
    result = ask_question(question_context, payload.question)
    snapshot_sentences = build_extractive_snapshot(question_context)
    keywords = extract_keywords(question_context)

    return JSONResponse(
        {
            "question": payload.question,
            "result": result,
            "snapshot_sentences": snapshot_sentences,
            "keywords": keywords,
            "active_section": {
                "section_id": active_section.section_id,
                "title": active_section.title,
            }
            if active_section
            else None,
        }
    )


@app.get("/api/document/{document_id}/file")
async def preview_document_file(document_id: str):
    document = documents.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    headers = {"Content-Disposition": f'inline; filename="{document.name}"'}
    return StreamingResponse(io.BytesIO(document.file_bytes), media_type=document.media_type, headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
