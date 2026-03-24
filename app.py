import re
from collections import Counter

import PyPDF2
import streamlit as st
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODEL_NAME = "distilbert-base-cased-distilled-squad"
MAX_LENGTH = 384
DOC_STRIDE = 128
MAX_ANSWER_TOKENS = 50
TOP_K = 8
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP = 180
TOP_RETRIEVAL_CHUNKS = 5

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
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

QUESTION_WORDS = {"what", "when", "where", "which", "who", "why", "how", "define", "explain"}
LOW_SIGNAL_ANSWERS = {
    "",
    "test",
    "chapter",
    "section",
    "figure",
    "table",
    "page",
    "example",
}


@st.cache_resource(show_spinner=False)
def load_qa_model():
    """Loads the QA model once and reuses it across Streamlit reruns."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def normalize_text(text):
    """Cleans common PDF extraction issues while preserving section structure."""
    text = text.replace("\r", "\n")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf(file_path):
    """Reads a PDF file and returns cleaned text."""
    pdf = PyPDF2.PdfReader(file_path)
    pages = []

    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)

    return normalize_text("\n\n".join(pages))


def split_chapters(text):
    """
    Splits text into chapters based on "Chapter X" headings.
    Returns a dict: { "Chapter 1": "text", "Chapter 2": "text", ... }
    """
    chapters = {}
    current_chapter = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if re.match(r"^Chapter\s+\d+", line, re.IGNORECASE):
            current_chapter = re.split(r"\s(?:-|\u2014)\s", line, maxsplit=1)[0].strip()
            chapters[current_chapter] = line + "\n"
            continue

        if current_chapter:
            chapters[current_chapter] += line + "\n"

    return {name: normalize_text(content) for name, content in chapters.items()}


def tokenize_terms(text):
    return [token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if token not in STOPWORDS]


def normalize_question(question):
    """Turns shorthand prompts into a question the QA model can handle better."""
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


def chunk_text(text, max_chars=MAX_CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """Builds overlapping chunks, preferring paragraph boundaries."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return [text]

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

    return chunks or [text]


def score_text_match(text, question, question_terms):
    """Scores how relevant a chunk or sentence is for the question."""
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


def select_relevant_chunks(context, question):
    """Retrieves the most relevant chunks before running QA."""
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


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def term_coverage(text, question_terms):
    if not question_terms:
        return 0.0

    text_terms = set(tokenize_terms(text))
    if not text_terms:
        return 0.0

    matched = sum(1 for term in question_terms if term in text_terms)
    return matched / len(question_terms)


def best_matching_sentence(context, question):
    """Finds the most relevant sentence and expands short heading-like matches."""
    question_terms = tokenize_terms(question)
    sentences = split_sentences(context)
    best_sentence = ""
    best_score = 0.0
    best_index = -1

    for index, sentence in enumerate(sentences):
        score = score_text_match(sentence, question, question_terms)
        if score > best_score:
            best_sentence = sentence
            best_score = score
            best_index = index

    if best_index >= 0 and len(best_sentence) < 90 and best_index + 1 < len(sentences):
        next_sentence = sentences[best_index + 1]
        combined = f"{best_sentence} {next_sentence}".strip()
        combined_score = score_text_match(combined, question, question_terms)
        if combined_score >= best_score:
            best_sentence = combined
            best_score = combined_score

    return best_sentence, best_score


def answer_is_low_signal(answer, raw_question):
    cleaned = answer.strip().lower().strip(".,:;!?")
    if cleaned in LOW_SIGNAL_ANSWERS:
        return True

    if len(cleaned) < 3:
        return True

    question_terms = set(tokenize_terms(raw_question))
    answer_terms = set(tokenize_terms(cleaned))
    if answer_terms and answer_terms.issubset(question_terms) and len(answer_terms) <= 2:
        return True

    return False


def sentence_with_answer(chunk, answer):
    for sentence in split_sentences(chunk):
        if answer.lower() in sentence.lower():
            return sentence
    return answer


def run_qa_on_chunk(chunk, question):
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


def ask_question(context, raw_question):
    """Retrieves relevant context, runs QA, and falls back to a relevant sentence."""
    normalized = normalize_question(raw_question)
    normalized_terms = tokenize_terms(normalized)
    relevant_chunks = select_relevant_chunks(context, normalized)

    keyword_sentence, keyword_score = best_matching_sentence(context, raw_question)
    question_terms = tokenize_terms(raw_question)
    keyword_coverage = term_coverage(keyword_sentence, question_terms)
    keyword_like_query = (
        raw_question.strip()
        and not raw_question.strip().endswith("?")
        and not any(word in raw_question.lower().split() for word in QUESTION_WORDS)
        and len(question_terms) <= 6
    )

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

    if keyword_like_query and keyword_sentence and keyword_score >= 4.0:
        if best_candidate is None:
            return keyword_sentence

        candidate_text = best_candidate["snippet"]
        candidate_coverage = term_coverage(candidate_text, question_terms)
        exact_phrase_in_keyword = raw_question.lower() in keyword_sentence.lower()
        exact_phrase_in_candidate = raw_question.lower() in candidate_text.lower()

        if answer_is_low_signal(best_candidate["answer"], raw_question):
            return keyword_sentence
        if exact_phrase_in_keyword and not exact_phrase_in_candidate:
            return keyword_sentence
        if keyword_coverage > candidate_coverage and best_candidate["margin"] < 5.0:
            return keyword_sentence
        if keyword_coverage >= 0.75 and best_candidate["margin"] < 3.5:
            return keyword_sentence

    if best_candidate and not answer_is_low_signal(best_candidate["answer"], raw_question):
        if len(best_candidate["answer"].split()) <= 3 and len(best_candidate["snippet"]) <= 240:
            return best_candidate["snippet"]
        return best_candidate["answer"]

    if keyword_sentence:
        return keyword_sentence

    return "No confident answer found."


def main():
    st.set_page_config(page_title="Document QA System", page_icon="DOC", layout="wide")
    st.markdown(
        """
        <h1 style='text-align: center; color: #4B0082;'>Document Question Answering System</h1>
        <p style='text-align: center; color: #6A5ACD;'>Upload a PDF or TXT file and ask questions about its content.</p>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload a PDF or TXT file:", type=["pdf", "txt"], key="file_uploader")

    if not uploaded_file:
        return

    if uploaded_file.type == "application/pdf":
        text = read_pdf(uploaded_file)
    else:
        text = normalize_text(uploaded_file.read().decode("utf-8"))

    chapters = split_chapters(text)

    st.sidebar.header("Chapters in Document")
    if chapters:
        for chapter in chapters:
            st.sidebar.write(chapter)
    else:
        st.sidebar.write("No chapter headings detected.")

    st.markdown("### Ask a Question About Your Document")
    question = st.text_input(
        "Enter your question here:",
        placeholder="Example: What is the sigmoid function?",
    )

    if not question:
        return

    chapter_match = re.search(r"Chapter (\d+)", question, re.IGNORECASE)
    if chapter_match:
        chapter_key = f"Chapter {chapter_match.group(1)}"
        context = chapters.get(chapter_key, text)
    else:
        context = text

    with st.spinner("Finding the best answer..."):
        answer = ask_question(context, question)

    st.markdown(
        f"""
        <div style='background-color:#E6E6FA; padding:15px; border-radius:10px;'>
            <h3 style='color:#4B0082;'>Answer:</h3>
            <p style='font-size:18px; color:#333;'>{answer}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
