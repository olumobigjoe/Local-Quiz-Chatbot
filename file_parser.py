"""
file_parser.py
---------------
Handles reading text out of uploaded .docx / .pdf files and splitting that
text into small chunks so we never send a whole document to qwen3:4b in one
go. This matters a lot on an 8GB RAM / CPU-only machine: a huge single
prompt is slow, can time out, and risks the model losing track of the
instructions entirely.
"""

import os
import re
from typing import List

# python-docx is used for Word documents
from docx import Document

# pdfplumber gives better text layout extraction than PyPDF2 in most cases,
# but PyPDF2 is kept as a fallback in case a PDF is malformed / encrypted
# and pdfplumber chokes on it.
import pdfplumber
from PyPDF2 import PdfReader


def extract_text_from_docx(file_path: str) -> str:
    """Read all paragraph text out of a .docx file, in order."""
    doc = Document(file_path)
    # Join non-empty paragraphs with newlines so paragraph breaks survive,
    # which helps the chunker later split on natural boundaries.
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_pdf(file_path: str) -> str:
    """
    Read text out of a .pdf file.
    Tries pdfplumber first (better at preserving reading order / layout).
    Falls back to PyPDF2 if pdfplumber fails (e.g. on some scanned or
    unusually-encoded PDFs).
    """
    text_parts: List[str] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
        if text_parts:
            return "\n".join(text_parts)
    except Exception:
        # Swallow and fall through to the PyPDF2 fallback below.
        pass

    # Fallback path
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
    except Exception as exc:
        raise RuntimeError(f"Could not extract text from PDF: {exc}")

    return "\n".join(text_parts)


def extract_text(file_path: str) -> str:
    """
    Dispatch to the right extractor based on file extension.
    Raises ValueError for unsupported file types so the UI can show a
    friendly error instead of crashing.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        text = extract_text_from_docx(file_path)
    elif ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Please upload a .docx or .pdf file.")

    text = text.strip()
    if not text:
        raise ValueError(
            "No readable text was found in this file. "
            "It may be a scanned/image-only document, which this tool can't read."
        )
    return text


def chunk_text(text: str, chunk_word_size: int = 900) -> List[str]:
    """
    Split text into word-count-limited chunks, breaking on paragraph/sentence
    boundaries where possible rather than mid-sentence. Keeping chunks small
    (roughly 900 words ~= 1200-1500 tokens) keeps each Ollama call fast and
    memory-light on constrained hardware.
    """
    # Normalize whitespace, then split into paragraphs first.
    paragraphs = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]

    chunks: List[str] = []
    current_words: List[str] = []

    for para in paragraphs:
        para_words = para.split()
        if len(current_words) + len(para_words) > chunk_word_size and current_words:
            chunks.append(" ".join(current_words))
            current_words = []
        current_words.extend(para_words)

        # Safety net: if a single paragraph itself is longer than the chunk
        # size (rare, but happens with dense reports), hard-split it too.
        while len(current_words) > chunk_word_size:
            chunks.append(" ".join(current_words[:chunk_word_size]))
            current_words = current_words[chunk_word_size:]

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks if chunks else [text]
