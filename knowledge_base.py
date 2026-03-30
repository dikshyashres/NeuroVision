"""
knowledge_base.py — NeuroVision RAG Pipeline Knowledge Loader
==============================================================
All medical knowledge lives in:  knowledge_docs/knowledge_base.txt

HOW IT WORKS:
  1. knowledge_base.txt is the single source of truth.
  2. This file reads it, strips comments (#) and blank lines,
     and returns each remaining line as one knowledge fact.
  3. Any extra .txt or .pdf files placed in knowledge_docs/
     are also loaded and chunked automatically.

TO ADD NEW KNOWLEDGE:
  - Open knowledge_docs/knowledge_base.txt
  - Add a new sentence on its own line under the right section
  - No code changes needed — restart the app and it reloads.
"""

import os
import re

# ------------------------------------------------------------------
# Path to the primary knowledge base text file
# ------------------------------------------------------------------
KNOWLEDGE_DOCS_FOLDER = "knowledge_docs"
PRIMARY_KB_FILE       = os.path.join(KNOWLEDGE_DOCS_FOLDER, "knowledge_base.txt")


# ------------------------------------------------------------------
# Load the primary structured knowledge_base.txt
# ------------------------------------------------------------------

def load_primary_knowledge(filepath: str = PRIMARY_KB_FILE) -> list[str]:
    """
    Read knowledge_base.txt and return a list of facts.

    Rules applied:
      - Lines starting with # are treated as comments → skipped.
      - Empty or whitespace-only lines               → skipped.
      - Every other line                             → one knowledge fact.
    """
    if not os.path.exists(filepath):
        print(f"⚠️  Primary knowledge file not found: {filepath}")
        return []

    facts = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:                  # skip blank lines
                continue
            if line.startswith("#"):      # skip comment lines
                continue
            facts.append(line)

    print(f"✅ Loaded {len(facts)} facts from '{filepath}'")
    return facts


# ------------------------------------------------------------------
# Chunker — used for large external .txt / .pdf files
# ------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = 200, overlap: int = 30) -> list[str]:
    """
    Split a large block of text into overlapping word-level chunks.

    Args:
        text       : Plain text to split.
        chunk_size : Maximum number of words per chunk.
        overlap    : Number of words shared between consecutive chunks
                     (helps the retriever not miss cross-boundary context).

    Returns:
        List of text chunk strings.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ------------------------------------------------------------------
# Loaders for external .txt and .pdf files
# ------------------------------------------------------------------

def _load_txt_file(filepath: str) -> list[str]:
    """Load a plain .txt file and return word-level chunks."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    text = re.sub(r"\s+", " ", text).strip()
    return _chunk_text(text)


def _load_pdf_file(filepath: str) -> list[str]:
    """
    Load a PDF file and return word-level chunks.
    Requires: pip install pypdf
    """
    try:
        from pypdf import PdfReader
        reader    = PdfReader(filepath)
        full_text = " ".join(page.extract_text() or "" for page in reader.pages)
        full_text = re.sub(r"\s+", " ", full_text).strip()
        return _chunk_text(full_text)
    except ImportError:
        print("⚠️  pypdf not installed. Run: pip install pypdf")
        return []
    except Exception as e:
        print(f"⚠️  Could not read PDF '{filepath}': {e}")
        return []


# ------------------------------------------------------------------
# Scan knowledge_docs/ for any additional .txt / .pdf files
# ------------------------------------------------------------------

def load_extra_documents(folder: str = KNOWLEDGE_DOCS_FOLDER) -> list[str]:
    """
    Scan the knowledge_docs folder for supplementary documents
    (any .txt or .pdf that is NOT the primary knowledge_base.txt).

    Drop extra medical PDFs or text files into knowledge_docs/ and
    they will be picked up automatically on the next run.

    Returns:
        List of text chunks from all extra documents.
    """
    if not os.path.exists(folder):
        return []

    primary_name = os.path.basename(PRIMARY_KB_FILE)   # "knowledge_base.txt"
    chunks = []

    for filename in sorted(os.listdir(folder)):
        if filename == primary_name:                    # skip the primary file
            continue

        filepath = os.path.join(folder, filename)

        if filename.endswith(".txt"):
            print(f"📄 Loading extra text file : {filename}")
            chunks.extend(_load_txt_file(filepath))

        elif filename.endswith(".pdf"):
            print(f"📄 Loading extra PDF file  : {filename}")
            chunks.extend(_load_pdf_file(filepath))

    if chunks:
        print(f"✅ Loaded {len(chunks)} extra chunks from supplementary documents")
    return chunks


# ------------------------------------------------------------------
# Public API — called by the RAG pipeline
# ------------------------------------------------------------------

def get_all_knowledge() -> list[str]:
    """
    Returns the complete knowledge base used by the RAG retriever:

        Primary facts  (knowledge_base.txt, line-by-line)
      + Extra chunks   (any other .txt / .pdf in knowledge_docs/)

    Returns:
        List of strings — each string is one retrievable knowledge unit.
    """
    primary = load_primary_knowledge()
    extra   = load_extra_documents()
    combined = primary + extra

    print(f"✅ Total knowledge base: {len(combined)} retrievable units")
    return combined