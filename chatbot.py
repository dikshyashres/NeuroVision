"""
chatbot.py — SBERT + Gemini RAG chatbot for NeuroVision

How it works:
1. At startup, SBERT encodes the entire knowledge base into embeddings
2. When a user sends a message, SBERT finds the top-K most relevant chunks
3. Those chunks are injected into Gemini's system prompt as context
4. Gemini generates a grounded, accurate reply using that context
"""

import requests
from sentence_transformers import SentenceTransformer, util

from knowledge_base import get_all_knowledge

# -----------------------------
# Config
# -----------------------------
GEMINI_API_KEY = "enter your api key"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
)
TOP_K_CHUNKS   = 10  # how many knowledge chunks to retrieve per query
HISTORY_WINDOW = 8   # how many past messages to send to Gemini

# -----------------------------
# SBERT Setup (loads once at import)
# -----------------------------
print("🔄 Loading SBERT model...")
_sbert_model = SentenceTransformer("all-MiniLM-L6-v2")  # fast & lightweight

print("🔄 Encoding knowledge base...")
_knowledge_chunks: list[str] = get_all_knowledge()
_knowledge_embeddings = _sbert_model.encode(
    _knowledge_chunks,
    convert_to_tensor=True,
    show_progress_bar=False,
)
print(f"✅ SBERT ready — {len(_knowledge_chunks)} chunks indexed")

# -----------------------------
# RAG Retrieval
# -----------------------------

def retrieve_relevant_chunks(query: str, top_k: int = TOP_K_CHUNKS) -> list[str]:
    """
    Use SBERT cosine similarity to find the top-K knowledge chunks
    most semantically relevant to the user's query.
    """
    query_embedding = _sbert_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, _knowledge_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k].tolist()
    return [_knowledge_chunks[i] for i in top_indices]

# -----------------------------
# Prompt Builders
# -----------------------------

def _build_analysis_block(context: dict) -> str:
    """Turn the scan context dict into a readable text block."""
    if not context or not context.get("tumor_type"):
        return "No scan has been analyzed yet in this session."

    tumor_type         = context.get("tumor_type", "Unknown")
    highest_confidence = context.get("highest_confidence", "N/A")
    conf_dict          = context.get("confidence", {})
    has_gradcam        = context.get("has_gradcam", False)

    scores_text = (
        ", ".join(f"{k}: {v}" for k, v in conf_dict.items()) if conf_dict else "N/A"
    )
    gradcam_text = (
        "Grad-CAM heatmap was generated. Red/warm regions = areas the model focused on "
        "(likely tumor). Blue/green = low attention."
        if has_gradcam
        else "No Grad-CAM generated (no tumor detected)."
    )

    return (
        f"CURRENT SCAN RESULTS:\n"
        f"- Detected: {tumor_type}\n"
        f"- Confidence: {highest_confidence}\n"
        f"- All class scores: {scores_text}\n"
        f"- Grad-CAM: {gradcam_text}"
    )


def _build_system_prompt(analysis_block: str, retrieved_chunks: list[str]) -> str:
    """Build the full Gemini system prompt with RAG context injected."""
    rag_context = "\n".join(f"• {chunk}" for chunk in retrieved_chunks)

    return f"""You are NeuroVision Assistant — a helpful, empathetic AI inside a brain MRI tumor detection system.

The VGG16 model classifies MRI scans into:
- Glioma: Tumor from glial cells; can be low-grade (slow) or high-grade (aggressive)
- Meningioma: Tumor from brain/spinal cord lining; usually benign and slow-growing
- Pituitary: Tumor in pituitary gland; usually benign, affects hormone production
- No Tumor: No tumor found

Grad-CAM = heatmap showing which MRI regions the AI focused on. Red/yellow = high activation. Blue = low activation.

{analysis_block}

RELEVANT MEDICAL KNOWLEDGE (retrieved for this query):
{rag_context}

INSTRUCTIONS:
- Use the retrieved knowledge above to give accurate, grounded answers
- Explain scan results clearly in simple language
- Answer questions about Grad-CAM, tumor types, symptoms, and treatments
- Be warm and empathetic — users may be patients or worried family members
- Keep answers concise (2-4 sentences) unless more detail is needed
- ALWAYS remind users to consult a qualified doctor — this tool is NOT a medical diagnosis
- Do not fabricate medical information outside the knowledge provided
- If user asks about accuracy, dataset, kaggle, model performance or results, answer using the retrieved knowledge chunks above"""

# -----------------------------
# Main Chat Function
# -----------------------------

def get_chat_reply(user_message: str, context: dict, history: list) -> str:
    """
    Full RAG pipeline:
      1. Retrieve top-K relevant chunks via SBERT
      2. Build system prompt with retrieved context + scan results
      3. Send to Gemini and return the reply

    Returns an error string (never raises) for API failures.
    """
    if not user_message:
        raise ValueError("Empty message")

    # Step 1 — SBERT retrieval
    retrieved_chunks = retrieve_relevant_chunks(user_message)
    print(f"🔍 Retrieved {len(retrieved_chunks)} chunks for: '{user_message[:60]}'")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"   [{i}] {chunk[:80]}...")

    # Step 2 — Build prompt
    analysis_block = _build_analysis_block(context)
    system_prompt  = _build_system_prompt(analysis_block, retrieved_chunks)

    # Step 3 — Build Gemini conversation history
    gemini_contents = []
    for msg in history[-HISTORY_WINDOW:]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    gemini_contents.append({"role": "user", "parts": [{"text": user_message}]})

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": gemini_contents,
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 512},
    }

    # Step 4 — Call Gemini
    try:
        response = requests.post(
            GEMINI_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]

    except requests.exceptions.Timeout:
        return "Sorry, the response timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        print(f"❌ Gemini HTTP error {status}: {e.response.text}")
        if status == 429:
            return "⏳ Too many requests — free API limit hit. Please wait 30 seconds."
        if status == 400:
            return "⚠️ Invalid API key. Please check your GEMINI_API_KEY."
        return f"⚠️ API error ({status}). Please try again in a moment."
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return "Sorry, something went wrong. Please try again."