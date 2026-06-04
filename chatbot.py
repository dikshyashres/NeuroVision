"""
chatbot.py — SBERT + Gemini RAG chatbot for NeuroVision

RAG = Retrieval Augmented Generation

FLOW:
1. Convert knowledge base → embeddings (SBERT)
2. User asks question
3. Find most relevant chunks using similarity
4. Send those chunks + context to Gemini
5. Gemini generates accurate answer
"""

# ==============================
# Imports
# ==============================
import requests
# Used to call Gemini API

from sentence_transformers import SentenceTransformer, util
# SBERT model for embeddings + similarity

from knowledge_base import get_all_knowledge
# Custom function that returns all medical knowledge text

<<<<<<< Updated upstream
# -----------------------------
# Config
# -----------------------------

# Add as many Gemini API keys as you want here.
# When one hits the rate limit (429) or daily limit, the next key is tried automatically.
GEMINI_API_KEYS = [
    "enter your api key",  # Key 1
    "enter your api key",             # Key 2
    "enter your api key",              # Key 3 
]

=======
# ==============================
# CONFIGURATION
# ==============================

# List of Gemini API keys (for rotation if limit hits)
GEMINI_API_KEYS = [
    ""   # Add your API keys here
]

# Gemini API endpoint
>>>>>>> Stashed changes
GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent?key={api_key}"
)
<<<<<<< Updated upstream

TOP_K_CHUNKS   = 10  # how many knowledge chunks to retrieve per query
HISTORY_WINDOW = 8   # how many past messages to send to Gemini

# -----------------------------
# Key Rotation State
# -----------------------------
_current_key_index = 0  # tracks which key is currently active

def _get_active_url() -> str:
    """Return the Gemini URL using the currently active API key."""
    key = GEMINI_API_KEYS[_current_key_index]
    return GEMINI_BASE_URL.format(api_key=key)

def _rotate_key() -> bool:
    """
    Rotate to the next available API key.
    Returns True if a new key is available, False if all keys are exhausted.
    """
    global _current_key_index
    next_index = _current_key_index + 1
    if next_index < len(GEMINI_API_KEYS):
        _current_key_index = next_index
        print(f"🔑 Rotated to API key {_current_key_index + 1} of {len(GEMINI_API_KEYS)}")
        return True
    print("❌ All API keys exhausted.")
    return False

# -----------------------------
# SBERT Setup (loads once at import)
# -----------------------------
print(" Loading SBERT model...")
_sbert_model = SentenceTransformer("all-MiniLM-L6-v2")  # fast & lightweight

print(" Encoding knowledge base...")
_knowledge_chunks: list[str] = get_all_knowledge()
=======

# How many knowledge chunks to retrieve
TOP_K_CHUNKS = 10

# How many previous messages to send (chat memory)
HISTORY_WINDOW = 8

# ==============================
# API KEY ROTATION LOGIC
# ==============================

_current_key_index = 0  # Tracks current API key

# Get API URL using current key
def _get_active_url():
    key = GEMINI_API_KEYS[_current_key_index]
    return GEMINI_BASE_URL.format(api_key=key)

# Switch to next API key if current one fails
def _rotate_key():
    global _current_key_index
    next_index = _current_key_index + 1

    if next_index < len(GEMINI_API_KEYS):
        _current_key_index = next_index
        print("🔑 Switched to next API key")
        return True

    print("❌ No more API keys available")
    return False

# ==============================
# SBERT MODEL LOAD (RUNS ONCE)
# ==============================

print("🔄 Loading SBERT model...")
_sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
# Lightweight embedding model (fast)

print("🔄 Encoding knowledge base...")
_knowledge_chunks = get_all_knowledge()
# Load all knowledge (text chunks)

# Convert text → embeddings (vector form)
>>>>>>> Stashed changes
_knowledge_embeddings = _sbert_model.encode(
    _knowledge_chunks,
    convert_to_tensor=True,
    show_progress_bar=False,
)
<<<<<<< Updated upstream
print(f" SBERT ready — {len(_knowledge_chunks)} chunks indexed")
=======

print(f"✅ SBERT ready — {len(_knowledge_chunks)} chunks indexed")
>>>>>>> Stashed changes

# ==============================
# RETRIEVAL FUNCTION (RAG)
# ==============================

def retrieve_relevant_chunks(query, top_k=TOP_K_CHUNKS):
    """
    Find most relevant knowledge chunks using cosine similarity
    """

    # Convert query → embedding
    query_embedding = _sbert_model.encode(query, convert_to_tensor=True)

    # Compare with all knowledge embeddings
    scores = util.cos_sim(query_embedding, _knowledge_embeddings)[0]

    # Get top K highest scores
    top_indices = scores.argsort(descending=True)[:top_k].tolist()

    # Return corresponding text chunks
    return [_knowledge_chunks[i] for i in top_indices]

# ==============================
# BUILD SCAN CONTEXT TEXT
# ==============================

<<<<<<< Updated upstream
def _build_analysis_block(context: dict) -> str:
    """Turn the scan context dict into a readable text block, including full history."""

    # ── Current scan ──────────────────────────────────────────────────────────
    if not context or not context.get("tumor_type"):
        current_block = "No scan has been analyzed yet in this session."
    else:
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
        current_block = (
            f"CURRENT SCAN RESULTS:\n"
            f"- Detected: {tumor_type}\n"
            f"- Confidence: {highest_confidence}\n"
            f"- All class scores: {scores_text}\n"
            f"- Grad-CAM: {gradcam_text}"
        )

    # ── Scan history ──────────────────────────────────────────────────────────
    # scan_history is a list of dicts, each with keys:
    #   tumor_type, highest_confidence, confidence (dict), timestamp, has_gradcam
    history_list = context.get("scan_history", [])
    if history_list:
        history_lines = ["PREVIOUS SCAN HISTORY:"]
        for i, scan in enumerate(history_list, 1):
            t          = scan.get("tumor_type", "Unknown")
            conf       = scan.get("highest_confidence", "N/A")
            ts         = scan.get("timestamp", "Unknown time")
            all_scores = scan.get("confidence", {})
            scores_str = (
                ", ".join(f"{k}: {v}" for k, v in all_scores.items())
                if all_scores else "N/A"
            )
            history_lines.append(
                f"  Scan {i} [{ts}]: {t} — {conf} (all scores: {scores_str})"
            )
        history_block = "\n".join(history_lines)
    else:
        history_block = "No previous scans in history."

    return f"{current_block}\n\n{history_block}"
=======
def _build_analysis_block(context):
    """
    Converts scan results + history into readable text
    """

    # ----- CURRENT SCAN -----
    if not context or not context.get("tumor_type"):
        current_block = "No scan has been analyzed yet."
    else:
        tumor_type = context.get("tumor_type")
        highest_confidence = context.get("highest_confidence")
        conf_dict = context.get("confidence", {})
        has_gradcam = context.get("has_gradcam", False)

        # Convert confidence dict to readable string
        scores_text = ", ".join(f"{k}: {v}" for k, v in conf_dict.items())

        # Grad-CAM explanation
        gradcam_text = (
            "Heatmap shows tumor focus areas"
            if has_gradcam else
            "No tumor detected"
        )

        current_block = (
            f"CURRENT SCAN:\n"
            f"- Tumor: {tumor_type}\n"
            f"- Confidence: {highest_confidence}\n"
            f"- Scores: {scores_text}\n"
            f"- Grad-CAM: {gradcam_text}"
        )
>>>>>>> Stashed changes

    # ----- SCAN HISTORY -----
    history_list = context.get("scan_history", [])

    if history_list:
        history_lines = ["PREVIOUS SCANS:"]
        for i, scan in enumerate(history_list, 1):
            history_lines.append(
                f"Scan {i}: {scan.get('tumor_type')} ({scan.get('highest_confidence')})"
            )
        history_block = "\n".join(history_lines)
    else:
        history_block = "No previous scans"

    return current_block + "\n\n" + history_block

# ==============================
# BUILD SYSTEM PROMPT
# ==============================

def _build_system_prompt(analysis_block, retrieved_chunks):
    """
    Combines:
    - Scan data
    - Retrieved knowledge
    - Instructions for Gemini
    """

    # Join knowledge chunks
    rag_context = "\n".join(f"• {chunk}" for chunk in retrieved_chunks)

    return f"""
You are NeuroVision AI Assistant.

SCAN INFO:
{analysis_block}

KNOWLEDGE:
{rag_context}

INSTRUCTIONS:
<<<<<<< Updated upstream
- Use the retrieved knowledge above to give accurate, grounded answers
- Explain scan results clearly in simple language
- Answer questions about Grad-CAM, tumor types, symptoms, and treatments
- Be warm and empathetic — users may be patients or worried family members
- Keep answers concise (2-4 sentences) unless more detail is needed
- ALWAYS remind users to consult a qualified doctor — this tool is NOT a medical diagnosis
- Do not fabricate medical information outside the knowledge provided
- If user asks about accuracy, dataset, kaggle, model performance or results, answer using the retrieved knowledge chunks above

COMPARISON INSTRUCTIONS:
- If the user asks to compare scans (e.g. 'compare with previous', 'how did it change', 'compare results'),
  use the PREVIOUS SCAN HISTORY block above to compare tumor type and confidence scores across ALL scans
- Compare across ALL tumor types — Glioma, Meningioma, Pituitary, and No Tumor
- Highlight any changes in tumor type between scans (e.g. Glioma → Meningioma → Pituitary)
- Point out significant differences in confidence scores across scans (e.g. 87% → 94% → 99%)
- If tumor type is the same across scans, mention whether confidence is increasing or decreasing
- If only one scan exists and no history, say there is not enough history to compare yet
- Always summarize the comparison clearly with scan timestamps if available"""
=======
- Explain results simply
- Be helpful and empathetic
- Use knowledge above
- Always suggest consulting a doctor
"""
>>>>>>> Stashed changes

# ==============================
# MAIN CHAT FUNCTION
# ==============================

def get_chat_reply(user_message, context, history):
    """
<<<<<<< Updated upstream
    Full RAG pipeline:
      1. Retrieve top-K relevant chunks via SBERT
      2. Build system prompt with retrieved context + scan results + scan history
      3. Send to Gemini and return the reply

    context dict should contain:
      - tumor_type (str)             e.g. "Glioma"
      - highest_confidence (str)     e.g. "99.98%"
      - confidence (dict)            e.g. {"Glioma": "99.98%", "Meningioma": "0.01%", ...}
      - has_gradcam (bool)
      - scan_history (list of dicts) each dict has same keys above + "timestamp" (str)

    Example scan_history entry:
      {
        "tumor_type": "Pituitary",
        "highest_confidence": "99.38%",
        "confidence": {"Glioma": "0.01%", "Meningioma": "0.01%", "Pituitary": "99.38%", "No Tumor": "0.60%"},
        "has_gradcam": True,
        "timestamp": "12:02:31 PM"
      }

    Returns an error string (never raises) for API failures.
=======
    Full pipeline:
    1. Retrieve knowledge
    2. Build prompt
    3. Send to Gemini
    4. Return response
>>>>>>> Stashed changes
    """

    if not user_message:
        raise ValueError("Empty message")

    # ----- STEP 1: RETRIEVE -----
    retrieved_chunks = retrieve_relevant_chunks(user_message)

    # ----- STEP 2: BUILD PROMPT -----
    analysis_block = _build_analysis_block(context)
    system_prompt = _build_system_prompt(analysis_block, retrieved_chunks)

    # ----- STEP 3: CHAT HISTORY -----
    gemini_contents = []

    # Add past messages (limited window)
    for msg in history[-HISTORY_WINDOW:]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })

    # Add current message
    gemini_contents.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    # Request payload
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": gemini_contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 512
        },
    }

<<<<<<< Updated upstream
    # Step 4 — Call Gemini with automatic key rotation on rate limit
=======
    # ----- STEP 4: CALL GEMINI -----
>>>>>>> Stashed changes
    for attempt in range(len(GEMINI_API_KEYS)):
        try:
            response = requests.post(
                _get_active_url(),
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
<<<<<<< Updated upstream
            response.raise_for_status()
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]

        except requests.exceptions.Timeout:
            return "Sorry, the response timed out. Please try again."

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            print(f" Gemini HTTP error {status} on key {_current_key_index + 1}: {e.response.text}")

            if status in (429, 503):
                # Rate limit or quota exceeded — try next key
                print(f" Key {_current_key_index + 1} limit reached. Trying next key...")
                if not _rotate_key():
                    return " All API keys have reached their limit. Please add a new key or wait until the quota resets."
                continue  # retry with new key

            if status == 400:
                return " Invalid API key. Please check your GEMINI_API_KEYS list."

            return f" API error ({status}). Please try again in a moment."

        except Exception as e:
            print(f" Gemini error: {e}")
            return "Sorry, something went wrong. Please try again."

    return " All API keys have reached their limit. Please add a new key or wait until the quota resets."
=======

            response.raise_for_status()

            result = response.json()

            # Extract reply text
            return result["candidates"][0]["content"]["parts"][0]["text"]

        except requests.exceptions.Timeout:
            return "Request timed out. Try again."

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code

            # If rate limit → switch key
            if status in (429, 503):
                if not _rotate_key():
                    return "All API keys exhausted."
                continue

            if status == 400:
                return "Invalid API key."

            return "API error occurred."

        except Exception:
            return "Something went wrong."

    return "All API keys exhausted."
>>>>>>> Stashed changes
