"""
chatbot.py — Gemini chatbot logic for NeuroVision
"""

import requests

GEMINI_API_KEY = "enter your api key"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
)

SYSTEM_PROMPT_TEMPLATE = """You are NeuroVision Assistant — a helpful, empathetic AI inside a brain MRI tumor detection system.

The VGG16 model classifies MRI scans into:
- Glioma: Tumor from glial cells; can be low-grade (slow) or high-grade (aggressive)
- Meningioma: Tumor from brain/spinal cord lining; usually benign and slow-growing
- Pituitary: Tumor in pituitary gland; usually benign, affects hormone production
- No Tumor: No tumor found

Grad-CAM = heatmap showing which MRI regions the AI focused on. Red/yellow = high activation (likely tumor). Blue = low activation.

{analysis_block}

INSTRUCTIONS:
- Explain scan results clearly in simple language
- Answer questions about Grad-CAM, tumor types, symptoms, and treatments
- Be warm and empathetic — users may be patients or worried family members
- Keep answers concise (2-4 sentences) unless more detail is needed
- ALWAYS remind users to consult a qualified doctor — this tool is NOT a medical diagnosis"""


def _build_analysis_block(context: dict) -> str:
    """Turn the scan context dict into a readable text block for the system prompt."""
    if not context or not context.get("tumor_type"):
        return "No scan has been analyzed yet in this session."

    tumor_type = context.get("tumor_type", "Unknown")
    highest_confidence = context.get("highest_confidence", "N/A")
    conf_dict = context.get("confidence", {})
    has_gradcam = context.get("has_gradcam", False)

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


def get_chat_reply(user_message: str, context: dict, history: list) -> str:
    """
    Send a message to Gemini and return the reply string.
    Raises ValueError for empty messages.
    Returns an error string (never raises) for API failures.
    """
    if not user_message:
        raise ValueError("Empty message")

    analysis_block = _build_analysis_block(context)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(analysis_block=analysis_block)

    # Build conversation history (last 8 turns)
    gemini_contents = []
    for msg in history[-8:]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    gemini_contents.append({"role": "user", "parts": [{"text": user_message}]})

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": gemini_contents,
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 512},
    }

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
            return "⏳ Too many requests — the free API limit was hit. Please wait 30 seconds and try again."
        if status == 400:
            return "⚠️ Invalid API key. Please check your GEMINI_API_KEY."
        return f"⚠️ API error ({status}). Please try again in a moment."
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return "Sorry, something went wrong. Please try again."