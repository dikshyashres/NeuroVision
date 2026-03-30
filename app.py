from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from tensorflow.keras.models import model_from_json
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import json
import re
import hashlib

# ── Local modules ────────────────────────────────────────────────
from gradcam import generate_gradcam
from chatbot import get_chat_reply

app = Flask(__name__)
app.secret_key = "neurovision_secret_key_2024_prod_12345"

# -----------------------------
# Configuration
# -----------------------------
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.gif']
app.config['SESSION_PERMANENT'] = False

# -----------------------------
# Simple User Database
# -----------------------------
USERS_FILE = 'users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    return True, ""

# -----------------------------
# Load the model
# -----------------------------
MODEL_JSON    = os.path.join("model", "vgg16_model.json")
MODEL_WEIGHTS = os.path.join("model", "vgg16_weights.weights.h5")

try:
    if os.path.exists(MODEL_JSON):
        with open(MODEL_JSON, "r") as f:
            loaded_model = model_from_json(f.read())

        if os.path.exists(MODEL_WEIGHTS):
            loaded_model.load_weights(MODEL_WEIGHTS)
            loaded_model.trainable = False
            print("✅ Model loaded successfully")
        else:
            print(f"❌ Weights file not found: {MODEL_WEIGHTS}")
            loaded_model = None
    else:
        print(f"❌ Model JSON not found: {MODEL_JSON}")
        loaded_model = None
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    loaded_model = None

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE = 224

def decode_base64_image(b64str):
    try:
        if ',' in b64str:
            b64str = b64str.split(',')[1]
        img_bytes = base64.b64decode(b64str)
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))
        return np.array(pil_img) / 255.0
    except Exception as e:
        print(f"Error decoding image: {str(e)}")
        raise

# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if 'user' in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        if 'email' in request.form:  # Registration form
            name             = request.form.get("fullname", "").strip()
            email            = request.form.get("email", "").strip().lower()
            password         = request.form.get("password", "").strip()
            confirm_password = request.form.get("confirm_password", "").strip()
            terms            = request.form.get("terms")

            if not all([name, email, password, confirm_password]):
                return render_template("index.html", register_error="All fields are required")
            if not validate_email(email):
                return render_template("index.html", register_error="Please enter a valid email address")
            is_valid, msg = validate_password(password)
            if not is_valid:
                return render_template("index.html", register_error=msg)
            if password != confirm_password:
                return render_template("index.html", register_error="Passwords do not match")
            if not terms:
                return render_template("index.html", register_error="You must agree to the Terms & Conditions")

            users = load_users()
            if email in users:
                return render_template("index.html", register_error="Email already registered")

            users[email] = {
                'name': name,
                'email': email,
                'password_hash': hash_password(password),
                'created_at': str(np.datetime64('now')),
            }
            save_users(users)
            print(f"✅ New user registered: {email}")
            return render_template("index.html", success_message="Registration successful! Please login with your credentials.")

        else:  # Login form
            email    = request.form.get("username", "").strip().lower()
            password = request.form.get("password", "").strip()

            if not email or not password:
                return render_template("index.html", error="Please enter both email and password")
            if not validate_email(email):
                return render_template("index.html", error="Please enter a valid email address")

            users = load_users()
            if email in users and hashlib.sha256(password.encode()).hexdigest() == users[email]['password_hash']:
                session['user'] = email
                session['name'] = users[email]['name']
                print(f"✅ User '{email}' logged in")
                return redirect(url_for("home"))

            error_msg = "Account not found. Please register first." if email not in users else "Invalid email or password"
            return render_template("index.html", error=error_msg)

    return render_template("index.html")

@app.route("/home")
def home():
    if 'user' not in session:
        return redirect(url_for("login"))
    return render_template("home.html")

@app.route("/detection")
def detection():
    if 'user' not in session:
        return redirect(url_for("login"))
    return render_template("detection.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'user' not in session:
        return jsonify({"error": "Not authenticated. Please login first."}), 401
    if loaded_model is None:
        return jsonify({"error": "Model not loaded. Please contact administrator."}), 500

    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        images = data["image"]
        if not images:
            return jsonify({"error": "Empty image list"}), 400

        processed_imgs = []
        for img_str in images[:5]:
            try:
                processed_imgs.append(decode_base64_image(img_str))
            except Exception as img_error:
                return jsonify({"error": f"Invalid image: {str(img_error)}"}), 400

        if not processed_imgs:
            return jsonify({"error": "No valid images to process"}), 400

        batch        = np.array(processed_imgs)
        predictions  = loaded_model.predict(batch)
        class_indices = np.argmax(predictions, axis=1)

        results = []
        for idx, (img_array, pred) in enumerate(zip(processed_imgs, predictions)):
            class_idx  = class_indices[idx]
            tumor_type = CLASS_NAMES[class_idx]

            print(f"\nImage {idx+1}: {tumor_type} ({pred[class_idx]*100:.2f}%)")

            # Grad-CAM only when a tumor is detected
            gradcam_data = None
            if tumor_type != 'No Tumor':
                print(f"🔥 Generating Grad-CAM for {tumor_type}...")
                try:
                    gradcam_data = generate_gradcam(loaded_model, img_array, class_idx)
                except Exception as grad_error:
                    print(f"❌ Grad-CAM error: {grad_error}")

            results.append({
                "tumor_type":        tumor_type,
                "confidence":        {cls: f"{conf*100:.2f}%" for cls, conf in zip(CLASS_NAMES, pred)},
                "highest_confidence": f"{pred[class_idx]*100:.2f}%",
                "gradcam":           gradcam_data,
            })

        return jsonify({"success": True, "predictions": results, "count": len(results)})

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    if 'user' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    data         = request.get_json()
    user_message = data.get("message", "").strip()
    context      = data.get("context", {})
    history      = data.get("history", [])

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    reply = get_chat_reply(user_message, context, history)
    return jsonify({"reply": reply})

@app.route("/logout")
def logout():
    if 'user' in session:
        email = session['user']
        session.clear()
        print(f"👋 User '{email}' logged out")
    return redirect(url_for("login"))

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory('static', filename)

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    for d in ['static', 'templates', 'model']:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(USERS_FILE):
        save_users({})
        print(f"📁 Created users database: {USERS_FILE}")

    print("\n" + "="*50)
    print("🧠 NeuroVision AI Diagnostic System")
    print("="*50)
    print(f"{'✅' if loaded_model else '⚠️ '} Model: {'VGG16 loaded' if loaded_model else 'NOT loaded!'}")
    print("✅ Gemini chatbot ready")
    print("🌐 http://localhost:5000")
    print("="*50 + "\n")

    app.run(debug=True, port=5000, host='0.0.0.0')