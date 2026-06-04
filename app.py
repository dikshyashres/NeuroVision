<<<<<<< Updated upstream
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory  
=======
# ==============================
# Import required libraries
# ==============================
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
# Flask → backend web framework

>>>>>>> Stashed changes
from tensorflow.keras.models import model_from_json
# Used to load model architecture from JSON

import numpy as np
import base64
from io import BytesIO
from PIL import Image
# Image processing utilities

import os
import json
import re
import hashlib
# os → file handling
# json → store user database
# re → regex validation
# hashlib → password hashing

# ==============================
# Import custom modules
# ==============================
from gradcam import generate_gradcam   # For model explainability (heatmaps)
from chatbot import get_chat_reply     # AI chatbot module

# ==============================
# Initialize Flask app
# ==============================
app = Flask(__name__)
<<<<<<< Updated upstream
app.secret_key = "neurovision_secret_key_2024_prod_12345" #session login security

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
=======

# Secret key used for session security (login sessions)
app.secret_key = "neurovision_secret_key_2024_prod_12345"

# ==============================
# Configuration settings
# ==============================
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size = 16MB
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.gif']  # Allowed formats
app.config['SESSION_PERMANENT'] = False  # Session expires when browser closes

# ==============================
# Simple User Database (JSON-based)
# ==============================
USERS_FILE = 'users.json'

# Load users from file
def load_users():
    if os.path.exists(USERS_FILE):
>>>>>>> Stashed changes
        with open(USERS_FILE, 'r') as f:
            return json.load(f) 
    return {}

<<<<<<< Updated upstream
def save_users(users): 
=======
# Save users to file
def save_users(users):
>>>>>>> Stashed changes
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2) 

# Hash password using SHA256 (security)
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest() 

# Validate email format using regex
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None 

# Validate password strength
def validate_password(password):
    if len(password) < 6: 
        return False, "Password must be at least 6 characters"
    return True, ""

# ==============================
# Load Deep Learning Model
# ==============================
MODEL_JSON    = os.path.join("model", "vgg16_model.json")
MODEL_WEIGHTS = os.path.join("model", "vgg16_weights.weights.h5")

try:
    # Load model architecture
    if os.path.exists(MODEL_JSON):
        with open(MODEL_JSON, "r") as f:
            loaded_model = model_from_json(f.read())

        # Load weights
        if os.path.exists(MODEL_WEIGHTS):
            loaded_model.load_weights(MODEL_WEIGHTS)
<<<<<<< Updated upstream
            loaded_model.trainable = False
            print(" Model loaded successfully")
        else:
            print(f" Weights file not found: {MODEL_WEIGHTS}")
            loaded_model = None
    else:
        print(f" Model JSON not found: {MODEL_JSON}")
=======
            loaded_model.trainable = False  # Freeze model
            print("✅ Model loaded successfully")
        else:
            print(f"❌ Weights file not found")
            loaded_model = None
    else:
        print(f"❌ Model JSON not found")
>>>>>>> Stashed changes
        loaded_model = None

except Exception as e:
    print(f" Error loading model: {str(e)}")
    loaded_model = None

# Class labels for prediction
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

IMG_SIZE = 224  # Input size for model (VGG16 standard)

# ==============================
# Convert Base64 → Image → Array
# ==============================
def decode_base64_image(b64str):
    try:
        # Remove metadata prefix if present
        if ',' in b64str:
            b64str = b64str.split(',')[1]

        # Decode base64 string
        img_bytes = base64.b64decode(b64str)

        # Convert to PIL image
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")

        # Resize to model input size
        pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))

        # Normalize pixel values (0–1)
        return np.array(pil_img) / 255.0

    except Exception as e:
        print(f"Error decoding image: {str(e)}")
        raise

# ==============================
# ROUTES (Endpoints)
# ==============================

# Default route → redirect to login
@app.route("/")
def index():
    return redirect(url_for("login"))

# -----------------------------
# LOGIN & REGISTRATION
# -----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():

    # If already logged in → go to home
    if 'user' in session:
        return redirect(url_for("home"))

    if request.method == "POST":

        # -------------------------
        # REGISTRATION FLOW
        # -------------------------
        if 'email' in request.form:

            name  = request.form.get("fullname", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "").strip()
            confirm_password = request.form.get("confirm_password", "").strip()
            terms = request.form.get("terms")

            # Validate inputs
            if not all([name, email, password, confirm_password]):
                return render_template("index.html", register_error="All fields are required")

            if not validate_email(email):
                return render_template("index.html", register_error="Invalid email")

            is_valid, msg = validate_password(password)
            if not is_valid:
                return render_template("index.html", register_error=msg)

            if password != confirm_password:
                return render_template("index.html", register_error="Passwords do not match")

            if not terms:
                return render_template("index.html", register_error="Accept terms")

            users = load_users()

            # Check duplicate user
            if email in users:
                return render_template("index.html", register_error="Email already registered")

            # Save new user
            users[email] = {
                'name': name,
                'email': email,
                'password_hash': hash_password(password),
                'created_at': str(np.datetime64('now')),
            }
<<<<<<< Updated upstream
            save_users(users)
            print(f" New user registered: {email}")
            return render_template("index.html", success_message="Registration successful! Please login with your credentials.")
=======
>>>>>>> Stashed changes

            save_users(users)
            return render_template("index.html", success_message="Registration successful!")

        # -------------------------
        # LOGIN FLOW
        # -------------------------
        else:
            email = request.form.get("username", "").strip().lower()
            password = request.form.get("password", "").strip()

            if not email or not password:
                return render_template("index.html", error="Enter credentials")

            users = load_users()

            # Check credentials
            if email in users and hash_password(password) == users[email]['password_hash']:
                session['user'] = email
                session['name'] = users[email]['name']
<<<<<<< Updated upstream
                print(f" User '{email}' logged in")
=======
>>>>>>> Stashed changes
                return redirect(url_for("home"))

            return render_template("index.html", error="Invalid login")

    return render_template("index.html")

# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/home")
def home():
    if 'user' not in session:
        return redirect(url_for("login"))
    return render_template("home.html")

# -----------------------------
# DETECTION PAGE
# -----------------------------
@app.route("/detection")
def detection():
    if 'user' not in session:
        return redirect(url_for("login"))
    return render_template("detection.html")

# -----------------------------
# PREDICTION API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # Authentication check
    if 'user' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    if loaded_model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        images = data["image"]

        processed_imgs = []

        # Process up to 5 images
        for img_str in images[:5]:
            processed_imgs.append(decode_base64_image(img_str))

        batch = np.array(processed_imgs)

        # Model prediction
        predictions = loaded_model.predict(batch)

        class_indices = np.argmax(predictions, axis=1)

        results = []

        for idx, (img_array, pred) in enumerate(zip(processed_imgs, predictions)):

            class_idx = class_indices[idx]
            tumor_type = CLASS_NAMES[class_idx]

            # Generate Grad-CAM only if tumor detected
            gradcam_data = None
            if tumor_type != 'No Tumor':
<<<<<<< Updated upstream
                print(f" Generating Grad-CAM for {tumor_type}...")
                try:
                    gradcam_data = generate_gradcam(loaded_model, img_array, class_idx)
                except Exception as grad_error:
                    print(f"Grad-CAM error: {grad_error}")
=======
                gradcam_data = generate_gradcam(loaded_model, img_array, class_idx)
>>>>>>> Stashed changes

            results.append({
                "tumor_type": tumor_type,
                "confidence": {cls: f"{conf*100:.2f}%" for cls, conf in zip(CLASS_NAMES, pred)},
                "highest_confidence": f"{pred[class_idx]*100:.2f}%",
                "gradcam": gradcam_data,
            })

        return jsonify({"success": True, "predictions": results})

    except Exception as e:
<<<<<<< Updated upstream
        print(f" Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
=======
        return jsonify({"error": str(e)}), 500
>>>>>>> Stashed changes

# -----------------------------
# CHATBOT API
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    if 'user' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.get_json()

    user_message = data.get("message", "")
    context = data.get("context", {})
    history = data.get("history", [])

    reply = get_chat_reply(user_message, context, history)

    return jsonify({"reply": reply})

# -----------------------------
# LOGOUT
# -----------------------------
@app.route("/logout")
def logout():
<<<<<<< Updated upstream
    if 'user' in session:
        email = session['user']
        session.clear()
        print(f" User '{email}' logged out")
=======
    session.clear()
>>>>>>> Stashed changes
    return redirect(url_for("login"))

# -----------------------------
# STATIC FILES
# -----------------------------
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory('static', filename)

# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":

    # Ensure required folders exist
    for d in ['static', 'templates', 'model']:
        os.makedirs(d, exist_ok=True)

    # Create empty user database if not exists
    if not os.path.exists(USERS_FILE):
        save_users({})
<<<<<<< Updated upstream
        print(f" Created users database: {USERS_FILE}")

    print("\n" + "="*50)
    print(" NeuroVision AI Diagnostic System")
    print("="*50)
    print(f"{'✅' if loaded_model else '⚠️ '} Model: {'VGG16 loaded' if loaded_model else 'NOT loaded!'}")
    print(" Gemini chatbot ready")
    print(" http://localhost:5000")
    print("="*50 + "\n")
=======

    print("🧠 NeuroVision System Running...")
    print("🌐 http://localhost:5000")
>>>>>>> Stashed changes

    # Start Flask server
    app.run(debug=True, port=5000, host='0.0.0.0')