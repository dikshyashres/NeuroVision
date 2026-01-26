from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from tensorflow.keras.models import model_from_json, Model
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import json
import re
import hashlib
import cv2

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
MODEL_JSON = os.path.join("model", "vgg16_model.json")
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
        img_array = np.array(pil_img) / 255.0
        
        return img_array
    except Exception as e:
        print(f"Error decoding image: {str(e)}")
        raise

# -----------------------------
# Grad-CAM Function
# -----------------------------
def generate_gradcam(model, img_array, pred_index=None):
    """
    Generate Grad-CAM heatmap for the given image
    """
    try:
        print(f"🔥 Starting Grad-CAM generation...")
        print(f"   Model: {type(model)}")
        print(f"   Image shape: {img_array.shape}")
        print(f"   Pred index: {pred_index}")
        
        # Find the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            print("❌ No convolutional layer found")
            return None
        
        print(f"✅ Using layer: {last_conv_layer.name}")
        
        # Create a model that outputs both predictions and conv layer output
        grad_model = Model(
            inputs=model.input,
            outputs=[model.output, last_conv_layer.output]
        )
        
        # Expand dimensions for batch processing
        img_tensor = np.expand_dims(img_array, axis=0)
        print(f"   Image tensor shape: {img_tensor.shape}")
        
        # Get gradients
        with tf.GradientTape() as tape:
            preds, conv_outputs = grad_model(img_tensor)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        print(f"   Conv outputs shape: {conv_outputs.shape}")
        
        # Get gradients of the predicted class with respect to conv layer
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            print("❌ Gradients are None")
            return None
        
        print(f"   Gradients shape: {grads.shape}")
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for i in range(pooled_grads.shape[0]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Average the weighted feature maps
        heatmap = np.mean(conv_outputs, axis=-1)
        
        print(f"   Heatmap shape before resize: {heatmap.shape}")
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        
        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        
        # Apply colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert original image back to uint8
        original_img = np.uint8(img_array * 255)
        
        # Superimpose heatmap on original image
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        # Convert to base64 for sending to frontend
        _, buffer = cv2.imencode('.png', cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
        gradcam_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print(f"✅ Grad-CAM base64 length: {len(gradcam_base64)} characters")
        
        return f"data:image/png;base64,{gradcam_base64}"
        
    except Exception as e:
        print(f"❌ Grad-CAM error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle BOTH login and registration"""
    if 'user' in session:
        return redirect(url_for("home"))
    
    if request.method == "POST":
        # Check which form was submitted based on field names
        if 'email' in request.form:  # This is a REGISTER form
            # Get registration data
            name = request.form.get("fullname", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "").strip()
            confirm_password = request.form.get("confirm_password", "").strip()
            terms = request.form.get("terms")
            
            # Validation
            if not all([name, email, password, confirm_password]):
                return render_template("login.html", 
                    register_error="All fields are required")
            
            if not validate_email(email):
                return render_template("login.html", 
                    register_error="Please enter a valid email address")
            
            is_valid, msg = validate_password(password)
            if not is_valid:
                return render_template("login.html", 
                    register_error=msg)
            
            if password != confirm_password:
                return render_template("login.html", 
                    register_error="Passwords do not match")
            
            if not terms:
                return render_template("login.html", 
                    register_error="You must agree to the Terms & Conditions")
            
            # Check if user exists
            users = load_users()
            
            if email in users:
                return render_template("login.html", 
                    register_error="Email already registered")
            
            # Create new user
            users[email] = {
                'name': name,
                'email': email,
                'password_hash': hash_password(password),
                'created_at': str(np.datetime64('now'))
            }
            
            save_users(users)
            
            print(f"✅ New user registered: {email}")
            
            return render_template("login.html", 
                success_message="Registration successful! Please login with your credentials.")
        
        else:  # This is a LOGIN form
            email = request.form.get("username", "").strip().lower()
            password = request.form.get("password", "").strip()
            
            if not email or not password:
                return render_template("login.html", 
                    error="Please enter both email and password")
            
            if not validate_email(email):
                return render_template("login.html", 
                    error="Please enter a valid email address")
            
            users = load_users()
            
            if email in users:
                stored_hash = users[email]['password_hash']
                input_hash = hash_password(password)
                
                if stored_hash == input_hash:
                    session['user'] = email
                    session['name'] = users[email]['name']
                    print(f"✅ User '{email}' logged in successfully")
                    return redirect(url_for("home"))
                else:
                    return render_template("login.html", 
                        error="Invalid email or password")
            else:
                return render_template("login.html", 
                    error="Account not found. Please register first.")
    
    return render_template("login.html")

@app.route("/home")
def home():
    if 'user' not in session:
        print("⚠️ Unauthorized access to home page")
        return redirect(url_for("login"))
    
    return render_template("home.html")

@app.route("/detection")
def detection():
    if 'user' not in session:
        print("⚠️ Unauthorized access to detection page")
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
                img_array = decode_base64_image(img_str)
                processed_imgs.append(img_array)
            except Exception as img_error:
                print(f"❌ Error processing image: {str(img_error)}")
                return jsonify({"error": f"Invalid image: {str(img_error)}"}), 400
        
        if not processed_imgs:
            return jsonify({"error": "No valid images to process"}), 400
        
        processed_imgs_batch = np.array(processed_imgs)
        predictions = loaded_model.predict(processed_imgs_batch)
        class_indices = np.argmax(predictions, axis=1)
        
        results = []
        for idx, (img_array, pred) in enumerate(zip(processed_imgs, predictions)):
            class_idx = class_indices[idx]
            
            # Generate Grad-CAM only if tumor is detected
            gradcam_image = None
            if CLASS_NAMES[class_idx] != 'No Tumor':
                print(f"🔥 Generating Grad-CAM for {CLASS_NAMES[class_idx]}...")
                gradcam_image = generate_gradcam(loaded_model, img_array, class_idx)
                if gradcam_image:
                    print(f"✅ Grad-CAM generated successfully")
                else:
                    print(f"❌ Grad-CAM generation failed")
            else:
                print(f"ℹ️ No tumor detected, skipping Grad-CAM")
            
            confidence_dict = {}
            for cls, conf in zip(CLASS_NAMES, pred):
                confidence_dict[cls] = f"{conf*100:.2f}%"
            
            results.append({
                "tumor_type": CLASS_NAMES[class_idx],
                "confidence": confidence_dict,
                "highest_confidence": f"{pred[class_idx]*100:.2f}%",
                "gradcam": gradcam_image
            })
        
        return jsonify({
            "success": True,
            "predictions": results,
            "count": len(results)
        })
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

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
# Run the app
# -----------------------------
if __name__ == "__main__":
    required_dirs = ['static', 'templates', 'model']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"📁 Created directory: {dir_name}")
    
    if not os.path.exists(USERS_FILE):
        save_users({})
        print(f"📁 Created users database: {USERS_FILE}")
    
    print("\n" + "="*50)
    print("🧠 NeuroVision AI Diagnostic System with Grad-CAM")
    print("="*50)
    
    if loaded_model:
        print("✅ Model: VGG16 loaded successfully")
        print("✅ Grad-CAM: Enabled for tumor visualization")
    else:
        print("⚠️  Warning: Model not loaded!")
        print("⚠️  Prediction functionality will not work")
    
    print("\n📧 Registration System:")
    print("   • Users must register with email")
    print("   • Registration redirects to login page")
    print("\n🌐 Starting server at http://localhost:5000")
    print("="*50 + "\n")
    
    try:
        app.run(debug=True, port=5000, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")