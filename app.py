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
# Enhanced Grad-CAM Function
# -----------------------------
def generate_gradcam(model, img_array, pred_index=None):
    """
    Generate Grad-CAM heatmap for the given image
    Returns: Dictionary with overlay and pure heatmap
    """
    try:
        print(f"\n{'='*60}")
        print(f"🔥 Starting Grad-CAM generation...")
        print(f"   Model type: {type(model)}")
        print(f"   Model layers count: {len(model.layers)}")
        print(f"   Image shape: {img_array.shape}")
        print(f"   Pred index: {pred_index}")
        
        # Debug: Print all layer names
        print(f"\n   Available layers:")
        for i, layer in enumerate(model.layers):
            print(f"   [{i}] {layer.name} - {type(layer).__name__}")
        
        # Find the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            layer_type = type(layer).__name__
            if 'conv' in layer.name.lower() or 'Conv' in layer_type:
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            print("\n❌ No convolutional layer found in model!")
            print("   Tried to find layers with 'conv' in name or Conv2D type")
            return None
        
        print(f"\n✅ Using conv layer: {last_conv_layer.name}")
        try:
            print(f"   Layer output shape: {last_conv_layer.output.shape}")
        except:
            print(f"   Layer output shape: (unable to determine)")
        
        # Create a model that outputs both predictions and conv layer output
        try:
            grad_model = Model(
                inputs=model.input,
                outputs=[model.output, last_conv_layer.output]
            )
            print(f"✅ Created gradient model successfully")
        except Exception as e:
            print(f"❌ Failed to create gradient model: {str(e)}")
            return None
        
        # Expand dimensions for batch processing
        img_tensor = np.expand_dims(img_array, axis=0)
        img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
        print(f"   Image tensor shape: {img_tensor.shape}")
        print(f"   Image tensor dtype: {img_tensor.dtype}")
        
        # Get gradients
        try:
            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                model_outputs = grad_model(img_tensor, training=False)
                
                print(f"   Model output type: {type(model_outputs)}")
                
                # Handle both list and tensor outputs
                if isinstance(model_outputs, list):
                    print(f"   Output is a list with {len(model_outputs)} elements")
                    preds = model_outputs[0]
                    conv_outputs = model_outputs[1]
                else:
                    preds, conv_outputs = model_outputs
                
                # Convert to tensors if needed
                preds = tf.convert_to_tensor(preds)
                conv_outputs = tf.convert_to_tensor(conv_outputs)
                
                print(f"   Raw preds shape: {preds.shape}")
                print(f"   Raw conv outputs shape: {conv_outputs.shape}")
                
                # Fix preds shape - should be (batch_size, num_classes)
                # Remove any extra dimensions
                while len(preds.shape) > 2:
                    print(f"   ⚠️  Predictions has extra dimension, squeezing")
                    preds = tf.squeeze(preds, axis=1)
                    print(f"   After squeeze: {preds.shape}")
                
                # Add batch dimension if completely missing
                if len(preds.shape) == 1:
                    print(f"   ⚠️  Predictions is 1D, adding batch dimension")
                    preds = tf.expand_dims(preds, axis=0)
                    print(f"   After expand: {preds.shape}")
                
                # Fix conv_outputs shape - should be (batch_size, height, width, channels)
                if len(conv_outputs.shape) == 3:
                    print(f"   ⚠️  Conv outputs is 3D, adding batch dimension")
                    conv_outputs = tf.expand_dims(conv_outputs, axis=0)
                    print(f"   Reshaped conv outputs: {conv_outputs.shape}")
                
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                pred_index = int(pred_index)
                
                print(f"   Final preds shape: {preds.shape}")
                print(f"   Final conv outputs shape: {conv_outputs.shape}")
                print(f"   Trying to access class index: {pred_index}")
                print(f"   Max index possible: {preds.shape[-1] - 1}")
                
                class_channel = preds[:, pred_index]
            
            print(f"   Predictions shape: {preds.shape}")
            print(f"   Conv outputs shape: {conv_outputs.shape}")
            print(f"   Selected class index: {pred_index}")
            print(f"   Class channel value: {class_channel.numpy()}")
        except Exception as e:
            print(f"❌ Error during forward pass: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        # Get gradients of the predicted class with respect to conv layer
        try:
            grads = tape.gradient(class_channel, conv_outputs)
            
            if grads is None:
                print("❌ Gradients are None - this shouldn't happen!")
                return None
            
            print(f"✅ Gradients shape: {grads.shape}")
        except Exception as e:
            print(f"❌ Error computing gradients: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        print(f"   Pooled gradients shape: {pooled_grads.shape}")
        
        # Weight the channels by the gradients
        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()
        
        for i in range(pooled_grads.shape[0]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Average the weighted feature maps
        heatmap = np.mean(conv_outputs, axis=-1)
        print(f"   Heatmap shape: {heatmap.shape}")
        print(f"   Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        else:
            print("⚠️  Warning: Heatmap max is 0, no activation detected")
        
        print(f"   Normalized heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
        
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        
        # Apply colormap for heatmap visualization
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_jet = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        
        # Convert original image back to uint8
        original_img = np.uint8(img_array * 255)
        
        # Create overlay image (blended)
        superimposed_img = cv2.addWeighted(original_img, 0.5, heatmap_jet, 0.5, 0)
        
        # Convert images to base64
        try:
            # 1. Overlay (heatmap on original)
            _, buffer1 = cv2.imencode('.png', cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
            overlay_base64 = base64.b64encode(buffer1).decode('utf-8')
            
            # 2. Pure heatmap visualization
            _, buffer2 = cv2.imencode('.png', heatmap_jet)
            heatmap_base64 = base64.b64encode(buffer2).decode('utf-8')
            
            print(f"✅ Overlay base64 length: {len(overlay_base64)}")
            print(f"✅ Heatmap base64 length: {len(heatmap_base64)}")
        except Exception as e:
            print(f"❌ Error encoding images: {str(e)}")
            return None
        
        result = {
            "overlay": f"data:image/png;base64,{overlay_base64}",
            "heatmap": f"data:image/png;base64,{heatmap_base64}"
        }
        
        print(f"✅ Grad-CAM generated successfully!")
        print(f"{'='*60}\n")
        
        return result
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ GRAD-CAM FATAL ERROR: {str(e)}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
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
            tumor_type = CLASS_NAMES[class_idx]
            
            print(f"\n{'='*60}")
            print(f"Processing image {idx + 1}/{len(processed_imgs)}")
            print(f"Predicted tumor type: {tumor_type}")
            print(f"Confidence: {pred[class_idx]*100:.2f}%")
            print(f"{'='*60}")
            
            # Generate Grad-CAM only if tumor is detected
            gradcam_data = None
            if tumor_type != 'No Tumor':
                print(f"🔥 Tumor detected! Generating Grad-CAM for {tumor_type}...")
                try:
                    gradcam_data = generate_gradcam(loaded_model, img_array, class_idx)
                    if gradcam_data:
                        print(f"✅ Grad-CAM generated successfully!")
                        print(f"   Has overlay: {bool(gradcam_data.get('overlay'))}")
                        print(f"   Has heatmap: {bool(gradcam_data.get('heatmap'))}")
                    else:
                        print(f"❌ Grad-CAM generation returned None!")
                except Exception as grad_error:
                    print(f"❌ Exception during Grad-CAM generation: {str(grad_error)}")
                    import traceback
                    traceback.print_exc()
                    gradcam_data = None
            else:
                print(f"ℹ️  No tumor detected, skipping Grad-CAM generation")
            
            confidence_dict = {}
            for cls, conf in zip(CLASS_NAMES, pred):
                confidence_dict[cls] = f"{conf*100:.2f}%"
            
            result = {
                "tumor_type": tumor_type,
                "confidence": confidence_dict,
                "highest_confidence": f"{pred[class_idx]*100:.2f}%",
                "gradcam": gradcam_data
            }
            
            print(f"\nResult for image {idx + 1}:")
            print(f"  - Tumor: {result['tumor_type']}")
            print(f"  - Confidence: {result['highest_confidence']}")
            print(f"  - Grad-CAM: {bool(result['gradcam'])}")
            
            results.append(result)
        
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
    print("🧠 NeuroVision AI Diagnostic System with Enhanced Grad-CAM")
    print("="*50)
    
    if loaded_model:
        print("✅ Model: VGG16 loaded successfully")
        print("✅ Grad-CAM: Enhanced 3-panel visualization enabled")
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