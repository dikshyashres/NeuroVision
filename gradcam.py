"""
gradcam.py — Grad-CAM heatmap generation for NeuroVision
"""

import base64
import traceback

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

IMG_SIZE = 224


def generate_gradcam(model, img_array, pred_index=None):
    """
    Generate Grad-CAM heatmap for the given image.

    Args:
        model:      Loaded Keras model
        img_array:  Numpy array of shape (H, W, 3), values in [0, 1]
        pred_index: Class index to explain (defaults to argmax)

    Returns:
        dict with keys "overlay" and "heatmap" (base64 PNG data URIs),
        or None on failure.
    """
    try:
        print(f"\n{'='*60}")
        print("🔥 Starting Grad-CAM generation...")
        print(f"   Model layers: {len(model.layers)}")
        print(f"   Image shape:  {img_array.shape}")
        print(f"   Pred index:   {pred_index}")

        # ── Find last conv layer ──────────────────────────────────────
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() or 'Conv' in type(layer).__name__:
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            print("❌ No convolutional layer found in model!")
            return None

        print(f"✅ Using conv layer: {last_conv_layer.name}")

        # ── Build gradient model ──────────────────────────────────────
        try:
            grad_model = Model(
                inputs=model.input,
                outputs=[model.output, last_conv_layer.output]
            )
        except Exception as e:
            print(f"❌ Failed to create gradient model: {e}")
            return None

        # ── Forward pass with gradient tape ──────────────────────────
        img_tensor = tf.convert_to_tensor(
            np.expand_dims(img_array, axis=0), dtype=tf.float32
        )

        try:
            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                outputs = grad_model(img_tensor, training=False)

                preds = tf.convert_to_tensor(
                    outputs[0] if isinstance(outputs, list) else outputs[0]
                )
                conv_outputs = tf.convert_to_tensor(
                    outputs[1] if isinstance(outputs, list) else outputs[1]
                )

                # Normalise prediction shape → (batch, num_classes)
                while len(preds.shape) > 2:
                    preds = tf.squeeze(preds, axis=1)
                if len(preds.shape) == 1:
                    preds = tf.expand_dims(preds, axis=0)

                # Normalise conv shape → (batch, H, W, C)
                if len(conv_outputs.shape) == 3:
                    conv_outputs = tf.expand_dims(conv_outputs, axis=0)

                if pred_index is None:
                    pred_index = int(tf.argmax(preds[0]))

                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)
            if grads is None:
                print("❌ Gradients are None!")
                return None

        except Exception as e:
            print(f"❌ Error during forward/gradient pass: {e}")
            traceback.print_exc()
            return None

        # ── Build heatmap ─────────────────────────────────────────────
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        conv_np = conv_outputs[0].numpy()

        for i in range(pooled_grads.shape[0]):
            conv_np[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_np, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() != 0:
            heatmap /= heatmap.max()

        # ── Render & encode ───────────────────────────────────────────
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_jet = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

        original_bgr = cv2.cvtColor(np.uint8(img_array * 255), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original_bgr, 0.5, heatmap_jet, 0.5, 0)

        def to_b64(img_bgr):
            _, buf = cv2.imencode('.png', img_bgr)
            return "data:image/png;base64," + base64.b64encode(buf).decode('utf-8')

        result = {
            "overlay": to_b64(overlay),
            "heatmap": to_b64(heatmap_jet),
        }

        print("✅ Grad-CAM generated successfully!")
        print(f"{'='*60}\n")
        return result

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ GRAD-CAM FATAL ERROR: {e}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        return None