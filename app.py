from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ✅ Load model (your path)
MODEL_PATH = "model/cnn_retinal_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Class names (same order as training)
class_names = ["CATARACT", "DIABETIC RETINOPATHY", "GLAUCOMA", "NORMAL"]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def softmax(x):
    """Convert logits to probabilities safely."""
    x = np.array(x, dtype=np.float32)
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


@app.route("/")
def home():
    return render_template("index.html", prediction=None, confidence=None, image_path=None)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template("index.html", prediction=None, confidence=None, image_path=None)

    # ✅ Save image safely
    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    # ✅ Read + preprocess image
    img = cv2.imread(image_path)
    if img is None:
        return render_template("index.html", prediction="Invalid image file", confidence=None, image_path=None)

    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # ✅ Predict
    pred = model.predict(img)
    pred = np.array(pred)

    prediction_label = None
    confidence = None

    # ------------------ MULTI-CLASS (softmax / logits) ------------------
    if pred.ndim == 2 and pred.shape[1] > 1:
        raw = pred[0]  # shape (4,) usually

        # Convert to probabilities safely
        probs = softmax(raw)

        class_index = int(np.argmax(probs))
        prediction_label = class_names[class_index]
        confidence = round(float(probs[class_index]) * 100, 2)

        # DEBUG (check in terminal)
        print("DEBUG raw output:", raw)
        print("DEBUG probs:", probs)
        print("DEBUG prediction:", prediction_label, "| confidence:", confidence)

    # ------------------ BINARY (sigmoid) ------------------
    else:
        p = float(pred.reshape(-1)[0])  # 0..1
        if p >= 0.5:
            prediction_label = "DISEASE"
            confidence = round(p * 100, 2)
        else:
            prediction_label = "NORMAL"
            confidence = round((1 - p) * 100, 2)

        print("DEBUG binary p:", p)
        print("DEBUG prediction:", prediction_label, "| confidence:", confidence)

    return render_template(
        "index.html",
        prediction=prediction_label,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)