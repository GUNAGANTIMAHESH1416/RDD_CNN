from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

model = tf.keras.models.load_model("model/cnn_retinal_model.h5")

class_names = ["CATARACT", "DIABETIC RETINOPATHY", "GLAUCOMA", "NORMAL"]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    if file:
        # Save image
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # Read image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        result = class_names[class_index]

        # 👇 THIS IS WHERE IT GOES
        return render_template(
            "index.html",
            prediction=result,
            image_path=image_path
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
