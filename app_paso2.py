from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["*"]}})

# ---------------------------------------
# 1. CARGAR MODELO TFLITE MULTICLASE
# ---------------------------------------
TFLITE_MODEL = "CNN2_Model_FondoOjo_Neli_1_COMPAT.tflite"

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (128, 128)

# ---------------------------------------
# 2. PREPROCESAR IMAGEN
# ---------------------------------------
def preprocesar_imagen(image_path):
    try:
        image = cv2.imread(image_path)

        if image is None:
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMG_SIZE)
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)

        return image
    except:
        return None

# ---------------------------------------
# 3. CLASIFICACIÓN MULTICLASE
# ---------------------------------------
CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative"]

def clasificar_imagen(image_path):
    img = preprocesar_imagen(image_path)
    if img is None:
        return None

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(pred))
    prob = float(pred[idx])

    return {
        "categoria": CLASSES[idx],
        "probabilidad": round(prob * 100, 2)
    }

# ---------------------------------------
# 4. ENDPOINT PARA CLASIFICAR
# ---------------------------------------
@app.route("/clasificar-imagen", methods=["POST"])
def clasificar():
    if "imagen" not in request.files:
        return jsonify({"error": "No se envió imagen."}), 400

    imagen = request.files["imagen"]
    temp_path = "temp_clasif.jpg"
    imagen.save(temp_path)

    resultado = clasificar_imagen(temp_path)
    os.remove(temp_path)

    if resultado is None:
        return jsonify({"error": "Error procesando la imagen."}), 500

    return jsonify(resultado)

# ---------------------------------------
# 5. HOME (probar backend)
# ---------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "mensaje": "Backend CLASIFICACIÓN activo (TFLITE)"})

# ---------------------------------------
# 6. EJECUCIÓN LOCAL
# ---------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
