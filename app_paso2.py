from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Permitir peticiones desde otros orígenes

# Cargar el modelo multiclase entrenado
modelo_path = "CNN2_Model_FondoOjo_Neli_1.h5"
model = keras.models.load_model(modelo_path)

# Definir etiquetas de clase
categorias = ["Mild", "Moderate", "No_DR", "Proliferative", "Severe"]

def preprocesar_imagen_bytes(image_bytes, size=(128, 128)):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(size)
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/clasificar-retinopatia", methods=["POST"])
def clasificar_retinopatia():
    if "imagen" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    archivo = request.files["imagen"]
    if archivo.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    try:
        imagen_bytes = archivo.read()
        imagen_procesada = preprocesar_imagen_bytes(imagen_bytes)
        prediccion = model.predict(imagen_procesada)
        clase = categorias[np.argmax(prediccion)]
        return jsonify({"clasificacion": clase})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
