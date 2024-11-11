import time
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from ml_auth import train_model, predict_anomaly

app = Flask(__name__)
CORS(app)

# Swagger setup
SWAGGER_URL = '/docs'  # URL where Swagger UI will be available
API_URL = '/static/swagger.json'  # Path to the Swagger JSON file
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI URL
    API_URL,  # Swagger file path
    config={  # Swagger UI config overrides
        'app_name': "ML Detection API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/train', methods=['POST'])
def train():
    # Verificar si el cuerpo de la solicitud tiene datos JSON
    if not request.is_json:
        return jsonify({"error": "No se adjuntó un JSON válido"}), 400

    # Obtener los datos del cuerpo de la solicitud
    data = request.get_json()

    # Verificar si se envió información válida para entrenamiento
    if not data or 'data' not in data:
        return jsonify({"error": "El JSON no contiene datos válidos"}), 400

    # Obtener el valor de updateModelVersion; si no se envía, por defecto es False
    update_model_version = bool(data.get('updateModelVersion', False))

    # Registrar el inicio del proceso de entrenamiento
    start_time = time.time()

    try:
        # Llamar a la función de entrenamiento del modelo con los datos recibidos
        result = train_model(data, update_model_version)

        # Verificar si hubo un error en el entrenamiento
        if "error" in result:
            return jsonify(result), 500

        # Devolver el resultado del entrenamiento del modelo como respuesta JSON
        return jsonify(result), 200

    except Exception as e:
        # Manejar cualquier error durante el entrenamiento del modelo
        result = {
            "status": "failure",
            "error_message": str(e),
            "training_details": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)),
                "duration": int(time.time() - start_time)
            }
        }

        return jsonify(result), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Realiza predicciones de anomalías en nuevas sesiones.
    ---
    parameters:
      - in: body
        name: body
        required: true
        description: Datos de sesiones para predecir
        schema:
          type: object
          properties:
            session_duration:
              type: number
    responses:
      200:
        description: Predicciones de anomalías
    """
    # Obtener datos de predicción en formato JSON
    data = request.get_json()
        
    # Realizar predicción
    predictions = predict_anomaly(data)
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
