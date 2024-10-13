import pandas as pd
import time
from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from ml_auth import train_model, predict_anomaly

app = Flask(__name__)

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
    update_model_version = data.get('updateModelVersion', False)

    # Registrar el inicio del proceso de entrenamiento
    start_time = time.time()

    try:
        # Llamar a la función de entrenamiento del modelo con los datos recibidos
        result = train_model(data, update_model_version)

        # Calcular la duración del entrenamiento
        duration = int(time.time() - start_time)

        # Agregar información adicional al resultado
        result.update({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)),
            "duration": duration,
            "status": "success",
            "updateModelVersion": update_model_version  # Devolver el valor del parámetro recibido
        })

    except Exception as e:
        # Manejar cualquier error durante el entrenamiento del modelo
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)),
            "duration": int(time.time() - start_time),
            "status": "failure",
            "error_message": str(e),
            "updateModelVersion": update_model_version  # Devolver el valor del parámetro recibido
        }

    # Devolver el resultado del entrenamiento del modelo como respuesta JSON
    return jsonify(result)


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
    
    # Convertir el JSON a un DataFrame
    new_data = pd.DataFrame(data)
    
    # Realizar predicción
    predictions = predict_anomaly(new_data)
    
    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(debug=True)
