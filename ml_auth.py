import pickle
import os
import threading
import time
import psutil
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from preprocess import load_and_preprocess_json
from preprocess import preprocess_json_for_prediction

def train_model(train_data, update_model_version=False):
    """
    Entrena un modelo KNN con validación cruzada estratificada y registra el uso de CPU.
    """
    update_model_version = bool(update_model_version)

    # Variables para almacenar el uso de CPU
    cpu_percentages = []
    stop_cpu_measurement = False

    # Función para medir el uso de CPU
    def measure_cpu_usage():
        while not stop_cpu_measurement:
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_percentages.append(cpu_percent)
            time.sleep(0.5)  # Medir cada 0.5 segundos

    # Iniciar el hilo que medirá el uso de CPU
    cpu_thread = threading.Thread(target=measure_cpu_usage)
    cpu_thread.start()

    try:
        # Registrar el inicio del entrenamiento
        start_time = time.time()

        # Preprocesar los datos recibidos en formato JSON
        df = load_and_preprocess_json(train_data)

        # Usar las nuevas características para el modelo
        X = df[['session_duration', 'time_between_opened_interaction', 'time_between_interaction_closed', 'sequence_encoded']]
        y = df['anomaly']

        # Verificar si existen NaN en las etiquetas
        if y.isnull().any():
            stop_cpu_measurement = True
            cpu_thread.join()
            return {"error": "Las etiquetas contienen valores NaN. Revisa el preprocesamiento de los datos."}, 400

        # Escalar los datos (normalización)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Crear la validación cruzada estratificada con 5 pliegues
        skf = StratifiedKFold(n_splits=5)

        # Definir el rango de valores para k en la búsqueda
        param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15]}

        # Configurar la búsqueda en cuadrícula con validación cruzada estratificada
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=skf, scoring='accuracy')

        # Entrenar el modelo usando la búsqueda en cuadrícula
        grid_search.fit(X_scaled, y)

        # Obtener el mejor valor de k encontrado
        best_k = grid_search.best_params_['n_neighbors']
        print(f"Mejor valor de k encontrado: {best_k}")

        # Entrenar el modelo KNN con el mejor valor de k
        knn_best = KNeighborsClassifier(n_neighbors=best_k)
        knn_best.fit(X_scaled, y)

        # Guardar el modelo entrenado en un archivo
        model_path = 'knn_model.pkl'
        with open(model_path, 'wb') as model_file:
            pickle.dump(knn_best, model_file)

        # Guardar el escalador
        scaler_path = 'scaler.pkl'
        with open(scaler_path, 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)

        # Realizar predicciones en todo el conjunto de datos para calcular las métricas
        y_pred = knn_best.predict(X_scaled)

        # Calcular las métricas
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        confusion = confusion_matrix(y, y_pred)

        # Convertir los valores numpy a tipos nativos de Python (int) para evitar problemas con JSON
        confusion_matrix_dict = {
            "true_negatives": int(confusion[0, 0]),
            "false_positives": int(confusion[0, 1]),
            "false_negatives": int(confusion[1, 0]),
            "true_positives": int(confusion[1, 1])
        }

        # Calcular timestamp y duración
        end_time = time.time()
        duration = int(end_time - start_time)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time))

        # Detener la medición de CPU y esperar a que el hilo termine
        stop_cpu_measurement = True
        cpu_thread.join()

        # Calcular el uso promedio de CPU
        avg_cpu_usage = sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0.0

        # Construir el resultado en la estructura solicitada
        result = {
            "status": "success",
            "message": f"Modelo entrenado y guardado en {model_path}",
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            },
            "confusion_matrix": confusion_matrix_dict,
            "training_details": {
                "cpu_usage": avg_cpu_usage,
                "duration": duration,
                "timestamp": timestamp
            }
        }

        return result

    except Exception as e:
        # Detener la medición de CPU y esperar a que el hilo termine en caso de error
        stop_cpu_measurement = True
        cpu_thread.join()
        return {"error": str(e)}, 500

def predict_anomaly(data):
    """
    Carga el modelo entrenado y predice si los datos nuevos contienen anomalías.
    """
    # Preprocesar los datos recibidos en formato JSON
    df = preprocess_json_for_prediction(data)

    # Cargar el modelo entrenado
    model_path = 'knn_model.pkl'
    if not os.path.exists(model_path):
        return {"error": "Modelo no encontrado. Entrena el modelo primero."}

    with open(model_path, 'rb') as model_file:
        knn = pickle.load(model_file)

    # Usar las mismas características que el entrenamiento para hacer la predicción
    X_new = df[['session_duration', 'time_between_opened_interaction', 'time_between_interaction_closed', 'sequence_encoded']]

    # Escalar los datos antes de hacer predicciones
    scaler_path = 'scaler.pkl'
    if not os.path.exists(scaler_path):
        return {"error": "Escalador no encontrado. Entrena el modelo primero."}

    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    X_new_scaled = scaler.transform(X_new)

    # Realizar las predicciones
    predictions = knn.predict(X_new_scaled)

    # Convertir predicciones a una lista
    df['prediction'] = predictions
    result = df[['session_id', 'prediction']].to_dict(orient='records')
    
    return result
