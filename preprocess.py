import pandas as pd
import numpy as np

def load_and_preprocess_json(data):
    """
    Preprocesa los datos en formato JSON y los convierte en un DataFrame adecuado para el modelo.
    """
    sessions = []

    # Recorremos cada sesión de usuario en el JSON
    for session_data in data['data']:
        user_id = session_data['user_id']
        session_id = session_data['session_id']

        # Recorremos cada evento en el historial de uso de la sesión
        for event in session_data['usage_history']:
            timestamp = event['timestamp']
            app_name = event['app_name']
            event_type = event['event_type']

            sessions.append({
                'user_id': user_id,
                'session_id': session_id,
                'timestamp': timestamp,
                'app_name': app_name,
                'event_type': event_type
            })

    # Convertimos la lista de sesiones en un DataFrame de pandas
    df = pd.DataFrame(sessions)

    # Convertir la columna 'timestamp' a formato datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')  # Usa 'coerce' para manejar conversiones inválidas

    # Eliminar filas con timestamp NaN
    df = df.dropna(subset=['timestamp'])

    # Calcular la duración de la sesión
    session_duration = df.groupby('session_id').apply(
        lambda x: (x['timestamp'].max() - x['timestamp'].min()).total_seconds()
    ).reset_index(name='session_duration')

    # Unir la duración de la sesión al DataFrame original
    df = pd.merge(df, session_duration, on='session_id', how='left')

    # Eliminar filas con session_duration NaN
    df = df.dropna(subset=['session_duration'])

    # Calcular los tiempos entre eventos y manejar sesiones con menos de 3 eventos
    def calculate_time_between_events(group):
        if len(group) < 3:
            return pd.Series([np.nan, np.nan], index=['time_between_opened_interaction', 'time_between_interaction_closed'])
        else:
            time_between_opened_interaction = (group['timestamp'].iloc[1] - group['timestamp'].iloc[0]).total_seconds()
            time_between_interaction_closed = (group['timestamp'].iloc[2] - group['timestamp'].iloc[1]).total_seconds()
            return pd.Series([time_between_opened_interaction, time_between_interaction_closed], 
                             index=['time_between_opened_interaction', 'time_between_interaction_closed'])

    df[['time_between_opened_interaction', 'time_between_interaction_closed']] = df.groupby('session_id').apply(calculate_time_between_events)

    # Rellenar valores NaN en los tiempos entre eventos con 0
    df['time_between_opened_interaction'] = df['time_between_opened_interaction'].fillna(0)
    df['time_between_interaction_closed'] = df['time_between_interaction_closed'].fillna(0)

    # Codificar la secuencia de eventos
    def encode_sequence(group):
        sequence = "-".join(group['event_type'].tolist())
        if sequence == "Opened-User Interaction-Closed":
            return 0  # Secuencia normal
        elif sequence == "Opened-User Interaction-Broken":
            return 1  # Secuencia anómala (termina con error)
        else:
            return 2  # Cualquier otra secuencia es anómala

    df['sequence_encoded'] = df.groupby('session_id').apply(encode_sequence).reset_index(drop=True)

    # --- Agregar la columna 'anomaly' usando label_sequence ---
    df['anomaly'] = df.groupby('session_id').apply(label_sequence).reset_index(drop=True)

    # Eliminar cualquier fila que aún contenga NaN
    df = df.dropna()

    return df

def label_sequence(group):
    """
    Etiqueta la secuencia de eventos en función de las reglas establecidas:
    - Secuencia correcta: Opened -> User Interaction -> Closed (o Broken)
    - Duración de la sesión entre límites normales
    """
    event_types = group['event_type'].tolist()

    # Etiquetar secuencias con menos de 3 eventos como anómalas
    if len(event_types) < 3:
        return 1  # Anomalía si la secuencia está incompleta

    # Etiquetar secuencias que no sigan el patrón correcto
    if event_types != ['Opened', 'User Interaction', 'Closed'] and event_types != ['Opened', 'User Interaction', 'Broken']:
        return 1  # Anomalía si la secuencia es incorrecta

    # Verificar la duración de la sesión
    session_duration = group['session_duration'].iloc[0]
    if session_duration < 30 or session_duration > 3600:  # Ejemplo: sesiones menores a 30s o mayores a 1h
        return 1  # Anomalía si la duración es anómala

    return 0  # Normal si todo está bien

def preprocess_json_for_prediction(data):
    """
    Preprocesa los datos en formato JSON para predicción y los convierte en un DataFrame adecuado.
    """
    sessions = []
    
    # Recorremos cada sesión de usuario en el JSON
    for session_data in data['data']:
        user_id = session_data['user_id']
        session_id = session_data['session_id']
        
        # Recorremos cada evento en el historial de uso de la sesión
        for event in session_data['usage_history']:
            timestamp = event['timestamp']
            app_name = event['app_name']
            event_type = event['event_type']
            
            sessions.append({
                'user_id': user_id,
                'session_id': session_id,
                'timestamp': timestamp,
                'app_name': app_name,
                'event_type': event_type
            })
    
    # Convertimos la lista de sesiones en un DataFrame de pandas
    df = pd.DataFrame(sessions)
    
    # Convertir el timestamp a formato datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calcular la duración de la sesión
    session_duration = df.groupby('session_id').apply(
        lambda x: (x['timestamp'].max() - x['timestamp'].min()).total_seconds()
    ).reset_index(name='session_duration')
    
    # Unir la duración de la sesión al DataFrame original
    df = pd.merge(df, session_duration, on='session_id')
    
    # Calcular el tiempo entre eventos
    df['time_between_opened_interaction'] = df.groupby('session_id').apply(
        lambda x: (x['timestamp'].iloc[1] - x['timestamp'].iloc[0]).total_seconds() if len(x) > 1 else 0
    ).reset_index(drop=True)
    
    df['time_between_interaction_closed'] = df.groupby('session_id').apply(
        lambda x: (x['timestamp'].iloc[2] - x['timestamp'].iloc[1]).total_seconds() if len(x) > 2 else 0
    ).reset_index(drop=True)
    
    # Codificar la secuencia de eventos
    def encode_sequence(group):
        sequence = "-".join(group['event_type'].tolist())
        if sequence == "Opened-User Interaction-Closed":
            return 0  # Secuencia normal
        elif sequence == "Opened-User Interaction-Broken":
            return 1  # Secuencia anómala (termina con error)
        else:
            return 2  # Cualquier otra secuencia es anómala
    
    df['sequence_encoded'] = df.groupby('session_id').apply(encode_sequence).reset_index(drop=True)
    
    # Rellenar cualquier NaN restante antes de la predicción
    df = df.fillna(0)

    return df