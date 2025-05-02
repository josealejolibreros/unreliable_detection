import os
import warnings
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
warnings.filterwarnings("ignore")
import concurrent.futures


# --------------------------
# Parámetros y configuración
# --------------------------
DATA_FILE = './combined_dataset/combined_checked_with_time_t.csv'  # Ruta al CSV
n_simulations = 100  # Número de simulaciones de cross validation
data_use_ratio = 0.9   # Se descarta el 10% para asegurar 70/10/10 exactos
train_ratio_used = 0.7778   # Dentro del 90%: ~70% para entrenamiento, 20% para validación y test
val_ratio_used = 0.5        # Del 22.22% restante, mitad para validación y mitad para test
num_epochs = 150  # Número de épocas
learning_rate = 0.0001

# Parámetros para la red feedforward
hidden_size1 = 32
hidden_size2 = 16

# Parámetros para la RNN (TimeSeriesModel)
USE_RNN = False  # Cambiar a False para usar la red feedforward
hidden_size_lstm = 64
hidden_size_fc1 = 32
hidden_size_fc2 = 16

# --------------------------
# Funciones de preprocesado
# --------------------------
def preprocess_datetime(df):
    """
    Extrae características de las columnas de fecha/hora, incluyendo 'datetime' y 'start_datetime'.
    Para cada columna encontrada, se extraen: año, mes, día, hora, día de la semana y se aplica codificación cíclica.
    Luego se elimina la columna original.
    """
    for col in ['start_datetime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='ignore')
            #df[f'{col}_year'] = df[col].dt.year
            #df[f'{col}_month'] = df[col].dt.month
            #df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            # Codificación cíclica para mes y hora
            #df[f'{col}_day_sin'] = np.sin(2 * np.pi * df[f'{col}_day'] / 31)
            #df[f'{col}_day_cos'] = np.cos(2 * np.pi * df[f'{col}_day'] / 31)
            #df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[f'{col}_month'] / 12)
            #df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[f'{col}_month'] / 12)
            #df[f'{col}_hour_sin'] = np.sin(2 * np.pi * df[f'{col}_hour'] / 24)
            #df[f'{col}_hour_cos'] = np.cos(2 * np.pi * df[f'{col}_hour'] / 24)
            #df.drop(columns=[col,f'{col}_month',f'{col}_day',f'{col}_hour',f'{col}_dayofweek'], inplace=True, errors="ignore")
            #df.drop(columns=[col,f'{col}_month',f'{col}_hour',f'{col}_dayofweek'], inplace=True, errors="ignore")
    return df

def workunit_types(df, column_name="workunit_id", verbose=False):
    keywords = ["DatesReha", "Amounts", "Dates", "Table", "Your Task"]
    for keyword in keywords:
        df[keyword] = 0
    for index, row in df.iterrows():
        found = False
        for keyword in keywords:
            if isinstance(row[column_name], str) and keyword in row[column_name]:
                df.at[index, keyword] = 1
                found = True
        if not found:
            for keyword in keywords:
                df.at[index, keyword] = 0
    if verbose:
        print("workunit_types:")
        for keyword in keywords:
            print(f"{keyword}: {df[keyword].sum()}")
    return df

def depure_and_divide_data(df, verbose=False, column_name_to_delete = ""):
    # Preprocesar columnas de fecha/hora
    df = preprocess_datetime(df.copy())
    
    # Reemplazar valores en la etiqueta si es necesario
    if 'class' in df.columns.tolist():
        df['class'].replace('Suspicious', 'Unreliable', inplace=True)
    elif 'Prediction' in df.columns.tolist():
        df['Prediction'].replace('Suspicious', 'Unreliable', inplace=True)

    # Añadir columnas derivadas a partir de workunit_id
    df = workunit_types(df, verbose=verbose)
    
    # Eliminar columnas que no se usarán para el entrenamiento
    drop_cols = ['worker_id', 'workunit_id', 'start_datetime', 'source', 'session_id', 
                 'reason', 'recording_id', 'job_id', 'step', 'Prediction', 'Confidence',
                 'date', 'merge', 'is_same_recording_id', 'is_same_focus_count', 
                 'is_samemutation_count', 'is_same_workunit_id', 'verbose_text', 'step_id.1',
                 'pattern', 'Prediction', 'Confidence', 'confidence_without_cal']
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df.drop(columns=[column_name_to_delete], errors='ignore')
    # Codificación de variables no numéricas
    label_encoders = {}
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    for column in non_numeric_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Separar la etiqueta y eliminarla del dataframe
    y = df['class'].values
    df = df.drop(columns=['class'])
    df = df.dropna(axis=1)
    X = df.values
    columns_df_original = df.columns.tolist()
    return X, y, columns_df_original, label_encoders

# --------------------------
# Funciones para calibración
# --------------------------
def histogram_binning_calibration(y_true, y_pred_proba, bins=10):
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins=bin_edges, right=True)
    bin_true_prob = np.zeros(bins)
    bin_count = np.zeros(bins)
    for b in range(1, bins + 1):
        bin_mask = (bin_indices == b)
        if np.any(bin_mask):
            bin_true_prob[b - 1] = np.mean(y_true[bin_mask])
            bin_count[b - 1] = np.sum(bin_mask)
    bin_weight = bin_count / len(y_true)
    return bin_edges, bin_true_prob, bin_weight

def apply_histogram_binning(bin_edges, bin_true_prob, y_pred_proba):
    bin_indices = np.digitize(y_pred_proba, bins=bin_edges, right=True)
    y_pred_calibrated = np.zeros_like(y_pred_proba)
    for b in range(1, len(bin_edges)):
        bin_mask = (bin_indices == b)
        y_pred_calibrated[bin_mask] = bin_true_prob[b - 1]
    return y_pred_calibrated

# --------------------------
# Modelo: red neuronal simple (Feedforward)
# --------------------------
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU6()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(self.relu(out))
        out = self.fc2(out)
        out = self.dropout(self.relu(out))
        out = self.fc3(out)
        return out

# --------------------------
# Modelo: TimeSeriesModel basado en RNN (LSTM)
# --------------------------
class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size_lstm, hidden_size_fc1, hidden_size_fc2, output_size):
        super(TimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size_lstm, batch_first=True)
        self.fc1 = nn.Linear(hidden_size_lstm, hidden_size_fc1)
        self.relu = nn.ReLU6()
        self.fc2 = nn.Linear(hidden_size_fc1, hidden_size_fc2)
        self.fc3 = nn.Linear(hidden_size_fc2, output_size)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        # x tiene forma (batch, seq_len, features); usamos el último timestep
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.dropout(self.relu(out))
        out = self.dropout(self.fc2(out))
        out = self.dropout(self.relu(out))
        out = self.fc3(out)
        return out

def balance_classes(X_train, X_test, y_train, y_test, random_state=42):
    """
    Reduce la clase predominante en X_train y X_test para obtener una distribución equilibrada 50%-50%.
    
    Parámetros:
        X_train (numpy array): Datos de entrenamiento.
        X_test (numpy array): Datos de prueba.
        y_train (numpy array): Etiquetas de entrenamiento.
        y_test (numpy array): Etiquetas de prueba.
        random_state (int): Semilla aleatoria para la selección de muestras.
    
    Retorna:
        X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced
    """

    # Convertir a DataFrame para facilitar manipulación
    train_df = pd.DataFrame(X_train)
    train_df['label'] = y_train

    test_df = pd.DataFrame(X_test)
    test_df['label'] = y_test

    # Identificar la clase predominante y minoritaria
    majority_class_train = train_df['label'].value_counts().idxmax()
    minority_class_train = train_df['label'].value_counts().idxmin()
    
    majority_class_test = test_df['label'].value_counts().idxmax()
    minority_class_test = test_df['label'].value_counts().idxmin()

    # Número de elementos de la clase minoritaria
    n_minority_train = train_df['label'].value_counts()[minority_class_train]
    n_minority_test = test_df['label'].value_counts()[minority_class_test]

    # Reducir la clase predominante en train
    train_majority_df = train_df[train_df['label'] == majority_class_train].sample(n=n_minority_train, random_state=random_state)
    train_minority_df = train_df[train_df['label'] == minority_class_train]

    # Reducir la clase predominante en test
    test_majority_df = test_df[test_df['label'] == majority_class_test].sample(n=n_minority_test, random_state=random_state)
    test_minority_df = test_df[test_df['label'] == minority_class_test]

    # Concatenar clases balanceadas y mezclar
    balanced_train_df = pd.concat([train_majority_df, train_minority_df]).sample(frac=1, random_state=random_state)
    balanced_test_df = pd.concat([test_majority_df, test_minority_df]).sample(frac=1, random_state=random_state)

    # Extraer nuevamente los valores
    X_train_balanced = balanced_train_df.drop(columns=['label']).values
    y_train_balanced = balanced_train_df['label'].values

    X_test_balanced = balanced_test_df.drop(columns=['label']).values
    y_test_balanced = balanced_test_df['label'].values

    return X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced

# --------------------------
# Funciones de división estratificada con reintentos
# --------------------------
def stratified_split_with_retry(X, y, test_size, min_ratio, max_retries=1000):
    """Realiza train_test_split, asegurando que todas las clases tengan al menos min_ratio en el subconjunto."""
    for _ in range(max_retries):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=np.random.randint(10000)
        )
        unique_classes, counts = np.unique(y_train, return_counts=True)
        min_required = np.ceil(len(y_train) * min_ratio).astype(int)
        if np.all(counts >= min_required):
            return X_train, X_test, y_train, y_test
    raise ValueError(f"No se pudo encontrar un split válido después de {max_retries} intentos.")

# --------------------------
# Funciones para SMOTE
# --------------------------
def apply_smote_to_minority(X_train, y_train):
    """Aplica SMOTE solo a la clase minoritaria en y_train."""
    unique_classes, counts = np.unique(y_train, return_counts=True)
    min_class = unique_classes[np.argmin(counts)]
    smote = SMOTE(sampling_strategy={min_class: int(np.median(counts))}, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# --------------------------
# Función para entrenar y evaluar una simulación
# --------------------------
def run_simulation(X, y, columns_df_original, label_encoders):
    min_ratio = 0.23  # Cada clase debe tener al menos el 23% de instancias en cada split

    # Selección del 90% de los datos
    X_used, X_drop, y_used, y_drop = stratified_split_with_retry(X, y, test_size=1 - data_use_ratio, min_ratio=min_ratio)
    # Separación en entrenamiento (70%) y temp (20%)
    X_train, X_temp, y_train, y_temp = stratified_split_with_retry(X_used, y_used, test_size=1 - train_ratio_used, min_ratio=min_ratio)
    # Aplicar SMOTE solo a la clase minoritaria en entrenamiento
    X_train, X_temp, y_train, y_temp = balance_classes(X_train, X_temp, y_train, y_temp)
    X_train, y_train = apply_smote_to_minority(X_train, y_train)

    # División en validación y test (~10% cada uno del total)
    X_val, X_test, y_val, y_test = stratified_split_with_retry(X_temp, y_temp, test_size=0.5, min_ratio=min_ratio)

    # Normalización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Conversión a tensores de PyTorch
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Construcción del modelo: se elige RNN o feedforward según USE_RNN
    input_size = X_train.shape[1]
    output_size = len(np.unique(y))
    if USE_RNN:
        # Para RNN, reacomodar la entrada a 3D: (batch, seq_len=1, features)
        X_train_tensor = X_train_tensor.unsqueeze(1)
        X_val_tensor = X_val_tensor.unsqueeze(1)
        X_test_tensor = X_test_tensor.unsqueeze(1)
        model = TimeSeriesModel(input_size, hidden_size_lstm, hidden_size_fc1, hidden_size_fc2, output_size)
    else:
        model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)

    # Definir los pesos inversamente proporcionales a la frecuencia de las clases
    #class_weights = torch.tensor([2900 / (2900 + 880), 880 / (2900 + 880)])  # Normalizado
    #class_weights = torch.tensor([0.3, 0.7])  # Normalizado
    #class_weights = class_weights.to(device)  # Si usas GPU

    # Crear la función de pérdida con los pesos
    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Entrenamiento
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Calibración con el conjunto de validación
    model.eval()
    with torch.no_grad():
        outputs_val = model(X_val_tensor)
        probabilities_val = torch.softmax(outputs_val, dim=1).numpy()
        y_val_proba = probabilities_val[:, 1]  # Probabilidad de la clase positiva

    bin_edges, bin_true_prob, _ = histogram_binning_calibration(y_val, y_val_proba, bins=10)

    # Predicción en el conjunto de prueba
    with torch.no_grad():
        outputs_test = model(X_test_tensor)
        probabilities_test = torch.softmax(outputs_test, dim=1).numpy()
        y_test_proba = probabilities_test[:, 1]
        #y_test_proba_calibrated = apply_histogram_binning(bin_edges, bin_true_prob, y_test_proba)
        #y_pred = (y_test_proba_calibrated >= 0.5).astype(int)
        y_pred = (y_test_proba >= 0.5).astype(int)

    # Cálculo de métricas
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    return precision, recall, accuracy, f1

# --------------------------
# Función para graficar las métricas
# --------------------------
def compress_metrics(metrics_list):
    precisions = np.array([m[0] for m in metrics_list])
    recalls    = np.array([m[1] for m in metrics_list])
    accuracies = np.array([m[2] for m in metrics_list])
    f1s        = np.array([m[3] for m in metrics_list])
    mean_precision = np.mean( precisions[precisions > 0.05])
    mean_recalls = np.mean( recalls[recalls > 0.05])
    mean_accuracies = np.mean( accuracies[accuracies > 0.05])
    mean_f1s = np.mean( f1s[f1s > 0.05])
    n = len(metrics_list)
    return n, mean_precision, mean_recalls, mean_accuracies, mean_f1s
    


def plot_metrics(metrics_list, column_name = ""):
    precisions = np.array([m[0] for m in metrics_list])
    recalls    = np.array([m[1] for m in metrics_list])
    accuracies = np.array([m[2] for m in metrics_list])
    f1s        = np.array([m[3] for m in metrics_list])
    mean_precision = np.mean( precisions[precisions > 0.05])
    mean_recalls = np.mean( recalls[recalls > 0.05])
    mean_accuracies = np.mean( accuracies[accuracies > 0.05])
    mean_f1s = np.mean( f1s[f1s > 0.05])
    n = len(metrics_list)

    
    df = pd.DataFrame({
        "Simulation": np.arange(1, n+1),
        "Precision": precisions,
        "Recall": recalls,
        "Accuracy": accuracies,
        "F1 Score": f1s
    })
    df.to_csv(f'./plots_deleting_columns/delete_{column_name}.csv', index=False, mode="w")

    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(12, 8))
    x = np.arange(1, n+1)
    plt.plot(x, precisions, 'o-', label='Precision')
    plt.hlines(mean_precision, 1, n, colors='blue', linestyles='dashed')
    plt.plot(x, recalls, 'o-', label='Recall')
    plt.hlines(mean_recalls, 1, n, colors='orange', linestyles='dashed')
    plt.plot(x, accuracies, 'o-', label='Accuracy')
    plt.hlines(mean_accuracies, 1, n, colors='green', linestyles='dashed')
    plt.plot(x, f1s, 'o-', label='F1')
    plt.hlines(mean_f1s, 1, n, colors='red', linestyles='dashed')
    plt.xlabel("Simulation")
    plt.ylabel("Metric")
    plt.title(f"Performance metrics each simulation and mean {column_name}")
    plt.legend()
    plt.grid(True)
    plt.xticks(x)
    #plt.show()
    plt.savefig(f'./plots_deleting_columns/delete_{column_name}.svg', format='svg', dpi=1200)

def plot_means(names_deleted, means_precision, means_recall, means_accuracy, means_f1, ns):
    # Convertir a numpy arrays para facilitar el manejo
    names_deleted = np.array(names_deleted)
    means_precision = np.array(means_precision)
    means_recall = np.array(means_recall)
    means_accuracy = np.array(means_accuracy)
    means_f1 = np.array(means_f1)
    
    # Ordenar los datos en función de means_recall en orden descendente
    sorted_indices = np.argsort(-means_recall)
    names_deleted = names_deleted[sorted_indices]
    means_precision = means_precision[sorted_indices]
    means_recall = means_recall[sorted_indices]
    means_accuracy = means_accuracy[sorted_indices]
    means_f1 = means_f1[sorted_indices]

    # Guardar datos en CSV
    df = pd.DataFrame({
        "Deleted Variable": names_deleted,
        "Mean Precision": means_precision,
        "Mean Recall": means_recall,
        "Mean Accuracy": means_accuracy,
        "Mean F1 Score": means_f1
    })
    df.to_csv(f'./plots_deleting_columns/delete_all.csv', index=False, mode="w")
    
    # Crear la figura con más espacio para los labels del eje Y
    fig, ax = plt.subplots(figsize=(10, 12))
    plt.subplots_adjust(left=0.3)  # Ajustar margen izquierdo para que no se corten los labels
    
    # Generar posiciones en el eje Y con mayor separación
    y_positions = np.arange(len(names_deleted)) * 1.5
    
    # Graficar cada métrica con un estilo diferente
    ax.scatter(means_precision, y_positions, label='Precision', marker='o', color='blue')
    ax.scatter(means_recall, y_positions, label='Recall', marker='s', color='red')
    ax.scatter(means_accuracy, y_positions, label='Accuracy', marker='^', color='green')
    ax.scatter(means_f1, y_positions, label='F1 Score', marker='D', color='purple')
    
    # Ajustar espaciado entre labels del eje y
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names_deleted, fontsize=12, fontname='Arial')
    ax.invert_yaxis()  # Invertir el eje Y para mantener el orden descendente
    
    # Etiquetas y título
    ax.set_xlabel("Valores de Métricas", fontsize=12, fontname='Arial')
    ax.set_ylabel("Nombres Eliminados", fontsize=12, fontname='Arial')
    ax.set_title("Comparación de Métricas por Elemento Eliminado", fontsize=14, fontname='Arial')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Mostrar la gráfica
    

    
    # Etiquetas y título
    plt.xlabel("Metrics")
    plt.ylabel("Deleted variables")
    plt.title("Performance metrics per deleted element")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Mostrar la gráfica
    #plt.show()
    plt.savefig(f'./plots_deleting_columns/delete_all.svg', format='svg', dpi=1200)


# --------------------------
# Flujo principal
# --------------------------
# def main():
#     df = pd.read_csv(DATA_FILE)
#     means_precision = []
#     means_recall = []
#     means_accuracy = []
#     means_f1 = []
#     ns = []
#     names_deleted = []
#     for column_name_to_delete in df.columns.values:
#         if column_name_to_delete == "class":
#             continue
#         X, y, columns_df_original, label_encoders = depure_and_divide_data(df, verbose=False, column_name_to_delete=column_name_to_delete)    
#         metrics_list = []
#         for sim in range(n_simulations):
#             print(f"Simulación {sim+1}/{n_simulations}")
#             precision, recall, accuracy, f1 = run_simulation(X, y, columns_df_original, label_encoders)
#             print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  Accuracy: {accuracy:.4f}  F1: {f1:.4f}")
#             metrics_list.append((precision, recall, accuracy, f1))
#         n, mean_precision, mean_recalls, mean_accuracies, mean_f1s = compress_metrics(metrics_list)
#         means_precision.append(mean_precision)
#         means_recall.append(mean_recalls)
#         means_accuracy.append(mean_accuracies)
#         means_f1.append(mean_f1s)
#         ns.append(n)
#         names_deleted.append(column_name_to_delete)
#         plot_metrics(metrics_list, column_name_to_delete)
#     plot_means(names_deleted, means_precision, means_recall, means_accuracy, means_f1, ns)

def process_column(column_name_to_delete, df, n_simulations):
    """ Ejecuta las simulaciones para una columna eliminada del dataset """
    if column_name_to_delete in ["class", "worker_id", "workunit_id", "step_i.1", "pattern", "merge", "Confidence", 
                                 "Prediction", "step_id.1", "is_same_focus_count", "is_same_workunit_id",
                                 "confidence_without_cal", "is_same_mouse_move_count", "is_same_recording_id",
                                 "is_same_mutation_count", "recording_id", "date"]:
        return None  # Ignorar la columna "class"

    X, y, columns_df_original, label_encoders = depure_and_divide_data(df, verbose=False, column_name_to_delete=column_name_to_delete)
    
    metrics_list = []
    for sim in range(n_simulations):
        print(f"Simulación {sim+1}/{n_simulations} para {column_name_to_delete}")
        precision, recall, accuracy, f1 = run_simulation(X, y, columns_df_original, label_encoders)
        print(f"  {column_name_to_delete} - Precision: {precision:.4f}  Recall: {recall:.4f}  Accuracy: {accuracy:.4f}  F1: {f1:.4f}")
        metrics_list.append((precision, recall, accuracy, f1))
    
    n, mean_precision, mean_recall, mean_accuracy, mean_f1 = compress_metrics(metrics_list)
    plot_metrics(metrics_list, column_name_to_delete)
    
    return (column_name_to_delete, n, mean_precision, mean_recall, mean_accuracy, mean_f1)

def main():
    df = pd.read_csv(DATA_FILE)
    df.drop(columns=["worker_id", "workunit_id", "step_i.1", "pattern", "merge", "Confidence", 
                                 "Prediction", "step_id.1", "is_same_focus_count", "is_same_workunit_id",
                                 "confidence_without_cal", "is_same_mouse_move_count", "is_same_recording_id",
                                 "is_same_mutation_count", "date", "recording_id"], errors="ignore")
    results = []
    
    #with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_column, col, df, n_simulations): col for col in df.columns.values}
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:  # Evita agregar None si se ignoró la columna "class"
                results.append(result)
    
    # Extraer métricas de los resultados
    names_deleted, ns, means_precision, means_recall, means_accuracy, means_f1 = zip(*results)
    
    # Graficar resultados agregados
    plot_means(names_deleted, means_precision, means_recall, means_accuracy, means_f1, ns)

if __name__ == '__main__':
    main()
