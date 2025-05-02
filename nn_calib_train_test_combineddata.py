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
warnings.filterwarnings("ignore")

# --------------------------
# Parámetros y configuración
# --------------------------
DATA_FILE = './combined_dataset/combined_checked_with_time_t.csv'  # Actualiza la ruta a tu archivo CSV
n_simulations = 100  # Número de simulaciones de cross validation
# Porcentaje de datos a utilizar (se descarta el 10% para asegurar 70/10/10 exactos)
data_use_ratio = 0.9  
# Splits dentro del subconjunto usado (del 90%):
# entrenamiento = 70% del total, validación = 10% y prueba = 10%.
# Calculando en relación al subconjunto usado (90%): 
# train_size_used = 70/90 ≈ 0.7778 y el restante 0.2222 se dividirá equitativamente para validación y test.
train_ratio_used = 0.7778  
val_ratio_used = 0.5  # del 22.22% restante se usa la mitad para validación y la otra mitad para test

num_epochs = 150  # Puedes ajustar el número de épocas según convenga
learning_rate = 0.0001

# Parámetros de la red
hidden_size1 = 32
hidden_size2 = 16

# --------------------------
# Funciones de preprocesado
# --------------------------
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

def depure_and_divide_data(df, verbose=False):
    # Si la columna 'class' existe, reemplaza valores si es necesario
    if 'class' in df.columns.tolist():
        df['class'].replace('Suspicious', 'Unreliable', inplace=True)
    elif 'Prediction' in df.columns.tolist():
        df['Prediction'].replace('Suspicious', 'Unreliable', inplace=True)

    # Añadir columnas derivadas
    df = workunit_types(df, verbose=verbose)
    
    # Se eliminan columnas que no se usarán para el entrenamiento
    drop_cols = ['worker_id', 'workunit_id', 'start_datetime', 'source', 'session_id', 
                 'reason', 'recording_id', 'job_id', 'step', 'Prediction', 'Confidence',
                 'date', 'merge', 'is_same_recording_id', 'is_same_focus_count', 
                 'is_samemutation_count', 'is_same_workunit_id', 'verbose_text', 'step_id.1',
                 'pattern', 'Prediction', 'Confidence', 'confidence_without_cal']
    df = df.drop(columns=drop_cols, errors='ignore')

    # Codificación de variables no numéricas
    label_encoders = {}
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    for column in non_numeric_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Se separa la etiqueta
    y = df['class'].values
    df = df.drop(columns=['class'])
    # Eliminar columnas con NaN
    df = df.dropna(axis=1)
    X = df.values
    columns_df_original = df.columns.tolist()

    return X, y, columns_df_original, label_encoders

# --------------------------
# Funciones para calibración
# --------------------------
def histogram_binning_calibration(y_true, y_pred_proba, bins=10):
    """
    Calcula los parámetros de calibración (bin_edges, bin_true_prob y bin_weight)
    usando el conjunto de validación.
    """
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
# Modelo: red neuronal simple
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

def stratified_split_with_retry1(X, y, test_size, min_ratio, max_retries=100000):
    """ Realiza train_test_split, asegurando que todas las clases tengan al menos min_ratio en el subconjunto. """
    for _ in range(max_retries):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=np.random.randint(10000)
        )
        unique_classes, counts = np.unique(y_train, return_counts=True)
        min_required = np.ceil(len(y_train) * min_ratio).astype(int)

        if np.all(counts >= min_required):  # Si todas las clases cumplen, devolvemos los datos
            return X_train, X_test, y_train, y_test

    raise ValueError(f"No se pudo encontrar un split válido después de {max_retries} intentos.")

def run_simulation1(X, y, columns_df_original, label_encoders):
    min_ratio = 0.20  # Cada clase debe tener al menos el 5% de instancias en cada split

    # Selección del 90% de los datos
    X_used, X_drop, y_used, y_drop = stratified_split_with_retry(X, y, test_size=1 - data_use_ratio, min_ratio=min_ratio)

    # Separación en entrenamiento (70%) y temp (20%)
    X_train, X_temp, y_train, y_temp = stratified_split_with_retry(X_used, y_used, test_size=1 - train_ratio_used, min_ratio=min_ratio)

    # División en validación y test (cada uno 50% del 22.22% ~10% del total)
    X_val, X_test, y_val, y_test = stratified_split_with_retry(X_temp, y_temp, test_size=0.5, min_ratio=min_ratio)

    # Normalización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Conversión a tensores de PyTorch
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Construcción del modelo
    input_size = X_train.shape[1]
    output_size = len(np.unique(y))
    model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)

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
        y_test_proba_calibrated = apply_histogram_binning(bin_edges, bin_true_prob, y_test_proba)
        y_pred = (y_test_proba_calibrated >= 0.5).astype(int)

    # Cálculo de métricas
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return precision, recall, accuracy, f1

def stratified_split_with_retry(X, y, test_size, min_ratio, max_retries=1000):
    """ Realiza train_test_split, asegurando que todas las clases tengan al menos min_ratio en el subconjunto. """
    for _ in range(max_retries):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=np.random.randint(10000)
        )
        unique_classes, counts = np.unique(y_train, return_counts=True)
        min_required = np.ceil(len(y_train) * min_ratio).astype(int)

        if np.all(counts >= min_required):  # Si todas las clases cumplen, devolvemos los datos
            return X_train, X_test, y_train, y_test

    raise ValueError(f"No se pudo encontrar un split válido después de {max_retries} intentos.")

def apply_smote_to_minority(X_train, y_train):
    """ Aplica SMOTE solo a la clase minoritaria en y_train. """
    unique_classes, counts = np.unique(y_train, return_counts=True)
    min_class = unique_classes[np.argmin(counts)]  # Encuentra la clase con menos ejemplos

    smote = SMOTE(sampling_strategy={min_class: int(np.median(counts))}, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    return X_resampled, y_resampled

def run_simulation(X, y, columns_df_original, label_encoders):
    min_ratio = 0.23  # Cada clase debe tener al menos el 5% de instancias en cada split

    # Selección del 90% de los datos
    X_used, X_drop, y_used, y_drop = stratified_split_with_retry(X, y, test_size=1 - data_use_ratio, min_ratio=min_ratio)

    # Separación en entrenamiento (70%) y temp (20%)
    X_train, X_temp, y_train, y_temp = stratified_split_with_retry(X_used, y_used, test_size=1 - train_ratio_used, min_ratio=min_ratio)

    # Aplicar SMOTE solo a la clase minoritaria en entrenamiento
    X_train, y_train = apply_smote_to_minority(X_train, y_train)

    # División en validación y test (cada uno 50% del 22.22% ~10% del total)
    X_val, X_test, y_val, y_test = stratified_split_with_retry(X_temp, y_temp, test_size=0.5, min_ratio=min_ratio)

    # Normalización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Conversión a tensores de PyTorch
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Construcción del modelo
    input_size = X_train.shape[1]
    output_size = len(np.unique(y))
    model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)

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
        y_test_proba_calibrated = apply_histogram_binning(bin_edges, bin_true_prob, y_test_proba)
        y_pred = (y_test_proba_calibrated >= 0.5).astype(int)

    # Cálculo de métricas
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return precision, recall, accuracy, f1

# --------------------------
# Función para graficar las métricas
# --------------------------
def plot_metrics(metrics_list):
    # metrics_list es una lista de tuplas (precision, recall, accuracy, f1)
    precisions = [m[0] for m in metrics_list]
    recalls    = [m[1] for m in metrics_list]
    accuracies = [m[2] for m in metrics_list]
    f1s        = [m[3] for m in metrics_list]
    n = len(metrics_list)
    plt.rcParams["font.family"] = "Arial"

    plt.figure(figsize=(12, 8))
    x = np.arange(1, n+1)
    
    # Graficar cada métrica con puntos y su media con línea horizontal
    plt.plot(x, precisions, 'o-', label='Precision')
    plt.hlines(np.mean(precisions), 1, n, colors='blue', linestyles='dashed')
    
    plt.plot(x, recalls, 'o-', label='Recall')
    plt.hlines(np.mean(recalls), 1, n, colors='orange', linestyles='dashed')
    
    plt.plot(x, accuracies, 'o-', label='Accuracy')
    plt.hlines(np.mean(accuracies), 1, n, colors='green', linestyles='dashed')
    
    plt.plot(x, f1s, 'o-', label='F1')
    plt.hlines(np.mean(f1s), 1, n, colors='red', linestyles='dashed')
    
    plt.xlabel("Simulación")
    plt.ylabel("Métrica")
    plt.title("Métricas en cada simulación y sus medias")
    plt.legend()
    plt.grid(True)
    plt.xticks(x)
    plt.show()

# --------------------------
# Flujo principal
# --------------------------
def main():
    # Cargar dataset
    df = pd.read_csv(DATA_FILE)
    X, y, columns_df_original, label_encoders = depure_and_divide_data(df, verbose=False)

    metrics_list = []
    for sim in range(n_simulations):
        print(f"Simulación {sim+1}/{n_simulations}")
        precision, recall, accuracy, f1 = run_simulation(X, y, columns_df_original, label_encoders)
        print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  Accuracy: {accuracy:.4f}  F1: {f1:.4f}")
        metrics_list.append((precision, recall, accuracy, f1))
    
    # Graficar resultados
    plot_metrics(metrics_list)

if __name__ == '__main__':
    main()