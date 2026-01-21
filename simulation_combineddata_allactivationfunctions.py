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
import torch.multiprocessing as mp

# =====================
# Configuración de GPU
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# --------------------------
# Parámetros y configuración
# --------------------------
DATA_FILE = './combined_dataset/combined_checked_with_time_t.csv'  # Ruta al CSV
n_simulations = 50
data_use_ratio = 0.9
train_ratio_used = 0.7778
val_ratio_used = 0.5
num_epochs = 150
learning_rate = 0.0001

hidden_size1 = 32
hidden_size2 = 16

torch.manual_seed(0)

INCLUDE_TIME_IN_SIMULATION = True
USE_RNN = False

OUTPUT_FILENAME = 'results_simulations_all'
#OUTPUT_FILENAME = 'results_simulations_all_with_rnn'
#OUTPUT_FILENAME = 'results_simulations_without_date_with_rnn'
#OUTPUT_FILENAME = 'results_simulations_without_date_without_rnn'
hidden_size_lstm = 64
hidden_size_fc1 = 32
hidden_size_fc2 = 16

class ReLUX(nn.Module):
    def __init__(self, value, max_value: float=1.0):
        super(ReLUX, self).__init__()
        self.max_value = float(max_value)
        self.scale     = value/self.max_value

    def forward(self, x):
        return nn.functional.relu6(x * self.scale) / (self.scale)

ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "softmax": nn.Softmax(dim=1),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "softsign": nn.Softsign(),
    "softmin": nn.Softmin(dim=1),
    #"softmax2d": nn.Softmax2d(),
    "logsigmoid": nn.LogSigmoid(),
    "logsoftmax": nn.LogSoftmax(dim=1),
    "hardswish": nn.Hardswish(),
    "relu6": nn.ReLU6(),
    "tanhshrink": nn.Tanhshrink(),
    "softshrink": nn.Softshrink(),
    "hardshrink": nn.Hardshrink(),
    "hardtanh": nn.Hardtanh(),
    #"gumbel_softmax": nn.GumbelSoftmax(),
    #"adaptive_log_softmax": nn.AdaptiveLogSoftmaxWithLoss(),
    "threshold": nn.Threshold(0.1, 20),
    "silu": nn.SiLU(),
    "hardsigmoid": nn.Hardsigmoid(),
    "mish": nn.Mish(),
    #"swish": nn.Swish(),
    #"glu": nn.GLU(),
    "gelu": nn.GELU(),
    "celu": nn.CELU(),
    "rrelu": nn.RReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "p_relu": nn.PReLU(),
    "seLU": nn.SELU(),
    "relu1": ReLUX(1,1),
    "relu3": ReLUX(3,3),
    "relu5": ReLUX(5,5),
    "relu10": ReLUX(10,10),
    "relu1-3": ReLUX(1,3),
    #"batch_norm": nn.BatchNorm1d(input_size),
    #"layer_norm": nn.LayerNorm(normalized_shape=input_size),
    #"instance_norm": nn.InstanceNorm1d(input_size),
    #"group_norm": nn.GroupNorm(num_groups=4, num_channels=input_size),
    #"local_response_norm": nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0)
}

# --------------------------
# Funciones de preprocesado
# --------------------------
def preprocess_datetime(df):
    for col in ['start_datetime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='ignore')
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
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
    return df

def depure_and_divide_data(df, verbose=False, column_name_to_delete=""):
    if INCLUDE_TIME_IN_SIMULATION:
        df = preprocess_datetime(df.copy())
    if 'class' in df.columns.tolist():
        df['class'].replace('Suspicious', 'Unreliable', inplace=True)
    elif 'Prediction' in df.columns.tolist():
        df['Prediction'].replace('Suspicious', 'Unreliable', inplace=True)
    df = workunit_types(df, verbose=verbose)
    drop_cols = ['worker_id', 'workunit_id', 'start_datetime', 'source', 'session_id', 
                 'reason', 'recording_id', 'job_id', 'step', 'Prediction', 'Confidence',
                 'date', 'merge', 'is_same_recording_id', 'is_same_focus_count', 
                 'is_samemutation_count', 'is_same_workunit_id', 'verbose_text', 'step_id.1',
                 'pattern', 'Prediction', 'Confidence', 'confidence_without_cal']
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df.drop(columns=[column_name_to_delete], errors='ignore')
    label_encoders = {}
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    for column in non_numeric_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    y = df['class'].values
    df = df.drop(columns=['class'])
    df = df.dropna(axis=1)
    X = df.values
    columns_df_original = df.columns.tolist()
    #print(columns_df_original)
    return X, y, columns_df_original, label_encoders

# --------------------------
# Calibración
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

# --------------------------
# Modelos
# --------------------------
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activation):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(p=0.001)
    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(self.activation(out))
        out = self.fc2(out)
        out = self.dropout(self.activation(out))
        out = self.fc3(out)
        return out

class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size_lstm, hidden_size_fc1, hidden_size_fc2, output_size, activation):
        super(TimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size_lstm, batch_first=True)
        self.fc1 = nn.Linear(hidden_size_lstm, hidden_size_fc1)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_size_fc1, hidden_size_fc2)
        self.fc3 = nn.Linear(hidden_size_fc2, output_size)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.dropout(self.activation(out))
        out = self.dropout(self.fc2(out))
        out = self.dropout(self.activation(out))
        out = self.fc3(out)
        return out

# --------------------------
# Balanceo y splits
# --------------------------
def balance_classes(X_train, X_test, y_train, y_test, random_state=42):
    train_df = pd.DataFrame(X_train)
    train_df['label'] = y_train
    test_df = pd.DataFrame(X_test)
    test_df['label'] = y_test
    majority_class_train = train_df['label'].value_counts().idxmax()
    minority_class_train = train_df['label'].value_counts().idxmin()
    majority_class_test = test_df['label'].value_counts().idxmax()
    minority_class_test = test_df['label'].value_counts().idxmin()
    n_minority_train = train_df['label'].value_counts()[minority_class_train]
    n_minority_test = test_df['label'].value_counts()[minority_class_test]
    train_majority_df = train_df[train_df['label'] == majority_class_train].sample(n=n_minority_train, random_state=random_state)
    train_minority_df = train_df[train_df['label'] == minority_class_train]
    test_majority_df = test_df[test_df['label'] == majority_class_test].sample(n=n_minority_test, random_state=random_state)
    test_minority_df = test_df[test_df['label'] == minority_class_test]
    balanced_train_df = pd.concat([train_majority_df, train_minority_df]).sample(frac=1, random_state=random_state)
    balanced_test_df = pd.concat([test_majority_df, test_minority_df]).sample(frac=1, random_state=random_state)
    X_train_balanced = balanced_train_df.drop(columns=['label']).values
    y_train_balanced = balanced_train_df['label'].values
    X_test_balanced = balanced_test_df.drop(columns=['label']).values
    y_test_balanced = balanced_test_df['label'].values
    return X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced

def stratified_split_with_retry(X, y, test_size, min_ratio, max_retries=1000):
    for _ in range(max_retries):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=np.random.randint(10000)
        )
        unique_classes, counts = np.unique(y_train, return_counts=True)
        min_required = np.ceil(len(y_train) * min_ratio).astype(int)
        if np.all(counts >= min_required):
            return X_train, X_test, y_train, y_test
    raise ValueError(f"No split válido tras {max_retries} intentos.")

def apply_smote_to_minority(X_train, y_train):
    unique_classes, counts = np.unique(y_train, return_counts=True)
    min_class = unique_classes[np.argmin(counts)]
    smote = SMOTE(sampling_strategy={min_class: int(np.median(counts))}, random_state=42)
    return smote.fit_resample(X_train, y_train)

# --------------------------
# Entrenamiento
# --------------------------
def run_simulation(X, y, columns_df_original, label_encoders, activation_name, activation_function):
    min_ratio = 0.23
    X_used, _, y_used, _ = stratified_split_with_retry(X, y, test_size=1 - data_use_ratio, min_ratio=min_ratio)
    X_train, X_temp, y_train, y_temp = stratified_split_with_retry(X_used, y_used, test_size=1 - train_ratio_used, min_ratio=min_ratio)
    X_train, X_temp, y_train, y_temp = balance_classes(X_train, X_temp, y_train, y_temp)
    X_train, y_train = apply_smote_to_minority(X_train, y_train)
    X_val, X_test, y_val, y_test = stratified_split_with_retry(X_temp, y_temp, test_size=0.5, min_ratio=min_ratio)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    input_size = X_train.shape[1]
    output_size = len(np.unique(y))
    if USE_RNN:
        X_train_tensor = X_train_tensor.unsqueeze(1)
        X_val_tensor = X_val_tensor.unsqueeze(1)
        X_test_tensor = X_test_tensor.unsqueeze(1)
        model = TimeSeriesModel(input_size, hidden_size_lstm, hidden_size_fc1, hidden_size_fc2, output_size, activation_function)
    else:
        model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size, activation_function)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        outputs_test = model(X_test_tensor)
        probabilities_test = torch.softmax(outputs_test, dim=1).cpu().numpy()
        y_test_proba = probabilities_test[:, 1]
        y_pred = (y_test_proba >= 0.5).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    return precision, recall, accuracy, f1

# --------------------------
# Funciones gráficas
# --------------------------
def compress_metrics(metrics_list):
    precisions = np.array([m[0] for m in metrics_list])
    recalls = np.array([m[1] for m in metrics_list])
    accuracies = np.array([m[2] for m in metrics_list])
    f1s = np.array([m[3] for m in metrics_list])
    return len(metrics_list), np.mean(precisions[precisions > 0.05]), np.mean(recalls[recalls > 0.05]), np.mean(accuracies[accuracies > 0.05]), np.mean(f1s[f1s > 0.05])

def plot_metrics(metrics_list, column_name=""):
    precisions = np.array([m[0] for m in metrics_list])
    recalls = np.array([m[1] for m in metrics_list])
    accuracies = np.array([m[2] for m in metrics_list])
    f1s = np.array([m[3] for m in metrics_list])
    mean_precision = np.mean(precisions[precisions > 0.05])
    mean_recalls = np.mean(recalls[recalls > 0.05])
    mean_accuracies = np.mean(accuracies[accuracies > 0.05])
    mean_f1s = np.mean(f1s[f1s > 0.05])
    n = len(metrics_list)
    df = pd.DataFrame({"Simulation": np.arange(1, n+1), "Precision": precisions, "Recall": recalls, "Accuracy": accuracies, "F1 Score": f1s})
    df.to_csv(f'./plots_deleting_columns/delete_{column_name}.csv', index=False, mode="w")
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
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./plots_deleting_columns/delete_{column_name}.svg', format='svg', dpi=1200)

def plot_means(names_deleted, means_precision, means_recall, means_accuracy, means_f1, ns):
    names_deleted = np.array(names_deleted)
    means_precision = np.array(means_precision)
    means_recall = np.array(means_recall)
    means_accuracy = np.array(means_accuracy)
    means_f1 = np.array(means_f1)
    sorted_indices = np.argsort(-means_recall)
    names_deleted = names_deleted[sorted_indices]
    means_precision = means_precision[sorted_indices]
    means_recall = means_recall[sorted_indices]
    means_accuracy = means_accuracy[sorted_indices]
    means_f1 = means_f1[sorted_indices]
    df = pd.DataFrame({"Deleted Variable": names_deleted, "Mean Precision": means_precision, "Mean Recall": means_recall, "Mean Accuracy": means_accuracy, "Mean F1 Score": means_f1})
    df.to_csv(f'./plots_deleting_columns/delete_all.csv', index=False, mode="w")
    fig, ax = plt.subplots(figsize=(10, 12))
    plt.subplots_adjust(left=0.3)
    y_positions = np.arange(len(names_deleted)) * 1.5
    ax.scatter(means_precision, y_positions, label='Precision', marker='o', color='blue')
    ax.scatter(means_recall, y_positions, label='Recall', marker='s', color='red')
    ax.scatter(means_accuracy, y_positions, label='Accuracy', marker='^', color='green')
    ax.scatter(means_f1, y_positions, label='F1 Score', marker='D', color='purple')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names_deleted, fontsize=12)
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'./plots_deleting_columns/delete_all.svg', format='svg', dpi=1200)

# --------------------------
# Main
# --------------------------
def process_column(column_name_to_delete, df, n_simulations, activation_name, activation_function):
    if column_name_to_delete in ["class", "worker_id", "workunit_id", "step_i.1", "pattern", "merge", "Confidence", 
                                 "Prediction", "step_id.1", "is_same_focus_count", "is_same_workunit_id",
                                 "confidence_without_cal", "is_same_mouse_move_count", "is_same_recording_id",
                                 "is_same_mutation_count", "recording_id", "date"]:
        return None
    X, y, columns_df_original, label_encoders = depure_and_divide_data(df, verbose=False, column_name_to_delete=column_name_to_delete)
    metrics_list = []
    for sim in range(n_simulations):
        print(f"Simulation # {sim+1}/{n_simulations} for {column_name_to_delete} for {activation_name}")
        precision, recall, accuracy, f1 = run_simulation(X, y, columns_df_original, label_encoders, activation_name, activation_function)
        metrics_list.append((precision, recall, accuracy, f1))
    n, mean_precision, mean_recall, mean_accuracy, mean_f1 = compress_metrics(metrics_list)
    metrics_list.append((mean_precision, mean_recall, mean_accuracy, mean_f1))
    header = 'precision, recall, accuracy, f1'
    np.savetxt(OUTPUT_FILENAME+"_"+activation_name+".csv", metrics_list, delimiter=',', header=header, comments='')
    #plot_metrics(metrics_list, column_name_to_delete)
    return (column_name_to_delete, n, mean_precision, mean_recall, mean_accuracy, mean_f1)

def main():
    df = pd.read_csv(DATA_FILE)
    df.drop(columns=["worker_id", "workunit_id", "step_i.1", "pattern", "merge", "Confidence", 
                                 "Prediction", "step_id.1", "is_same_focus_count", "is_same_workunit_id",
                                 "confidence_without_cal", "is_same_mouse_move_count", "is_same_recording_id",
                                 "is_same_mutation_count", "date", "recording_id", "verbose_text", "step_id"], errors="ignore")
    results = []

    for activation_name, activation_function in ACTIVATION_FUNCTIONS.items():
        result = process_column("All included", df, n_simulations, activation_name, activation_function)
        if result:
            results.append(result)

    
    
    #Commented for simulating with all the features
    ##for col in df.columns.values:
    ##    result = process_column(col, df, n_simulations)
    ##    if result:
    ##        results.append(result)
    
    
    # Extraer métricas de los resultados
    ##names_deleted, ns, means_precision, means_recall, means_accuracy, means_f1 = zip(*results)
    
    # Graficar resultados agregados
    ##plot_means(names_deleted, means_precision, means_recall, means_accuracy, means_f1, ns)

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()
    