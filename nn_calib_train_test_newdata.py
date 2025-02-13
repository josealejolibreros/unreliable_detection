import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
import warnings
from sklearn.calibration import calibration_curve
import numpy as np

warnings.filterwarnings("ignore")

# Settings
show_confusion_matrix_fig = False
TEST_FILE = '2024-11-18_aggregation_all.csv' # place test file in test_data folder
verbose = False

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
filename_without_format = TEST_FILE.replace('.csv', '')
BASE_TEST = BASE_DIR + '/test_data/'
PATH_MODEL = BASE_DIR + '/saved_models/model.model'
file_path_new_data_test = BASE_TEST + TEST_FILE

partial = True
resume_training = True
training_from_scratch = False


# Function to merge datasets from a directory, but only one file
def merge_datasets_training(directory, verbose=verbose):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    
    if len(all_files) > 0:
        df = pd.read_csv(all_files[0])  # Just read one file
        
        if 'class' in df.columns.tolist():
            df['class'].replace('Suspicious', 'Unreliable', inplace=True)
        elif 'Prediction' in df.columns.tolist():
            df['Prediction'].replace('Suspicious', 'Unreliable', inplace=True)
    else:
        raise FileNotFoundError("No CSV files found in directory.")

    if verbose:
        print(f"Found {len(all_files)} files.")
        print(f"File 1: {all_files[0]}")
    
    return df


def workunit_types(df, column_name="workunit_id", verbose=verbose):
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
            print(keyword, ":", str(df[keyword].value_counts()[1]))
    
    return df

columns_df_original = []
def depure_and_divide_data(df, verbose=verbose):
    df = workunit_types(df, verbose=verbose)
    df = df.drop(columns=['worker_id', 'workunit_id', 'start_datetime', 'source', 'session_id', 'reason', 'recording_id', 'job_id', 'step', 'Prediction', 'Confidence'], errors='ignore')

    label_encoders = {}
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    for column in non_numeric_columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

    y = df['class'].values
    df = df.drop(columns=['class'])
    df = df.dropna(axis=1)

    X = df.values
    columns_df_original = df.columns.tolist()

    if True:
        print("Number columns train", len(columns_df_original))
        print(columns_df_original)

    return X, y, columns_df_original, label_encoders


# Histogram binning implementation for calibration
def histogram_binning_calibration(y_true, y_pred_proba, bins=10):
    """
    Calibrar las probabilidades utilizando Histogram Binning.
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

    y_pred_calibrated = np.zeros_like(y_pred_proba)
    for b in range(1, bins + 1):
        bin_mask = (bin_indices == b)
        y_pred_calibrated[bin_mask] = bin_true_prob[b - 1]

    bin_weight = bin_count / len(y_true)
    
    return y_pred_calibrated, bin_edges, bin_true_prob, bin_weight


def apply_histogram_binning(bin_edges, bin_true_prob, y_pred_proba_new):
    bin_indices_new = np.digitize(y_pred_proba_new, bins=bin_edges, right=True)

    y_pred_calibrated_new = np.zeros_like(y_pred_proba_new)
    for b in range(1, len(bin_edges)):
        bin_mask_new = (bin_indices_new == b)
        y_pred_calibrated_new[bin_mask_new] = bin_true_prob[b - 1]

    return y_pred_calibrated_new


def plot_reliability_diagram(y_true, y_proba, y_proba_calibrated, bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=bins, strategy='uniform')
    prob_true_calibrated, prob_pred_calibrated = calibration_curve(y_true, y_proba_calibrated, n_bins=bins, strategy='uniform')

    bin_edges = np.linspace(0, 1, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_indices = np.digitize(y_proba, bins=bin_edges, right=True)
    bin_weight_before = np.array([np.sum(bin_indices == i) for i in range(1, bins + 1)]) / len(y_true)

    bin_indices_calibrated = np.digitize(y_proba_calibrated, bins=bin_edges, right=True)
    bin_weight_after = np.array([np.sum(bin_indices_calibrated == i) for i in range(1, bins + 1)]) / len(y_true)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(prob_pred, prob_true, marker='o', linestyle='-', color='blue', label='Frecuencia Positiva')
    plt.plot(prob_pred, prob_pred, marker='o', linestyle='--', color='red', label='Confianza Promedio')
    plt.bar(bin_centers - 0.05, bin_weight_before, width=0.1, alpha=0.3, color='gray', label='Frecuencia de Datos Antes de Calibrar')
    plt.xlabel('Probabilidad Predicha')
    plt.ylabel('Frecuencia')
    plt.title('Antes de Calibrar')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(prob_pred_calibrated, prob_true_calibrated, marker='o', linestyle='-', color='blue', label='Frecuencia Positiva')
    plt.plot(prob_pred_calibrated, prob_pred_calibrated, marker='o', linestyle='--', color='red', label='Confianza Promedio')
    plt.bar(bin_centers - 0.05, bin_weight_after, width=0.1, alpha=0.3, color='gray', label='Frecuencia de Datos Después de Calibrar')
    plt.xlabel('Probabilidad Predicha')
    plt.ylabel('Frecuencia')
    plt.title('Después de Calibrar')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


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
        out = self.dropout(self.fc2(out))
        out = self.dropout(self.relu(out))
        out = self.fc3(out)
        return out


# Training data location dir
df = merge_datasets_training(BASE_DIR + '/training_data_compiled/', verbose=verbose)

X, y, columns_df_original, label_encoders = depure_and_divide_data(df)

# Stratified K-Fold cross-validation
kf = StratifiedKFold(n_splits=5)
accuracies = []
calibration_results = []

for train_idx, test_idx in kf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Conversion to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Model
    input_size = X_train.shape[1]
    hidden_size1 = 32
    hidden_size2 = 16
    output_size = len(np.unique(y_train))
    model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)
    
    # Class weights (for reliable and unreliable classes)
    class_weights = torch.tensor([1.4, 3.52], dtype=torch.float32) #necessary for giving more weight for unreliable
    class_weights = torch.tensor([0.5, 10], dtype=torch.float32) #necessary for giving more weight for unreliable
    criterion = nn.CrossEntropyLoss(class_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    ##New: loading state
    if partial:
        pass
        #model.load_state_dict(torch.load(PATH_MODEL))
        #optimizer.load_state_dict(PATH_MODEL)

    # Training
    num_epochs = 500 #100 is too low, 400 too much, after tests 150 is ok
    train_losses = []
    test_losses = [] 

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        train_losses.append(loss.item())  
        loss.backward()
        optimizer.step()

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            test_loss = criterion(outputs, y_test_tensor)
            test_losses.append(test_loss.item())  # Guardar la pérdida de prueba

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probabilities = torch.softmax(outputs, dim=1).numpy()
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test_tensor, predicted.numpy())
        accuracies.append(accuracy)

        y_pred_proba = probabilities[:, 1]  # For binary classification

        # Calibration using histogram binning
        y_pred_calibrated, bin_edges, bin_true_prob, bin_weight = histogram_binning_calibration(y_test, y_pred_proba)

        # Store calibration results
        calibration_results.append((y_test, y_pred_proba, y_pred_calibrated))

        # Reliability diagram
        #plot_reliability_diagram(y_test, y_pred_proba, y_pred_calibrated)

# Print overall performance
print(f"Average Accuracy: {accuracies}, mean: {np.mean(accuracies):.4f} std: {np.std(accuracies):.4f}")

plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss per Epoch")
plt.legend()
plt.grid()
plt.show()
torch.save(model.state_dict(), PATH_MODEL)


all_files_test_newdata = [(os.path.join(BASE_TEST, f), f) for f in os.listdir(BASE_TEST) if f.endswith('.csv')]
print("with new data - to inference")
for path_filename_test_newdata, filename_test_newdata in all_files_test_newdata:
    filename_without_format = filename_test_newdata.replace('.csv', '')

    ##################################
    #  With new data - to inference  #
    ##################################
    

    ######new data
    new_data_df = pd.read_csv(path_filename_test_newdata)

    classes_available = False
    if 'class' in new_data_df.columns.to_list():        
        classes_available = True
        classes_test = new_data_df['class']
    print(filename_test_newdata, "classes_available", classes_available)

    new_data_df = workunit_types(new_data_df, verbose=verbose)
    recording_ids = new_data_df[['recording_id', 'step_id', 'workunit_id']]
    new_data_df=new_data_df.drop(columns=['worker_id', 'workunit_id', 'start_datetime', 'source', 'session_id', 'reason', 'recording_id', 'job_id', 'step', 'class'], errors='ignore')
    #new_data_df=new_data_df.drop(columns= ['median_time_diff', 'na_count', 'field_count', 'na_share', 'workflow_id', 'step_id', 'iqr_time_diff'], errors='ignore')


    non_numeric_columns_new = new_data_df.select_dtypes(exclude=['number']).columns
    new_data_df = new_data_df.drop(columns=non_numeric_columns_new)


    columns_df_new = new_data_df.columns.tolist()
    one_not_two = set(columns_df_original).difference(columns_df_new)
    two_not_one = set(columns_df_new).difference(columns_df_original)
    new_data_df = new_data_df.drop(columns = list(two_not_one))

    if True:
        print(two_not_one)
        print("Number columns test",len(new_data_df.columns.to_list()))
        #print(new_data_df.columns.to_list())
        print(one_not_two)

    new_data_scaled = scaler.transform(new_data_df)


    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

    # Ensure model is in evaluation mode
    model.eval()


    # Ensure no gradients are calculated during inference
    with torch.no_grad():
        # Forward pass through the model with new data
        outputs_new_data = model(new_data_tensor)
        
        # Get softmax probabilities from the logits
        softmax_probs = nn.functional.softmax(outputs_new_data, dim=1)
        
        
        # Apply histogram binning for calibration on the softmax probabilities
        y_pred_calibrated_new = apply_histogram_binning(bin_edges, bin_true_prob, softmax_probs.numpy())
        
        # Obtain the predicted class based on calibrated probabilities
        y_pred_class_new = np.argmax(y_pred_calibrated_new, axis=1)
        
        # Prepare predictions and confidence for dataframe
        results_df = pd.DataFrame({
            # Convert predicted class indices to class labels
            'Prediction': label_encoders['class'].inverse_transform(y_pred_class_new),
            
            # Maximum confidence per sample from the calibrated probabilities
            'Confidence': np.max(y_pred_calibrated_new, axis=1),  # Get max calibrated probability for confidence

            'confidence_without_cal': torch.max(softmax_probs, dim=1)[0].numpy()

        })
        if classes_available:
            results_df = pd.concat([results_df,classes_test], axis=1)
        
        # Concatenate with the original dataframe (ensure dimensions match)
        new_data_df = pd.concat([new_data_df.reset_index(drop=True), results_df, recording_ids.reset_index(drop=True)], axis=1)
        
        # Filter data based on confidence threshold
        new_data_df_095 = new_data_df[new_data_df['Confidence'] >= 0.95]
        
        # Filter rows where prediction is 'Unreliable'
        new_data_df_unrel = new_data_df[new_data_df['Prediction'] == 'Unreliable']


    new_data_df.to_csv(BASE_DIR + '/predictions/' + filename_without_format + '_predictions_ALL__withcalibration.csv', index=False)
    #new_data_df_095.to_csv(BASE_DIR + '/predictions/' + filename_without_format + '_predictions_g095__withcalibration.csv', index=False)
    #new_data_df_unrel.to_csv(BASE_DIR + '/predictions/' + filename_without_format + '_predictions_unrel__withcalibration.csv', index=False)


