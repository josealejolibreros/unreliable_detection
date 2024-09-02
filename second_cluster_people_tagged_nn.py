import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import warnings

warnings.filterwarnings("ignore")

#Settings
show_confusion_matrix_fig = False
TEST_FILE = 'suspicious_workers_noventi_dates_2024-09-02.csv' #place test file in test_data folder
verbose = False

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
filename_without_format = TEST_FILE.replace('.csv', '')
file_path_new_data_test = BASE_DIR + '/test_data/' + TEST_FILE


#will be deprecated replaced by merge more files
def merge_two_datasets_training(d1, d2, verbose = True):
    df1 = pd.read_csv(d1)
    df2 = pd.read_csv(d2)
    columns_df1 = df1.columns.tolist()
    columns_df2 = df2.columns.tolist()
    one_not_two = set(columns_df1).difference(columns_df2)
    two_not_one = set(columns_df2).difference(columns_df1)

    if verbose:
        print("two_not_one",two_not_one)
        print("two_not_one",two_not_one)

    frames = [df1, df2]
 
    result = pd.concat(frames)
    if verbose:
        print("len(df1.index)",len(df1.index))
        print("len(df2.index)",len(df2.index))
        print("len(result.index)",len(result.index))
    return result

def workunit_types(df, column_name = "workunit_id", verbose = False):
    keywords = ["DatesReha", "Amounts", "Dates", "Table", "Your Task"]
    
    # Create columns for each keyword and initialize with 0
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

def filter_confidences_prev_predictions(df, verbose = False):
    if verbose:
        print("old size without filtering based on confidence: ", len(df.index))
    if 'Confidence' in df.columns:
        #enterprise criterion: confidence of prev predictions greater than 0.95
        df = df[df['Confidence'] > 0.95]
        if verbose:
            print("new size by filtering based on confidence: ", len(df.index))

    if 'Prediction' in df.columns:
        df.rename(columns={'Prediction': 'class'}, inplace=True)

    if 'Confidence' in df.columns:
        df.drop(columns=['Confidence'], inplace=True)

    return df

def merge_datasets_training(directory, verbose=True):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    dataframes = []

    for file in all_files:
        df = pd.read_csv(file)
        dataframes.append(df)

    if verbose:
        print(f"Found {len(all_files)} files.")
        for idx, file in enumerate(all_files):
            print(f"File {idx+1}: {file}")

    if dataframes:
        for df_idx in range(len(dataframes)):
            dataframes[df_idx] = filter_confidences_prev_predictions(dataframes[df_idx], verbose=verbose)
        columns_all = [df.columns.tolist() for df in dataframes]
        columns_base = columns_all[0]
        for columns in columns_all[1:]:
            one_not_two = set(columns_base).difference(columns)
            two_not_one = set(columns).difference(columns_base)

            if verbose:
                print(f"Columns in base but not in current file: {one_not_two}")
                print(f"Columns in current file but not in base: {two_not_one}")

    #Concatenate all DataFrames
    result = pd.concat(dataframes, ignore_index=True)

    if verbose:
        for idx, df in enumerate(dataframes):
            print(f"Length of DataFrame {idx+1}: {len(df.index)}")
        print(f"Total length after concatenation: {len(result.index)}")

    return result

#training data location dir
df = merge_datasets_training(BASE_DIR + '/training_data/', verbose=verbose)



df = workunit_types(df, verbose=verbose)
df = df.drop(columns= ['workunit_id', 'recording_id', 'step_id'])

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

if verbose:
    print("Number columns train",len(columns_df_original))
    print(columns_df_original)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#conversion to pytorch
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU6() #0.897
        #self.relu = nn.SELU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(self.relu(out))
        out = self.dropout(self.fc2(out))
        out = self.dropout(self.relu(out))
        out = self.dropout(self.fc3(out))
        return out

input_size = X_train.shape[1]
hidden_size1 = 64
hidden_size2 = 32
output_size = len(pd.unique(y_train))
model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)

#loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

#test
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(y_test_tensor, predicted.numpy())
    if verbose:
        print("Accuracy:", accuracy)


conf_matrix = confusion_matrix(y_test_tensor, predicted.numpy())

class_labels = label_encoders['class'].classes_

if show_confusion_matrix_fig:
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()



######new data
new_data_df = pd.read_csv(file_path_new_data_test)
new_data_df = workunit_types(new_data_df, verbose=verbose)
recording_ids = new_data_df[['recording_id', 'step_id', 'workunit_id']]
new_data_df=new_data_df.drop(columns= ['median_time_diff', 'na_count', 'field_count', 'na_share', 'workflow_id', 'step_id', 'iqr_time_diff'], errors='ignore')


non_numeric_columns_new = new_data_df.select_dtypes(exclude=['number']).columns
new_data_df = new_data_df.drop(columns=non_numeric_columns_new)


columns_df_new = new_data_df.columns.tolist()
one_not_two = set(columns_df_original).difference(columns_df_new)
two_not_one = set(columns_df_new).difference(columns_df_original)
new_data_df = new_data_df.drop(columns = list(two_not_one))

if verbose:
    print(two_not_one)
    print("Number columns test",len(new_data_df.columns.to_list()))
    print(new_data_df.columns.to_list())

new_data_scaled = scaler.transform(new_data_df)




new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    #outputs_new_data = model(new_data_tensor)
    #_, predicted_new_data = torch.max(outputs_new_data, 1)

    
    outputs_new_data = model(new_data_tensor)
    softmax_probs = nn.functional.softmax(outputs_new_data, dim=1)
    _, predicted_new_data = torch.max(outputs_new_data, 1)
    
    #predictions+confidence
    results_df = pd.DataFrame({
        'Prediction': label_encoders['class'].inverse_transform(predicted_new_data.numpy()),#predicted_new_data.numpy(),
        'Confidence': torch.max(softmax_probs, dim=1)[0].numpy()
    })
    
new_data_df = pd.concat([new_data_df,results_df,recording_ids], axis=1)




new_data_df_095 = new_data_df[new_data_df['Confidence']>=0.95] 
new_data_df_unrel = new_data_df[new_data_df['Prediction']=='Unreliable'] 


new_data_df.to_csv(BASE_DIR + '/predictions/' + filename_without_format + '_predictions_ALL.csv', index=False)
new_data_df_095.to_csv(BASE_DIR + '/predictions/' + filename_without_format + '_predictions_g095.csv', index=False)
new_data_df_unrel.to_csv(BASE_DIR + '/predictions/' + filename_without_format + '_predictions_unrel.csv', index=False)
