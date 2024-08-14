import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

file_path = '/Users/joselibreros/Documents/implementations/data_for_t/aggregator/training_data/suspicious_workers_noventi_dates_per_workunit_all_agg.csv'
df = pd.read_csv(file_path)

label_encoders = {}
non_numeric_columns = df.select_dtypes(exclude=['number']).columns
for column in non_numeric_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

y = df['class'].values

df = df.drop(columns=['class'])

df = df.dropna(axis=1)

X = df.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


class ReLUX(nn.Module):
    def __init__(self, value, max_value: float=1.0):
        super(ReLUX, self).__init__()
        self.max_value = float(max_value)
        self.scale     = value/self.max_value

    def forward(self, x):
        return nn.functional.relu6(x * self.scale) / (self.scale)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activation):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        """
        out = self.fc1(x)
        out = self.dropout(self.activation(out))
        out = self.dropout(self.fc2(out))
        out = self.dropout(self.activation(out))
        out = self.dropout(self.fc3(out))
        
        return out

activation_functions = {
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


num_simulations = 10
num_epochs = 100

mean_accuracies = {}

for activation_name, activation_function in activation_functions.items():
    print(f"Running simulations for {activation_name} activation function...")
    accuracies = []

    for _ in range(num_simulations):
        input_size = X_train.shape[1]
        hidden_size1 = 64
        hidden_size2 = 32
        output_size = len(pd.unique(y_train))
        model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size, activation_function)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(y_test_tensor, predicted.numpy())
            accuracies.append(accuracy)

    mean_accuracy = sum(accuracies) / num_simulations
    mean_accuracies[activation_name] = mean_accuracy

plt.figure(figsize=(10, 6))
plt.bar(mean_accuracies.keys(), mean_accuracies.values(), color='skyblue')
plt.xlabel('Activation Function')
plt.ylabel('Mean Accuracy')
plt.title('Mean Accuracy of 10 Simulations for Each Activation Function')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation='vertical')

plt.show()
