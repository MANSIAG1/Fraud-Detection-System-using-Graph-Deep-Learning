# this code is used for training and testing Neural Networks on the extracted network features
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

torch.manual_seed(7)
smote = SMOTE(sampling_strategy=0.01, random_state=7)
rus = RandomUnderSampler(sampling_strategy=0.1, random_state=7)
scaler = MinMaxScaler()

for i in range(1, 6):
    locals()[f'features_{i}'] = pd.read_csv(f"/mnt/redpro/home/aid23001/Features/features_{i}.csv", index_col=0)
    locals()[f'true_labels_{i}'] = locals()[f'features_{i}']['Malicious']
    locals()[f'features_{i}'] = locals()[f'features_{i}'].drop(['Malicious'], axis=1)
    locals()[f'features_{i}'] = locals()[f'features_{i}'].drop(['Node'], axis=1)
    locals()[f'features_{i}'] = scaler.fit_transform(locals()[f'features_{i}'])
    z, y = smote.fit_resample(locals()[f'features_{i}'], locals()[f'true_labels_{i}'].to_numpy()) #Oversampling first, Undersampling later
    z, y = rus.fit_resample(z, y)
    locals()[f'train_dataset_{i}'] = TensorDataset(torch.tensor(z), torch.tensor(y))
    locals()[f'train_loader_{i}'] = DataLoader(locals()[f'train_dataset_{i}'], batch_size=128, shuffle=True)
print('Train datasets are completed')

test_features = pd.read_csv('/mnt/redpro/home/aid23001/Features/features_6.csv', index_col=0)
labels = test_features['Malicious']
test_features = test_features.drop(['Malicious'], axis=1)
test_features = test_features.drop(['Node'], axis=1)
z_6 = scaler.transform(test_features.to_numpy())
test_dataset = TensorDataset(torch.tensor(z_6), torch.tensor(labels))
del z_6
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('The used device is', device)


def create_model(num_layers, num_nodes_per_layer, input_size, output_size, activation_function):
    layers = []
    in_features = input_size

    for _ in range(num_layers):
        layers.append(nn.Linear(in_features, num_nodes_per_layer))
        layers.append(activation_function)  #activation_funtion must be in form nn.function()
        #         layers.append(nn.Dropout(0.5))
        in_features = num_nodes_per_layer

    layers.append(nn.Linear(in_features, output_size))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def train_model(model, optimizer, criterion, train_loader, weight, num_epochs=50):
    for epoch in tqdm(range(num_epochs + 1)):
        model.train()  # Set the model to training mode
        for inputs, labels in train_loader:
            if labels.sum() == 0:
                continue
            inputs, labels = inputs.to(device), labels.to(device).float()
            weights = labels * weight + 0.01  #add a small value (e.g., 0.01) to avoid zero weights.
            criterion.weight = weights
            optimizer.zero_grad()
            outputs = model(inputs.float()).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def evaluate_model(model, test_loader, device, threshold):
    model.eval()  # Set the model to evaluation mode
    propabilities = []
    predictions = []
    with torch.no_grad():  # No need to track gradients for testing
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float()).squeeze()  # Squeeze to remove the extra dimension
            pred = (outputs > threshold).float()
            prob = outputs
            propabilities.append(prob)
            predictions.append(pred)
        propabilities = torch.cat(propabilities, dim=0).to(device)
        predictions = torch.cat(predictions, dim=0).to(device)
    # Calculate metrics
    return propabilities, predictions


criterion = nn.BCELoss()
threshold = 0.5
weight = 10.0
results = []
for weight in [10.0, 50.0, 100.0]:
    for activation_function in [nn.Tanh(), nn.ReLU()]:
        for nodes_per_layer in [16 ,32, 64]:
            for layers in range(2, 5):
    # layers, nodes_per_layer, activation_function = 3, 128, nn.Tanh()
                model = create_model(layers, nodes_per_layer, 7, 1, activation_function)
                model.to(device)
    # Define optimizer
                optimizer = optim.AdamW(model.parameters(), lr=1e-3)
                for train_loader in [train_loader_1, train_loader_2, train_loader_3, train_loader_4, train_loader_5]:
                    train_model(model, optimizer, criterion, train_loader, weight, 99)
                print('Training completed')
                print('Testing starts')
                propabilities, predictions = evaluate_model(model, test_loader, device, threshold)
                results.append({
                        'sampling': 'SMOTE',
                        'layers': layers,
                        'nodes_per_layer': nodes_per_layer,
                        'activation_function': activation_function.__class__.__name__,
                        'optimizer': optimizer.__class__.__name__,
                        'roc_auc': roc_auc_score(labels, propabilities.cpu()),
                        'f1_score_binary': f1_score(labels, predictions.cpu(), average='binary',zero_division=0.0),
                        'f1_score_micro': f1_score(labels, predictions.cpu(), average='micro', zero_division=0.0),
                        'f1_score_macro': f1_score(labels, predictions.cpu(), average='macro', zero_division=0.0),
                        'confusion_matrix': confusion_matrix(labels, predictions.cpu())})
                if roc_auc_score(labels, propabilities.cpu()) > 0.85:
                    torch.save(model.state_dict(), f'/mnt/redpro/home/aid23001/Models_SMOTE_RUS/model_NN_{activation_function}_{layers}_{nodes_per_layer}_{weight}.pth')
                    torch.save(predictions.cpu(), f'/mnt/redpro/home/aid23001/Models_SMOTE_RUS/predictions_NN_{activation_function}_{layers}_{nodes_per_layer}_{weight}.pth')
                    torch.save(propabilities.cpu(), f'/mnt/redpro/home/aid23001/Models_SMOTE_RUS/probabilities_NN_{activation_function}_{layers}_{nodes_per_layer}_{weight}.pth')
    results_df = pd.DataFrame(results)
    results_df.to_csv('/mnt/redpro/home/aid23001/Models_SMOTE_RUS/features_NN_results.csv')
    print(results_df)
print('All done')