import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import product_catalog
import random
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import train_test_split
from adabelief_pytorch import AdaBelief

np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def introduce_fraud(data):
    products = product_catalog.get_product_catalog()
    data['price'] = None
    data['fraud'] = False
    data.rename(columns={'product id': 'product_id'}, inplace=True)
    for row in data.itertuples():
        product = products.loc[products['id'] == row.product_id]
        data.loc[row.Index, 'price'] = product['price'].values[0]

    data1 = data.copy()
    # get a list of distinct values in the 'purchase id' column
    distinct_purchase_ids = data['purchase id'].unique()

    # randomly shuffle the list of distinct values
    random.shuffle(distinct_purchase_ids)

    # take the first half of the shuffled list
    half_purchase_ids = distinct_purchase_ids[:len(distinct_purchase_ids)//2]

    # filter the original dataframe to only include rows with a 'purchase id' value
    # in the first half of the shuffled list
    filtered_df = data[data['purchase id'].isin(half_purchase_ids)]

    fraud = filtered_df['purchase id'].unique()

    # for all fraudulent purchases, change the price of the product to the lowest price of the same sub category
    # and adjust the product id
    for i in fraud:
        # extract the fraud purchase
        change = filtered_df.loc[filtered_df['purchase id'] == i]
        # selects a random number of the most expensive products in the purchase
        sorted = change.sort_values('price', ascending=False)
        sorted = sorted.head(n=1)  # sorted.head(int(len(sorted) * random.random()))
        sorted = sorted['product_id'].unique()
        # for each product the banana trick is used on, change the price to the lowest price of the same sub category
        # and change the product id
        for j in sorted:
            aux = products.loc[products['id'] == j]
            aux = aux['sub category'].values[0]
            aux = products.loc[products['sub category'] == aux]
            aux = aux.sort_values('price')
            aux = aux.iloc[0, :]
            data.loc[(data['purchase id'] == i) & (data['product_id'] == j), 'price'] = aux['price']
            data.loc[(data['purchase id'] == i) & (data['product_id'] == j), 'product_id'] = aux['id']
            data.loc[data['purchase id'] == i, 'fraud'] = True

    return data


def data_transform(data):
    products = product_catalog.get_product_catalog()
    data['price'] = None
    data['fraud'] = False
    data.rename(columns={'product id': 'product_id'}, inplace=True)
    for row in data.itertuples():
        product = products.loc[products['id'] == row.product_id]
        data.loc[row.Index, 'price'] = product['price'].values[0]
    return data


# Define the new LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(2, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_time, x_price):
        x = torch.cat((x_time.unsqueeze(-1), x_price.unsqueeze(-1)), dim=-1)
        x = x.view(x.size(0), x.size(1), -1)  # Flatten the last two dimensions
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


def preprocess_data(training_data):
    # Load and preprocess data
    data = introduce_fraud(training_data)
    data['fraud'] = data['fraud'].astype(int)
    data = data.groupby('purchase id').agg(list)

    # Scale numeric features
    scaler_price = StandardScaler()
    data['price'] = data['price'].apply(lambda x: scaler_price.fit_transform(np.array(x).reshape(-1, 1)).flatten())

    # Normalize time feature
    scaler_time = MinMaxScaler()
    data['time'] = data['time'].apply(lambda x: scaler_time.fit_transform(np.array(x).reshape(-1, 1)).flatten())

    # Create sequences
    X_price = data[['price']].apply(lambda x: np.stack(x, axis=1), axis=1)
    X_time = data[['time']].apply(lambda x: np.stack(x, axis=1), axis=1)
    y = data['fraud'].apply(lambda x: x[0])

    return X_price, X_time, y, scaler_price, scaler_time


def create_data_loaders(X_price_train, X_time_train, y_train, X_price_val, X_time_val, y_val, batch_size=64):
    # Train-test split
    X_time_train, X_time_test, X_price_train, X_price_test, y_train, y_test = train_test_split(X_time_train,
                                                                                               X_price_train, y_train,
                                                                                               test_size=0.2,
                                                                                               random_state=42)

    # Pad sequences to have the same length
    X_time_train = pad_sequence([torch.tensor(x, dtype=torch.long) for x in X_time_train], batch_first=True,
                                padding_value=0)
    X_time_test = pad_sequence([torch.tensor(x, dtype=torch.long) for x in X_time_test], batch_first=True,
                               padding_value=0)
    X_time_val = pad_sequence([torch.tensor(x, dtype=torch.long) for x in X_time_val], batch_first=True,
                              padding_value=0)

    X_price_train = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in X_price_train], batch_first=True,
                                 padding_value=-1)
    X_price_test = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in X_price_test], batch_first=True,
                                padding_value=-1)
    X_price_val = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in X_price_val], batch_first=True,
                               padding_value=-1)

    # Convert to PyTorch tensors
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader
    train_dataset = TensorDataset(X_time_train, X_price_train, y_train)
    test_dataset = TensorDataset(X_time_test, X_price_test, y_test)
    val_dataset = TensorDataset(X_time_val, X_price_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_evaluate_lstm(train_loader, val_loader, test_loader, learning_rate, epochs=10, patience=1000, checkpoint_path='model.pt'):
    model = LSTMModel()

    # Define loss and optimizer
    criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    optimizer = AdaBelief(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decouple=True,
                          rectify=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    # Initialize variables for early stopping and model checkpointing
    best_val_loss = float('inf')
    no_improvement_epochs = 0

    # Train the model
    for epoch in range(epochs):
        for batch_x_time, batch_x_price, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x_time, batch_x_price)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate on the validation set
        val_losses = []
        with torch.no_grad():
            for batch_x_time, batch_x_price, batch_y in val_loader:
                outputs = model(batch_x_time, batch_x_price)
                val_loss = criterion(outputs, batch_y)
                val_losses.append(val_loss.item())
        mean_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch+1}, Validation Loss: {mean_val_loss}")

        # Update learning rate
        scheduler.step(mean_val_loss)

        # Early stopping and model checkpointing
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            no_improvement_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print("Early stopping.")
            break

    # Load the best model
    model.load_state_dict(torch.load(checkpoint_path))

    # Evaluate the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x_time, batch_x_price, batch_y in test_loader:
            outputs = model(batch_x_time, batch_x_price)
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    test_accuracy = correct / total
    return model, test_accuracy


def predict_lstm(model, X_time_new, X_price_new, scaler_price):
    # Scale numeric features
    X_price_new = [scaler_price.transform(np.array(x).reshape(-1, 1)).flatten() for x in X_price_new]

    # Create sequences
    X_new = [np.stack((x_time, x_price), axis=1) for x_time, x_price in zip(X_time_new, X_price_new)]

    # Pad sequences
    X_new = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in X_new], batch_first=True, padding_value=-1)

    # Make predictions
    with torch.no_grad():
        new_outputs = model(X_new[:, :, 0], X_new[:, :, 1])  # Pass both x_time and x_price to the model
        new_predicted = (new_outputs > 0.5).float().numpy().flatten()

    return new_predicted


def main(training, predict):
    X_price, X_time, y, scaler_price, scaler_time = preprocess_data(training)

    # Split the dataset into training and validation sets
    X_price_train, X_price_val, X_time_train, X_time_val, y_train, y_val = train_test_split(X_price, X_time, y,
                                                                                            test_size=0.3,
                                                                                            random_state=42)

    train_loader, val_loader, test_loader = create_data_loaders(X_price_train, X_time_train, y_train, X_price_val,
                                                    X_time_val, y_val)  # Fixed the issue here

    model, test_accuracy = train_evaluate_lstm(train_loader, val_loader, test_loader, learning_rate=1e-3, epochs=500)

    new_data = data_transform(predict)
    new_data['fraud'] = new_data['fraud'].astype(int)
    new_data = new_data.groupby('purchase id').agg(list)

    new_predicted = predict_lstm(model, new_data['time'].tolist(), new_data['price'].tolist(), scaler_price)  # Fixed the issue here
    new_predicted = new_predicted.astype(int)
    return new_predicted



