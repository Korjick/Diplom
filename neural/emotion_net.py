import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Executed on:', device)

    dataset = pd.read_csv('./gaze_recognition/gaze_data.txt', sep=',')
    dataset = dataset.drop(['timestamp', 'index'], axis=1)
    X = dataset.drop(
        ['anger', 'tenderness', 'disgust', 'sadness', 'pupil_norm_pos_0_x', 'pupil_norm_pos_0_y', 'pupil_norm_pos_1_x',
         'pupil_norm_pos_1_y'], axis=1)
    X = X.fillna(0)
    y = dataset[['anger', 'tenderness', 'disgust', 'sadness']]
    X = X.to_numpy()
    y = y.to_numpy()

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    nn_model = NeuralNetwork().to(device)

    # Создание оптимизатора и функции потерь
    optimizer = optim.AdamW(nn_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_acc = -np.inf
    best_weights = None
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []

    # Обучение нейронной сети с использованием DataLoader
    for epoch in range(300):
        epoch_loss = []
        epoch_acc = []
        for batch_X, batch_y in train_dataloader:
            optimizer.zero_grad()
            y_pred = nn_model(batch_X)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            acc = (torch.argmax(y_pred, 1) == torch.argmax(batch_y, 1)).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))

        nn_model.eval()
        y_pred = nn_model(X_test_tensor)
        ce = criterion(y_pred, y_test_tensor)
        acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test_tensor, 1)).float().mean()
        ce = float(ce)
        acc = float(acc)
        train_loss_hist.append(np.mean(epoch_loss))
        train_acc_hist.append(np.mean(epoch_acc))
        test_loss_hist.append(ce)
        test_acc_hist.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(nn_model.state_dict())
        print(f"Epoch {epoch} validation: C-E={ce:.2f}, Accuracy={acc * 100:.1f}%")

    torch.save(best_weights, "nn_model_2.pth")

    # Загрузка моделей
    nn_model = NeuralNetwork().to(device)
    nn_model.load_state_dict(torch.load("nn_model_2.pth"))

    # Plot the loss and accuracy
    plt.plot(train_loss_hist, label="train")
    plt.plot(test_loss_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.show()

    plt.plot(train_acc_hist, label="train")
    plt.plot(test_acc_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()