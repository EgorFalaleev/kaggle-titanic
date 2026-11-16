import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from preprocessing import Preprocessor
from config import config

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

def train_model(model, train_dataset, val_dataset, device):
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.dl.learning_rate
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dl.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dl.batch_size,
        shuffle=False
    )

    best_val_loss = float('inf')
    best_model_state = None

    patience = 5
    epochs_without_improvement = 0

    for epoch in range(config.dl.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_metrics = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch + 1}/{config.dl.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Accuracy: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return best_model_state


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        predictions = (logits >= 0).float()

        total_loss += loss.item() * len(X_batch)
        correct += (predictions == y_batch).sum().item()

    return {
        'loss': total_loss / len(dataloader.dataset),
        'accuracy': correct / len(dataloader.dataset),
    }   

def prepare_data(X_train, X_val, y_train, y_val):
    preprocessor = Preprocessor()

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train.to_numpy(), dtype=torch.float32)
)

    val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val.to_numpy(), dtype=torch.float32)
)

    return train_dataset, val_dataset, preprocessor, scaler

