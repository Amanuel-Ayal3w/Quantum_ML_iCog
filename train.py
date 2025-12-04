import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


from src.data import load_data
from src.model import QuantumClassifier


def train(epochs: int = 30, batch_size: int = 16, lr: float = 0.01, device='cpu'):
    X_train, X_test, y_train, y_test, selected, scaler = load_data()

    # Convert to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = QuantumClassifier()
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.shape[0]
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += xb.shape[0]

        train_loss = epoch_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        with torch.no_grad():
            logits_val = model(X_test_t)
            probs = torch.sigmoid(logits_val)
            preds_val = (probs > 0.5).float()
            val_acc = (preds_val == y_test_t).sum().item() / y_test_t.shape[0]
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, time={elapsed:.2f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    # Save best model
    if best_state is not None:
        os.makedirs("models", exist_ok=True)
        torch.save({
            'model_state_dict': best_state,
            'selected_features': selected,
        }, os.path.join("models", "best_model.pth"))
        
        import json
        with open(os.path.join("models", "history.json"), "w") as f:
            json.dump(history, f)

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    train(epochs=30, batch_size=16, lr=0.01)
