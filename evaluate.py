import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from src.data import load_data
from src.model import QuantumClassifier


def evaluate(model_path: str = "models/best_model.pth"):
    X_train, X_test, y_train, y_test, selected, scaler = load_data()

    # Load model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = QuantumClassifier()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    print(f"Selected features: {selected}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print("Confusion matrix:")
    print(cm)


if __name__ == '__main__':
    evaluate()
