import matplotlib.pyplot as plt
import json
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from src.data import load_data
from src.model import QuantumClassifier

def plot_history(history_path="models/history.json"):
    try:
        with open(history_path, "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"History file not found at {history_path}. Run train.py first.")
        return
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    print("Saved training history plot to models/training_history.png")
    # plt.show() # Uncomment if running in a GUI environment

def plot_evaluation(model_path="models/best_model.pth"):
    try:
        checkpoint = torch.load(model_path)
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Run train.py first.")
        return

    X_train, X_test, y_train, y_test, selected, scaler = load_data()
    
    model = QuantumClassifier()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(int)
        
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('models/confusion_matrix.png')
    print("Saved confusion matrix plot to models/confusion_matrix.png")
    # plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('models/roc_curve.png')
    print("Saved ROC curve plot to models/roc_curve.png")
    # plt.show()

if __name__ == "__main__":
    plot_history()
    plot_evaluation()
