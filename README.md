# Heart Disease Binary Classifier — Variational Quantum Classifier (VQC)

This project implements a binary heart-disease prediction model using the
"Heart Disease (Cleveland)" dataset and a variational quantum classifier
implemented with PennyLane and PyTorch.

Features
- Data: downloads and cleans the Cleveland dataset from the UCI repository.
- Feature engineering: selects the top-4 numeric features using a Random Forest.
- Encoding: angle encoding (RY) on 4 qubits, mapping scaled features in [0,1] to angles in [0, π].
- Ansatz: 3 variational layers; each layer has parameterized single-qubit rotations (`Rot`) and a ring of CNOT entanglers.
- Integration: uses `qml.qnn.TorchLayer` to integrate the QNode with PyTorch; trains using `BCEWithLogitsLoss` and Adam.

Files
- `src/data.py`: data download, cleaning, feature selection, scaling, train/test split.
- `src/model.py`: quantum circuit and `QuantumClassifier` PyTorch module.
- `train.py`: training loop and model checkpoint saving (`models/best_model.pth`).
- `evaluate.py`: loads best model and reports accuracy, precision, recall, confusion matrix.
- `visualize.py`: plots training history (loss and accuracy) and saves to `models/training_history.png`.
- `requirements.txt`: Python dependencies.

Usage
1. Create and activate a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train the model:

```bash
python train.py
```

3. Evaluate the saved best model:

```bash
python evaluate.py
```

4. Visualize training history and evaluation metrics:

```bash
python visualize.py
```

Notes
- The data loader drops rows with missing values for simplicity.
- The selected 4 features are chosen automatically by feature importance; see console output after `train.py` runs.
- The quantum circuit runs on the PennyLane `default.qubit` simulator.
