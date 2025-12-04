import pennylane as qml
from pennylane import numpy as pnp
import torch
import torch.nn as nn
import numpy as np

# Quantum model hyperparameters
N_QUBITS = 4
N_LAYERS = 3

dev = qml.device("default.qubit", wires=N_QUBITS)

# Encode 4 features into 4 qubits via angle mbedding 
def state_preparation(x):
    qml.AngleEmbedding(features=x, wires=range(N_QUBITS), rotation='Y')

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")

def circuit(inputs, weights):
    state_preparation(inputs)

    # Variational layer
    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            # Each qubit has a Rot with three parameters
            qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)

        # Entangling layer: ring of CNOTs
        for q in range(N_QUBITS - 1):
            qml.CNOT(wires=[q, q + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])

    return qml.expval(qml.PauliZ(0))



# Define the weight shapes for TorchLayer
weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}

# Create a TorchLayer which will hold the trainable quantum weights
qml_layer = qml.qnn.TorchLayer(circuit, weight_shapes)


class QuantumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # qml_layer expects input shape (batch, 4)
        self.qlayer = qml_layer
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        
        expvals = self.qlayer(x)
        # Add a classical bias and treat the result as logits
        logits = expvals + self.bias

        return logits.squeeze()


if __name__ == '__main__':
    model = QuantumClassifier()




