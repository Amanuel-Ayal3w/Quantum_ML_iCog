# debug_encoding.py

import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt

N_QUBITS = 4
N_LAYERS = 3

dev = qml.device("default.qubit", wires=N_QUBITS)

def state_preparation(x):
    qml.AngleEmbedding(features=x, wires=range(N_QUBITS), rotation="Y")

@qml.qnode(dev)
def encoding_only(x):
    state_preparation(x)
    return qml.state()

@qml.qnode(dev)
def full_circuit(x, weights):
    state_preparation(x)
    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
        for q in range(N_QUBITS - 1):
            qml.CNOT(wires=[q, q + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])
    return qml.expval(qml.PauliZ(0))

if __name__ == "__main__":
    x = pnp.array([0.1, 0.2, 0.3, 0.4])
    weights = pnp.zeros((N_LAYERS, N_QUBITS, 3))

    # 1) Text diagram (terminal)
    print("=== Circuit diagram ===")
    print(qml.draw(full_circuit)(x, weights))

    # 2) State after encoding
    psi = encoding_only(x)
    print("\n=== State after encoding only ===")
    print("State vector:", psi)
    print("Shape:", psi.shape)
    print("Norm:", pnp.sum(pnp.abs(psi) ** 2))

    # 3) Matplotlib circuit diagram → saved as JPEG, no GUI
    fig, ax = qml.draw_mpl(full_circuit)(x, weights)
    fig.savefig("circuit_diagram.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved circuit diagram to circuit_diagram.jpg")
