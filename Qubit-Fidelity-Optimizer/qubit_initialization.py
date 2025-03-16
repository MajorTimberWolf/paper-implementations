# qubit_initialization.py

import cirq

def create_qubit_cirq(state='0', position=(0, 0)):
    qubit = cirq.GridQubit(*position)
    circuit = cirq.Circuit()
    
    if state == '0':
        pass  # |0> is the default state
    elif state == '1':
        circuit.append(cirq.X(qubit))
    elif state == '+':
        circuit.append(cirq.H(qubit))
    elif state == '-':
        circuit.append([cirq.X(qubit), cirq.H(qubit)])
    else:
        raise ValueError(f"Invalid state '{state}' provided.")
    
    # Add a measurement to verify the initial state
    circuit.append(cirq.measure(qubit, key='init_state'))
    
    return qubit, circuit


def initialize_entangled_pair():
    """
    Initializes two qubits in a Bell state (|00⟩ + |11⟩) / sqrt(2).

    Returns:
    - qubits: A list of initialized cirq.GridQubit objects.
    - circuit: A cirq.Circuit object containing the operations to create the Bell state.
    """
    # Define two qubits at different grid positions
    qubit_0 = cirq.GridQubit(0, 0)
    qubit_1 = cirq.GridQubit(0, 1)
    circuit = cirq.Circuit()

    # Prepare the Bell state (|00⟩ + |11⟩) / sqrt(2)
    circuit.append(cirq.H(qubit_0))
    circuit.append(cirq.CNOT(qubit_0, qubit_1))

    return [qubit_0, qubit_1], circuit
