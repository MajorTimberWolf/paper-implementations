# Quantum Circuit Topology Analysis

This project explores the impact of different qubit topologies (Star, Linear Chain, Ring, and Fully Connected) on the performance of quantum circuits. Using Qiskit, the project simulates a quantum circuit across these topologies, comparing metrics such as circuit depth, gate count, and final state fidelity to analyze the efficiency and effectiveness of each topology. See `output.png` for results.

## Requirements

- Python 3.11.x
- Qiskit = 1.1.1

## Usage

1. **Run the Scripts**:
   - **`main.ipynb`**: This notebook explores the general performance of quantum circuits across Star, Linear Chain, Ring, and Fully Connected topologies. The notebook:
     - Creates a quantum circuit with multi-qubit operations.
     - Simulates the circuit using various topologies.
     - Transpiles the circuits for each topology to observe how they differ.
     - Executes the transpiled circuits on a simulator.
     - Visualizes the results with a histogram comparing the measurement outcomes.

   - **`star-topology.ipynb`**: This notebook specifically showcases the advantages of the Star topology in the context of superconducting qubits. It demonstrates:
     - How a central qubit can efficiently distribute entanglement across other qubits in a Star topology.
     - The reduction in circuit complexity and execution time for specific quantum operations when using a Star topology.

2. **Analyze the Results**: Both notebooks output the transpiled circuits for each topology, the circuit depth, gate counts, and histograms comparing the outcomes.

### Example Output

- **Transpiled Circuits**: The notebooks print out the structure of the transpiled circuits for each topology.
- **Measurement Results**: The results are displayed in histograms, showing the distribution of outcomes (e.g., `000` vs `011` states) for each topology.
- **Circuit Depth and Gate Count**: The output includes the depth of each circuit and the number of gates used, which are key metrics for understanding the efficiency of each topology.

## Key Concepts

- **Topologies**:
  - **Star Topology**: Central qubit interacts with several others, but non-central qubits are not directly connected. Ideal for superconducting qubit systems where centralized control is crucial.
  - **Linear Chain Topology**: Qubits are connected in a straight line, allowing direct operations between adjacent qubits.
  - **Ring Topology**: Qubits are connected in a loop, reducing the need for SWAP operations in some cases.
  - **Fully Connected Topology**: Every qubit is directly connected to every other qubit, minimizing the need for additional gates.

- **Performance Metrics**:
  - **Circuit Depth**: The number of layers of gates in the circuit. A higher depth can increase the likelihood of errors in quantum hardware.
  - **Gate Count**: The number of quantum gates (e.g., CNOT, Hadamard, SWAP) in the circuit. More gates generally mean more complexity and a higher chance of errors.

## Observations

- The **Fully Connected Topology** generally performs the best in terms of minimizing circuit depth and gate count, leading to the most efficient execution.
- The **Star Topology** can be advantageous in specific scenarios, particularly in superconducting qubit systems, where a central qubit needs to control several others efficiently.
- **Linear Chain** and **Ring Topologies** offer a good balance but may require SWAP operations, depending on the circuit's structure.

## Contributions

Feel free to contribute to this project by:

- Adding more complex quantum circuits to compare across different topologies.
- Implementing additional topologies for comparison.
- Enhancing the visualization of results.
