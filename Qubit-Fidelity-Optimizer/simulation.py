import cirq
from qubit_characterization import measure_t1_t2
from real_time_feedback import real_time_feedback_control
from decoupling_sequences import choose_decoupling_sequence

def simulate_without_noise(qubit, simulator, sequence, time_steps=100, repetitions=1000):
    """
    Simulates the qubit dynamics without any noise modeling.
    """
    # Measure T1 and T2 times initially
    t1, t2 = measure_t1_t2(qubit, simulator, repetitions)
    print(f"Initial T1 Time: {t1:.2f} ns, T2 Time: {t2:.2f} ns")

    # Apply the chosen decoupling sequence
    circuit = cirq.Circuit(sequence)
    
    # Apply an initial Hadamard gate to put the qubit in superposition
    circuit.append(cirq.H(qubit))
    circuit.append(cirq.measure(qubit, key='result'))

    measurements = simulator.run(circuit, repetitions=repetitions)
    print("Initial Measurements:", measurements.histogram(key='result'))

    all_measurements = []

    for step in range(time_steps):
        print(f"Time Step {step + 1}/{time_steps}")

        # Perform real-time feedback
        circuit = real_time_feedback_control(circuit, measurements.histogram(key='result'), {}, qubit)

        # Run the simulation again
        measurements = simulator.run(circuit, repetitions=repetitions)

        # Use histogram to access measurement results
        step_results = measurements.histogram(key='result')
        all_measurements.extend([0] * step_results.get(0, 0) + [1] * step_results.get(1, 0))

        print(f"Measurements at Time Step {step + 1}:", step_results)

    print("Final Measurements:", measurements.histogram(key='result'))
    
    # Create a result object with all measurements
    final_result = cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'result': all_measurements})
    
    return final_result
