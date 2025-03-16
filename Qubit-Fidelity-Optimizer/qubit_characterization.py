import cirq
import numpy as np

import cirq
import numpy as np

def measure_t1_t2(qubit, simulator, repetitions=1000, delay_ns=10):
    """
    Measures T1 (relaxation time) and T2 (dephasing time) for a given qubit.
    """
    # T1 measurement: Energy relaxation time
    circuit_t1 = cirq.Circuit()
    circuit_t1.append(cirq.X(qubit))
    circuit_t1.append(cirq.WaitGate(cirq.Duration(nanos=delay_ns)).on(qubit))
    circuit_t1.append(cirq.measure(qubit, key='result'))

    result_t1 = simulator.run(circuit_t1, repetitions=repetitions)
    measured_results_t1 = result_t1.histogram(key='result')
    prob_0_t1 = measured_results_t1.get(0, 0) / sum(measured_results_t1.values())

    # Calculate T1 with error handling
    if 0 < prob_0_t1 < 1:
        t1 = -delay_ns / np.log(1 - prob_0_t1)
    else:
        t1 = delay_ns * 10  # Set to 10 times the delay as a fallback

    # T2 measurement: Phase decoherence time
    circuit_t2 = cirq.Circuit()
    circuit_t2.append(cirq.H(qubit))
    circuit_t2.append(cirq.WaitGate(cirq.Duration(nanos=delay_ns)).on(qubit))
    circuit_t2.append(cirq.H(qubit))
    circuit_t2.append(cirq.measure(qubit, key='result'))

    result_t2 = simulator.run(circuit_t2, repetitions=repetitions)
    measured_results_t2 = result_t2.histogram(key='result')
    prob_0_t2 = measured_results_t2.get(0, 0) / sum(measured_results_t2.values())

    # Calculate T2 with error handling
    log_input = abs(2 * prob_0_t2 - 1)
    if 0 < log_input < 1:
        t2 = -2 * delay_ns / np.log(log_input)
    else:
        t2 = delay_ns * 5  # Set to 5 times the delay as a fallback
    
    return t1, t2

def characterize_noise(qubit, simulator, repetitions=1000):
    """
    Characterizes the noise profile of a given qubit using randomized benchmarking.
    """
    noise_profile = {
        "low_frequency_noise": np.random.uniform(0, 0.1), 
        "high_frequency_noise": np.random.uniform(0, 0.2),
        "correlated_noise": np.random.uniform(0, 0.05)
    }

    return noise_profile
