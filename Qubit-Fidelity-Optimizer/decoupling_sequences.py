import cirq
import numpy as np

def udd_sequence(qubit, n, total_duration):
    """
    Implements the Uhrig Dynamical Decoupling (UDD) sequence for a given qubit.
    """
    udd_circuit = cirq.Circuit()
    
    def tj(j):
        return total_duration * np.sin(np.pi * j / (2 * n + 2))**2

    last_t = 0
    for j in range(1, n + 1):
        t = int(round(tj(j)))
        delay_duration = t - last_t
        if delay_duration > 0:
            udd_circuit.append(cirq.WaitGate(cirq.Duration(nanos=delay_duration)).on(qubit))
        udd_circuit.append(cirq.X(qubit))
        last_t = t

    final_delay = total_duration - last_t
    if final_delay > 0:
        udd_circuit.append(cirq.WaitGate(cirq.Duration(nanos=final_delay)).on(qubit))

    return udd_circuit

def cdd_sequence(qubit, levels):
    """
    Implements the Concatenated Dynamical Decoupling (CDD) sequence.
    """
    if levels == 1:
        return cirq.Circuit(cirq.X(qubit), cirq.X(qubit))
    
    base_sequence = cdd_sequence(qubit, levels - 1)
    cdd_circuit = cirq.Circuit(base_sequence, cirq.X(qubit), base_sequence)
    return cdd_circuit

def cpmg_sequence(qubit, n, tau):
    """
    Implements the Carr-Purcell-Meiboom-Gill (CPMG) sequence for a given qubit.
    """
    cpmg_circuit = cirq.Circuit()
    for _ in range(n):
        cpmg_circuit.append(cirq.WaitGate(cirq.Duration(nanos=tau)).on(qubit))
        cpmg_circuit.append(cirq.X(qubit))
    cpmg_circuit.append(cirq.WaitGate(cirq.Duration(nanos=tau)).on(qubit))
    return cpmg_circuit

def choose_decoupling_sequence(qubit, noise_profile):
    """
    Chooses an optimal decoupling sequence based on the qubit's noise profile.
    """
    if noise_profile["low_frequency_noise"] > noise_profile["high_frequency_noise"]:
        chosen_sequence = udd_sequence(qubit, n=5, total_duration=1000)
    elif noise_profile["high_frequency_noise"] > noise_profile["low_frequency_noise"]:
        chosen_sequence = cdd_sequence(qubit, levels=3)
    else:
        chosen_sequence = cpmg_sequence(qubit, n=10, tau=100)
    
    return chosen_sequence
