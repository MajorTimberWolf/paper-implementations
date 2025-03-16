import cirq

class AdvancedNoiseModel(cirq.NoiseModel):
    """Advanced noise model with dynamic noise suppression techniques."""

    def __init__(self, depolarizing_strength, damping_strength, phase_strength, correlated_strength=0.0):
        # Define the noise components
        self.depolarizing_noise = cirq.depolarize(p=depolarizing_strength)
        self.amplitude_damping_noise = cirq.amplitude_damp(gamma=damping_strength)
        self.phase_damping_noise = cirq.phase_damp(gamma=phase_strength)
        self.correlated_noise = correlated_strength

    def noisy_moment(self, moment, system_qubits):
        noisy_ops = []
        for op in moment:
            # Apply noise only to quantum gate operations
            if isinstance(op.gate, (cirq.GateOperation, cirq.SingleQubitGate, cirq.TwoQubitGate)):
                for qubit in op.qubits:
                    # Apply the appropriate noise types to each operation
                    noisy_ops.append(self.depolarizing_noise.on(qubit))
                    noisy_ops.append(self.amplitude_damping_noise.on(qubit))
                    noisy_ops.append(self.phase_damping_noise.on(qubit))
                    
                    # Optionally apply correlated noise
                    if self.correlated_noise > 0:
                        noisy_ops.append(cirq.depolarize(p=self.correlated_noise).on_each(system_qubits))
                    
                noisy_ops.append(op)  # Append the original operation
            else:
                noisy_ops.append(op)  # Non-gate operations are passed without noise
        return cirq.Moment(noisy_ops)
