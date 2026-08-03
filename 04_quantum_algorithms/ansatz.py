"""
Module: 04_quantum_algorithms.ansatz
Constructs custom parameterized quantum circuits (ansätze) for variational algorithms.
"""

import logging
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def build_portfolio_ansatz(
    num_qubits: int, 
    reps: int = 2, 
    entanglement: str = 'linear',
    insert_barriers: bool = True
) -> QuantumCircuit:
    """
    Builds a hardware-efficient ansatz using Ry and Rz rotations with CX entanglement.

    Args:
        num_qubits (int): Total number of qubits required.
        reps (int): Number of alternating rotation and entanglement layers.
        entanglement (str): Type of entanglement ('linear' or 'circular').
        insert_barriers (bool): Whether to insert visual barriers between layers.

    Returns:
        QuantumCircuit: The parameterized quantum circuit.
    """
    circuit = QuantumCircuit(num_qubits)
    
    # Total parameters: (reps + 1) layers of rotations * 2 rotations (Ry, Rz) * num_qubits
    num_params = (reps + 1) * num_qubits * 2
    theta = ParameterVector('θ', num_params)
    
    param_idx = 0
    
    # Initial rotation layer
    for q in range(num_qubits):
        circuit.ry(theta[param_idx], q)
        param_idx += 1
        circuit.rz(theta[param_idx], q)
        param_idx += 1
        
    if insert_barriers:
        circuit.barrier()
        
    # Repetition layers (Entanglement + Rotations)
    for rep in range(reps):
        # Entanglement layer
        if entanglement == 'linear':
            for q in range(num_qubits - 1):
                circuit.cx(q, q + 1)
        elif entanglement == 'circular':
            for q in range(num_qubits):
                circuit.cx(q, (q + 1) % num_qubits)
        else:
            logger.warning(f"Unknown entanglement '{entanglement}'. Defaulting to 'linear'.")
            for q in range(num_qubits - 1):
                circuit.cx(q, q + 1)
                
        if insert_barriers:
            circuit.barrier()
            
        # Rotation layer
        for q in range(num_qubits):
            circuit.ry(theta[param_idx], q)
            param_idx += 1
            circuit.rz(theta[param_idx], q)
            param_idx += 1
            
        if insert_barriers:
            circuit.barrier()
            
    logger.debug(f"Built ansatz with {num_qubits} qubits and {num_params} parameters.")
    return circuit