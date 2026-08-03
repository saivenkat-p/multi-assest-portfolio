"""
Module: 04_quantum_algorithms.backend
Configures execution backends and primitives for quantum solvers.
"""

import logging
from typing import Tuple, Dict, Any, Optional
from qiskit.primitives import StatevectorSampler, StatevectorEstimator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_primitives(
    backend_type: str = 'local_statevector', 
    options: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Any]:
    """
    Provisions the Sampler and Estimator primitives based on the target backend.

    Args:
        backend_type (str): The execution environment identifier.
        options (Optional[Dict]): Configuration dictionary for primitive options.

    Returns:
        Tuple[Any, Any]: (Sampler, Estimator) instances ready for algorithm execution.
    """
    target = backend_type.strip().lower()
    options = options or {}
    
    if target in ['local', 'local_statevector', 'statevector']:
        logger.debug("Initializing local Statevector primitives for exact simulation.")
        
        # Statevector primitives compute exact probabilities/expectation values
        sampler = StatevectorSampler(default_shots=options.get("shots"))
        estimator = StatevectorEstimator(default_shots=options.get("shots"))
        
        # Note: If random seeds are needed for Statevector execution in newer Qiskit versions,
        # they are typically passed at the `.run()` level or via specific options objects.
        
        return sampler, estimator
        
    elif target in ['ibmq', 'hardware', 'qpu']:
        logger.error("Hardware execution requires 'qiskit-ibm-runtime' and active credentials.")
        raise NotImplementedError("Hardware primitives via Qiskit Runtime are not yet configured in this environment.")
        
    else:
        logger.error(f"Unknown backend requested: {backend_type}")
        raise ValueError(f"Backend '{backend_type}' is not supported.")