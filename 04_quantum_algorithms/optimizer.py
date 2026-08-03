"""
Module: 04_quantum_algorithms.optimizer
Factory for initializing classical optimizers for hybrid quantum algorithms.
"""

import logging
from typing import Optional
from qiskit_algorithms.optimizers import COBYLA, SPSA, SLSQP, Optimizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_optimizer(
    optimizer_name: str = 'COBYLA', 
    maxiter: int = 100, 
    tol: Optional[float] = None
) -> Optimizer:
    """
    Returns a configured Qiskit classical optimizer.

    Args:
        optimizer_name (str): The name of the optimizer ('COBYLA', 'SPSA', 'SLSQP').
        maxiter (int): The maximum number of evaluation iterations.
        tol (Optional[float]): The convergence tolerance.

    Returns:
        Optimizer: The instantiated Qiskit optimizer.
    """
    opt_name = optimizer_name.strip().upper()
    
    if opt_name == 'COBYLA':
        logger.debug(f"Initializing COBYLA optimizer (maxiter={maxiter}, tol={tol}).")
        return COBYLA(maxiter=maxiter, tol=tol)
        
    elif opt_name == 'SPSA':
        logger.debug(f"Initializing SPSA optimizer (maxiter={maxiter}).")
        # SPSA handles noise well; tol is not standardly applied in the same way
        return SPSA(maxiter=maxiter)
        
    elif opt_name == 'SLSQP':
        logger.debug(f"Initializing SLSQP optimizer (maxiter={maxiter}, tol={tol}).")
        return SLSQP(maxiter=maxiter, tol=tol)
        
    else:
        logger.error(f"Unsupported optimizer requested: {optimizer_name}")
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported. Choose from COBYLA, SPSA, SLSQP.")