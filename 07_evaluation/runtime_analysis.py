"""
Module: 07_evaluation.runtime_analysis
Measures execution time for portfolio optimization solvers.
"""

import logging
import time
from typing import Callable, Tuple, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class RuntimeProfiler:
    """
    Profiles the execution time of any given solver function.
    """

    def __init__(self):
        """
        Initializes the runtime profiler.
        """
        pass

    def measure_execution_time(self, solver_function: Callable, *args, **kwargs) -> Tuple[float, Any]:
        """
        Executes a function and measures its runtime.

        Args:
            solver_function (Callable): The solver method to execute.
            *args: Positional arguments for the solver.
            **kwargs: Keyword arguments for the solver.

        Returns:
            Tuple[float, Any]: The elapsed time in seconds, followed by the solver's output.
        """
        solver_name = solver_function.__qualname__ if hasattr(solver_function, '__qualname__') else str(solver_function)
        logger.info(f"Starting runtime profile for: {solver_name}")
        
        # Start high-resolution timer
        start_time = time.perf_counter()
        
        # Execute the solver
        try:
            solver_output = solver_function(*args, **kwargs)
        except Exception as e:
            logger.error(f"Execution failed for {solver_name}. Error: {str(e)}")
            raise
            
        # Stop timer
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        logger.info(f"Execution complete. Runtime for {solver_name}: {elapsed_time:.4f} seconds.")
        
        return elapsed_time, solver_output