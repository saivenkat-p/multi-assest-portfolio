"""
Module: 04_quantum_algorithms.circuit_visualizer
Provides utilities to draw and save quantum circuit diagrams.
"""

import logging
from typing import Optional
from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def draw_circuit(
    circuit: QuantumCircuit, 
    output_path: Optional[str] = None, 
    style: str = 'mpl'
) -> None:
    """
    Draws a quantum circuit and optionally saves it to a file.

    Args:
        circuit (QuantumCircuit): The circuit to visualize.
        output_path (Optional[str]): Filepath to save the image (e.g., 'circuit.png').
        style (str): The drawing backend ('mpl' for graphic, 'text' for console).
    """
    if not isinstance(circuit, QuantumCircuit):
        logger.error("Provided object is not a Qiskit QuantumCircuit.")
        raise TypeError("Expected a QuantumCircuit object.")

    try:
        if output_path:
            logger.info(f"Saving circuit diagram to {output_path} using {style} style.")
            # draw() with a filename saves it directly
            circuit.draw(output=style, filename=output_path)
        else:
            logger.info("Drawing circuit to console (text mode).")
            # Force text style for console output
            print(circuit.draw(output='text'))
            
    except ImportError as e:
        logger.error(f"Missing dependency for drawing style '{style}': {str(e)}")
        logger.info("Fallback: Printing text representation instead.")
        print(circuit.draw(output='text'))
    except Exception as e:
        logger.error(f"Failed to draw circuit: {str(e)}")
        raise