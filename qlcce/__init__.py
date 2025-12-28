"""
QLCCE - Quantum-Logarithmic Chaotic Constraint Engine

A research environment for quantum field simulations, Benford's law analysis,
and chaotic systems.
"""

from .core.quantum_field import QuantumFieldSampler
from .core.log_transform import MultiLogTransformer
from .core.chaotic_system import ChaoticConstraintSystem
from .core.benford_analyzer import BenfordQuantumAnalyzer

__version__ = "1.0.0"
__all__ = [
    "QuantumFieldSampler",
    "MultiLogTransformer",
    "ChaoticConstraintSystem",
    "BenfordQuantumAnalyzer",
]
