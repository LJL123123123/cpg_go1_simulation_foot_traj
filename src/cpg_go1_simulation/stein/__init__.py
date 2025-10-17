"""
Stein oscillator based CPG implementations
"""

from .base import CPGBase
from .implementations import CPG8Neuron
from .foot_trajectory_cpg import FootTrajectoryCPG

__all__ = ["CPGBase", "CPG8Neuron", "FootTrajectoryCPG"]