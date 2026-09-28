"""
TMRM: Topological Manifold Resonant Machine
===========================================
A unified foundation machine learning architecture for tabular and high-dimensional predictive modeling.

Authors:
    Balaji P, Navaneetham V, Dhavan RG

License:
    MIT
"""

from tmrm.core import (
    TMRM,
    TopologicalManifoldResonantMachine,
    StreamingTMRM,
    __version__,
)

__all__ = [
    "TMRM",
    "TopologicalManifoldResonantMachine",
    "StreamingTMRM",
    "__version__",
]
