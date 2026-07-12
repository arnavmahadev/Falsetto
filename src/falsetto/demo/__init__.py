"""Self-contained interactive demo (FALSETTO Studio).

Runs the real MERT/beat/SSM/fusion pipeline on any audio; the classifier is
trained self-supervised on a structural-coherence task so the demo needs no
external datasets. See :mod:`falsetto.demo.studio`.
"""

from .assets import build_demo_assets, load_demo_model
from .pipeline import DemoAnalyzer, DemoFeatures, DemoPipeline, DemoResult, adaptive_ssm
from .synth import generate_clip

__all__ = [
    "generate_clip",
    "build_demo_assets",
    "load_demo_model",
    "DemoPipeline",
    "DemoAnalyzer",
    "DemoFeatures",
    "DemoResult",
    "adaptive_ssm",
]
