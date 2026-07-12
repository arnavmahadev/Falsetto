"""End-to-end inference: audio file -> P(AI) + label."""

from .predict import Prediction, Predictor

__all__ = ["Predictor", "Prediction"]
