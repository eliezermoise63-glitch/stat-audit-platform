"""
Core package — Statistical Audit Platform
"""
from .sanitizer import DataSanitizer
from .engine import MultivariateEngine, ACMResult, AFDMResult
from .detector import VariableDetector, DetectionReport

__all__ = [
    "DataSanitizer",
    "MultivariateEngine",
    "ACMResult",
    "AFDMResult",
    "VariableDetector",
    "DetectionReport",
]
