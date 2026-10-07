"""
TRACE-Tutor: Trajectory- and Role-Aware Curriculum Evidence Tutor

A research prototype for curriculum-grounded primary education tutoring.
"""

__version__ = "1.0.0"
__author__ = "TRACE-Tutor Research Team"

from trace_tutor.schema import Evidence, Query, Edge, Answer
from trace_tutor.tutor import TRACETutor

__all__ = [
    "Evidence",
    "Query",
    "Edge",
    "Answer",
    "TRACETutor",
]
