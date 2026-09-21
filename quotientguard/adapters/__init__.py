"""Trainer adapters shipped with QuotientGuard."""

from .callback import CallbackRuntimeCapture
from .openrlhf import OpenRLHFRuntimeCapture
from .runtime_base import FrameworkRuntimeCapture
from .trl_runtime import TRLRuntimeCapture
from .trl_sequence import (
    CoefficientTrace,
    effective_coefficient_trace,
    trl_sequence_batch_to_receipt,
    validate_trl_source_provenance,
)
from .verl import VerlRuntimeCapture

__all__ = [
    "FrameworkRuntimeCapture",
    "CallbackRuntimeCapture",
    "TRLRuntimeCapture",
    "VerlRuntimeCapture",
    "OpenRLHFRuntimeCapture",
    "CoefficientTrace",
    "effective_coefficient_trace",
    "trl_sequence_batch_to_receipt",
    "validate_trl_source_provenance",
]
