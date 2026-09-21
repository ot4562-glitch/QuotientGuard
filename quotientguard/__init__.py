"""QuotientGuard public API."""

from .adapters import (
    CallbackRuntimeCapture,
    FrameworkRuntimeCapture,
    OpenRLHFRuntimeCapture,
    TRLRuntimeCapture,
    VerlRuntimeCapture,
)
from .benchmark import benchmark_sampled_audit
from .certification import run_builtin_certification_suite, run_certification_suite
from .check import check_path, check_payload
from .compare import compare_certificates
from .contracts import (
    DecodedTextContract,
    EquivalenceContract,
    IntegerAnswerContract,
    VerifierContract,
)
from .core import audit_receipt, audit_class
from .demo import run_paper_a1_demo
from .explain import explain_trl_payload
from .sampled import audit_sampled_observations
from .schema import EstimatorFingerprint, Observation
from .sketch import HashProjection, sketch_observations
from .source_manifests import (
    SOURCE_MANIFESTS,
    verified_installed_trl_provenance,
    verify_source_file,
)
from .testing import (
    assert_no_measured_gap,
    assert_no_new_measured_gap,
    assert_policy_passed,
)
from .version import __version__

__all__ = [
    "audit_receipt",
    "audit_class",
    "audit_sampled_observations",
    "check_path",
    "check_payload",
    "compare_certificates",
    "run_paper_a1_demo",
    "explain_trl_payload",
    "run_certification_suite",
    "run_builtin_certification_suite",
    "FrameworkRuntimeCapture",
    "CallbackRuntimeCapture",
    "TRLRuntimeCapture",
    "VerlRuntimeCapture",
    "OpenRLHFRuntimeCapture",
    "verify_source_file",
    "verified_installed_trl_provenance",
    "SOURCE_MANIFESTS",
    "HashProjection",
    "sketch_observations",
    "benchmark_sampled_audit",
    "assert_policy_passed",
    "assert_no_measured_gap",
    "assert_no_new_measured_gap",
    "EquivalenceContract",
    "VerifierContract",
    "IntegerAnswerContract",
    "DecodedTextContract",
    "EstimatorFingerprint",
    "Observation",
    "__version__",
]
