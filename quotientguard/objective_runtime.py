"""Capture actual loss-graph derivatives with explicit, conservative evidence gates."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable

from .objective import (ObjectiveSpec, ObjectiveTrace, ObjectiveAuditResult,
                        audit_objective, _close, _plain, _json_safe)
from .provenance import sha256_json


class ObjectiveRuntimeCapture:
    """Local evidence, not authentication of a remote trainer or semantic theorem.

    `FRAMEWORK_INTEGRATION` must be declared by the integrator and bound to a
    production loss callable and independently pinned source SHA256. Synthetic
    inputs, source-equivalent reproductions and unbound tensors cannot confirm
    framework mismatches, even if all caller-provided booleans claim validation.
    """
    def __init__(self, *, policy_snapshot_id: str, model_version: str,
                 code_version: str, framework: str,
                 execution_kind: str = "UNVERIFIED_RUNTIME", synthetic_inputs: bool = True,
                 source_sha256: str | None = None):
        if not all(str(v).strip() for v in (policy_snapshot_id, model_version, code_version, framework)):
            raise ValueError("policy, model, code and framework identities are required")
        self.provenance = dict(policy_snapshot_id=policy_snapshot_id, model_version=model_version,
            code_version=code_version, framework=framework, execution_kind=execution_kind,
            synthetic_inputs=synthetic_inputs, source_sha256=source_sha256)
        self._repetitions: dict[str, set[str]] = {}
        self._results: list[ObjectiveAuditResult] = []

    def record_backward(self, spec: ObjectiveSpec, trace: ObjectiveTrace, loss: Any = None, *,
                        reproduction_id: str,
                        equivalence_contract: dict[str, Any] | None = None,
                        contract_verifier: Callable[[ObjectiveTrace], bool] | None = None,
                        reconstruction_loss: Callable[[], Any] | None = None,
                        production_loss: Callable[[ObjectiveTrace], Any] | None = None) -> ObjectiveAuditResult:
        import torch

        if not reproduction_id.strip():
            raise ValueError("reproduction_id is required")
        source_verified = False
        source_digest = None
        if production_loss is not None:
            source_path = inspect.getsourcefile(production_loss)
            if source_path and self.provenance["source_sha256"]:
                source_digest = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
                source_verified = source_digest == self.provenance["source_sha256"]
            loss = production_loss(trace)
        x = trace.current_log_probs
        if not isinstance(x, torch.Tensor) or not x.requires_grad:
            raise ValueError("current_log_probs must be a live requires_grad tensor")
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise ValueError("loss must be a scalar tensor from the production graph")
        gradient = (torch.autograd.grad(loss, x, retain_graph=True, allow_unused=True)[0]
                    if loss.requires_grad else None)
        connected = gradient is not None
        if gradient is None:
            gradient = torch.zeros_like(x)
        reconstruction_valid = False
        if reconstruction_loss is not None:
            reconstructed = reconstruction_loss()
            reconstructed_gradient = (torch.autograd.grad(reconstructed, x, retain_graph=True,
                allow_unused=True)[0] if reconstructed.requires_grad else None)
            reconstruction_valid = (connected and reconstructed_gradient is not None
                and _close(loss, reconstructed, spec)
                and _close(gradient, reconstructed_gradient, spec))
        contract = dict(equivalence_contract or {})
        contract_valid = bool(contract.get("id") and contract.get("version") and contract.get("claim")
            and contract_verifier is not None and contract_verifier(trace) is True)
        captured = replace(trace, observed_coefficients=gradient * spec.gradient_scale if connected else None,
                           observed_loss=loss, provenance={**trace.provenance, **self.provenance})
        result = audit_objective(spec, captured, oracle=True)
        evidence = dict(result.evidence)
        detach_validation = {}
        detach_violations = []
        for name, declared_detached in (("old_log_probs", spec.detach_old_policy),
                ("advantages", spec.detach_advantage),
                ("importance_weights", spec.detach_importance), ("mask", True)):
            tensor = getattr(trace, name)
            dependency = None
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad and loss.requires_grad:
                dependency = torch.autograd.grad(loss, tensor, retain_graph=True, allow_unused=True)[0]
            detach_validation[name] = {"declared_detached": declared_detached,
                                       "graph_connected": dependency is not None}
            if declared_detached and dependency is not None:
                detach_violations.append("declared_detach_boundary_violated:" + name)
        validation = dict(actual_backward=bool(loss.requires_grad), graph_connected=connected,
            detach_boundaries=detach_validation,
            reconstruction_validation=reconstruction_valid,
            equivalence_contract_verified=contract_valid, equivalence_contract=contract,
            source_verified=source_verified, executed_source_sha256=source_digest,
            reproduction_id=reproduction_id, independent_reproductions=0)
        evidence["runtime_validation"] = validation
        evidence["validation_level"] = "PRODUCTION_GRAPH_BACKWARD"
        verdict = result.verdict
        if detach_violations:
            if verdict != "NUMERICAL_ANOMALY_OBSERVED":
                verdict = "INSUFFICIENT_EVIDENCE"
            evidence["reasons"] = [*evidence["reasons"], *detach_violations]
        if not connected:
            if verdict != "NUMERICAL_ANOMALY_OBSERVED":
                verdict = "INSUFFICIENT_EVIDENCE"
            evidence["reasons"] = [*evidence["reasons"], "loss_not_connected_to_current_log_probs"]
        eligible = (verdict == "INSUFFICIENT_EVIDENCE" and
            evidence.get("matches_expected") is False and
            set(evidence["reasons"]).issubset({"unverified_coefficient_difference",
                "observed_loss_differs_from_oracle", "excluded_gradient_not_zero"}) and
            not evidence["numerical_events"] and connected and reconstruction_valid and contract_valid
            and source_verified and self.provenance["execution_kind"] == "FRAMEWORK_INTEGRATION"
            and self.provenance["synthetic_inputs"] is False)
        if eligible:
            key = sha256_json(_json_safe({"spec": asdict(spec), "inputs": evidence["inputs"],
                "provenance": self.provenance, "contract": contract,
                "observed": evidence["observed_coefficients"]}))
            reproductions = self._repetitions.setdefault(key, set())
            reproductions.add(reproduction_id)
            validation["independent_reproductions"] = len(reproductions)
            if len(reproductions) >= 2:
                verdict = "SEMANTIC_MISMATCH_CONFIRMED"
                evidence["reasons"] = ["repeated_source_bound_backward_violates_verified_contract"]
        evidence["limitations"] = [*evidence["limitations"],
            "Contract verifier and real-vs-synthetic input provenance are trusted integrator assertions.",
            "Source hash binds only the executed callable's file, not its transitive dependencies.",
            "Reproduction IDs must identify independent executions; independence is not remotely attested."]
        result = ObjectiveAuditResult(verdict, evidence)
        self._results.append(result)
        return result

    def report(self) -> dict[str, Any]:
        return {"schema_version": "quotientguard-objective-runtime-v1",
                "provenance": dict(self.provenance),
                "audits": [result.to_dict() for result in self._results]}
