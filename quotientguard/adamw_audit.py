"""Read-only numerical and state-lineage checks for one explicit AdamW step."""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from .native_validation import _seal


_EXECUTION = {
    "optimizer": "torch.optim.AdamW",
    "device": "cpu",
    "dtype": "float64",
    "layout": "dense",
    "fused": False,
    "foreach": False,
    "amsgrad": False,
    "maximize": False,
    "capturable": False,
    "differentiable": False,
    "mixed_precision": False,
    "sharded": False,
    "step_executed": True,
}
_HYPERPARAMETERS = ("lr", "beta1", "beta2", "eps", "weight_decay")
_ATOL = 1e-15
_RTOL = 1e-12


class _EvidenceError(ValueError):
    def __init__(self, status: str, reason: str):
        self.status = status
        super().__init__(reason)


def _mapping(value, label):
    if value is None:
        raise _EvidenceError("INCONCLUSIVE", f"missing_{label}")
    if not isinstance(value, Mapping):
        raise _EvidenceError("INVALID_EVIDENCE", f"invalid_{label}")
    return value


def _number(value, label):
    if type(value) not in (int, float):
        raise _EvidenceError("INVALID_EVIDENCE", f"invalid_{label}")
    try:
        result = float(value)
    except OverflowError:
        result = math.inf
    if not math.isfinite(result):
        raise _EvidenceError("INVALID_EVIDENCE", f"nonfinite_{label}")
    if type(value) is int and int(result) != value:
        raise _EvidenceError("INVALID_EVIDENCE", f"inexact_float64_{label}")
    return result


def _vector(value, size, label):
    if value is None:
        raise _EvidenceError("INCONCLUSIVE", f"missing_{label}")
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise _EvidenceError("INVALID_EVIDENCE", f"wrong_length_{label}")
    return [_number(item, label) for item in value]


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _binding(value):
    binding = _mapping(value, "binding")
    for field in ("run_id", "step_id", "parameter_id"):
        if not _nonempty(binding.get(field)):
            raise _EvidenceError("INCONCLUSIVE", f"missing_binding_{field}")
    shape = binding.get("shape")
    if shape is None:
        raise _EvidenceError("INCONCLUSIVE", "missing_shape")
    if not isinstance(shape, (tuple, list)) or any(type(n) is not int or n <= 0 for n in shape):
        raise _EvidenceError("INVALID_EVIDENCE", "invalid_shape")
    return {**{field: binding[field] for field in ("run_id", "step_id", "parameter_id")},
            "shape": list(shape)}, math.prod(shape)


def _bound_record(trace, name, binding, stage):
    record = _mapping(trace.get(name), name)
    try:
        observed, _ = _binding(record.get("binding"))
    except _EvidenceError as exc:
        raise _EvidenceError(exc.status, f"{name}_{exc}") from exc
    if observed != binding:
        raise _EvidenceError("INCONCLUSIVE", f"{name}_binding_mismatch")
    if record.get("stage") != stage or not _nonempty(record.get("source")) or record.get("verified") is not True:
        raise _EvidenceError("INCONCLUSIVE", f"{name}_provenance_unverified")
    return record


def _settings(value):
    settings = _mapping(value, "hyperparameters")
    if set(settings) - set(_HYPERPARAMETERS):
        raise _EvidenceError("UNSUPPORTED", "unknown_hyperparameter_fields")
    if any(field not in settings for field in _HYPERPARAMETERS):
        raise _EvidenceError("INCONCLUSIVE", "missing_hyperparameters")
    scalars = {field: _number(settings[field], field) for field in _HYPERPARAMETERS}
    if (any(scalars[key] < 0 for key in ("lr", "eps", "weight_decay")) or
            any(not 0 <= scalars[key] < 1 for key in ("beta1", "beta2"))):
        raise _EvidenceError("INVALID_EVIDENCE", "hyperparameters_out_of_range")
    return scalars


def _snapshot(record, size):
    parameters = _vector(record.get("parameters"), size, "parameters")
    state = _mapping(record.get("state"), "state")
    if state.get("kind") == "uninitialized":
        if set(state) != {"kind"}:
            raise _EvidenceError("INVALID_EVIDENCE", "uninitialized_state_has_moments_or_step")
        return {"parameters": parameters, "state": {"kind": "uninitialized"}}
    if state.get("kind") != "initialized":
        raise _EvidenceError("INCONCLUSIVE", "state_initialization_unknown")
    step = state.get("step")
    if type(step) is not int or step < 0:
        raise _EvidenceError("INVALID_EVIDENCE", "invalid_step")
    first = _vector(state.get("exp_avg"), size, "exp_avg")
    second = _vector(state.get("exp_avg_sq"), size, "exp_avg_sq")
    if any(moment < 0 for moment in second):
        raise _EvidenceError("INVALID_EVIDENCE", "negative_second_moment")
    return {"parameters": parameters, "state": {
        "kind": "initialized", "step": step, "exp_avg": first, "exp_avg_sq": second,
    }}


def _expected(before, gradient, settings, step_dtype):
    if gradient is None:
        return before, None
    state = before["state"]
    step = state.get("step", 0)
    if step >= 2 ** (24 if step_dtype == "float32" else 53):
        raise _EvidenceError("UNSUPPORTED", "step_counter_increment_not_exact")
    step += 1
    beta1, beta2 = settings["beta1"], settings["beta2"]
    lr, eps, decay = settings["lr"], settings["eps"], settings["weight_decay"]
    bias1, bias2 = 1 - beta1**step, 1 - beta2**step
    first = state.get("exp_avg", [0.0] * len(gradient))
    second = state.get("exp_avg_sq", [0.0] * len(gradient))
    weight = 1 - beta1
    result = {"kind": "initialized", "step": step, "exp_avg": [], "exp_avg_sq": []}
    parameters = []
    for p, m, v, g in zip(before["parameters"], first, second, gradient):
        # Evaluate the interpolation from the closer endpoint.
        m_next = m + weight * (g - m) if weight < 0.5 else g - (1 - weight) * (g - m)
        v_next = beta2 * v + ((1 - beta2) * g) * g
        denominator = math.sqrt(v_next) / math.sqrt(bias2) + eps
        if not (math.isfinite(m_next) and math.isfinite(denominator)):
            raise _EvidenceError("INCONCLUSIVE", "nonfinite_reference")
        if denominator == 0:
            raise _EvidenceError("INCONCLUSIVE", "zero_adamw_denominator")
        p_next = p * (1 - lr * decay) - (lr / bias1) * m_next / denominator
        if not math.isfinite(p_next):
            raise _EvidenceError("INCONCLUSIVE", "nonfinite_reference")
        result["exp_avg"].append(m_next)
        result["exp_avg_sq"].append(v_next)
        parameters.append(p_next)
    return {"parameters": parameters, "state": result}, {"first": bias1, "second": bias2}


def _compare(actual, expected, label, exact):
    """Keep the exact discrete checks separate from the numerical policy."""
    if isinstance(expected, Mapping):
        differences = []
        for key in sorted(actual.keys() | expected.keys()):
            name = f"{label}.{key}" if label else key
            if key not in actual or key not in expected:
                differences.append({"field": name, "observed": actual.get(key),
                                    "expected": expected.get(key)})
            else:
                differences.extend(_compare(actual[key], expected[key], name, exact))
        return differences
    if isinstance(expected, list):
        differences = []
        for index, (observed, wanted) in enumerate(zip(actual, expected, strict=True)):
            matches = observed == wanted if exact else math.isclose(
                observed, wanted, abs_tol=_ATOL, rel_tol=_RTOL)
            if not matches:
                differences.append({"field": f"{label}[{index}]", "observed": observed,
                                    "expected": wanted, "absolute_error": abs(observed - wanted)})
        return differences
    numerical = type(expected) is float
    matches = (math.isclose(actual, expected, abs_tol=_ATOL, rel_tol=_RTOL)
               if numerical and not exact else actual == expected)
    if matches:
        return []
    error = {"field": label, "observed": actual, "expected": expected}
    if numerical or type(expected) is int:
        error["absolute_error"] = abs(actual - expected)
    return [error]


def _verdict(boundary, actual, expected, *, exact=False, **details):
    errors = _compare(actual, expected, "", exact)
    return {"boundary": boundary, "status": "MISMATCH" if errors else "MATCH",
            "reason": "values_disagree" if errors else "values_agree",
            "observed": actual, "expected": expected, "errors": errors, **details}


def _execution(trace):
    execution = _mapping(trace.get("execution"), "execution")
    extra = set(execution) - {*_EXECUTION, "step_dtype", "torch_version"}
    if extra:
        raise _EvidenceError("UNSUPPORTED", "unknown_execution_fields")
    for key, supported in _EXECUTION.items():
        if key not in execution:
            raise _EvidenceError("INCONCLUSIVE", f"missing_execution_{key}")
        if type(execution[key]) is not type(supported) or execution[key] != supported:
            raise _EvidenceError("UNSUPPORTED", f"unsupported_execution_{key}")
    if execution.get("step_dtype") not in ("float32", "float64"):
        status = "UNSUPPORTED" if "step_dtype" in execution else "INCONCLUSIVE"
        raise _EvidenceError(status, "step_dtype_unavailable_or_unsupported")
    if not _nonempty(execution.get("torch_version")):
        raise _EvidenceError("INCONCLUSIVE", "missing_torch_version")
    return dict(execution)


def _issue(boundary, status, reason):
    return {"boundary": boundary, "status": status, "reason": reason,
            "observed": None, "expected": None, "errors": []}


def audit_adamw_step(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Audit one supplied step. Provenance flags are evidence-provider assertions.

    An absent gradient must be explicitly observed. Missing gradient evidence is
    inconclusive. See docs/ADAMW_AUDIT.md for the versioned trace contract.
    """
    report = {
        "schema_version": "quotientguard-adamw-audit-v1",
        "execution_scope": "SUPPLIED_OPTIMIZER_STEP",
        "optimizer_step_executed_by_auditor": False,
        "pipeline_verified": False,
        "supported_path": {**_EXECUTION, "parameter_tensor_count": 1},
        "tolerance": {"absolute": _ATOL, "relative": _RTOL},
    }
    try:
        trace = _mapping(trace, "trace")
        report["execution"] = _execution(trace)
        binding, size = _binding(trace.get("binding"))
        report["binding"] = binding
        report["compared_coordinates"] = size
        consumed = _bound_record(trace, "consumed", binding, "before_step")
        before = _snapshot(consumed, size)
        settings = _settings(consumed.get("hyperparameters"))
    except _EvidenceError as exc:
        report.update(status=exc.status, reason=str(exc),
                      lineage=_issue("state_lineage", exc.status, str(exc)),
                      transition=_issue("adamw_transition", exc.status, str(exc)))
        return _seal(report)

    try:
        authorized = _bound_record(trace, "authorized", binding, "authorized")
        want = _snapshot(authorized, size)
        report["lineage"] = _verdict(
            "state_lineage", {**before, "hyperparameters": settings},
            {**want, "hyperparameters": _settings(authorized.get("hyperparameters"))},
            exact=True, sources={"authorized": authorized["source"], "consumed": consumed["source"]},
        )
    except _EvidenceError as exc:
        report["lineage"] = _issue("state_lineage", exc.status, str(exc))

    try:
        evidence = _bound_record(trace, "gradient", binding, "consumed_gradient")
        kind = evidence.get("kind")
        if kind == "dense":
            gradient = _vector(evidence.get("values"), size, "gradient_values")
        elif kind == "none" and "values" not in evidence:
            gradient = None
        else:
            status = "UNSUPPORTED" if kind == "sparse" else "INCONCLUSIVE"
            raise _EvidenceError(status, "gradient_kind_unavailable_or_unsupported")
        observed = _bound_record(trace, "after", binding, "after_step")
        expected, bias = _expected(before, gradient, settings, report["execution"]["step_dtype"])
        report["transition"] = _verdict(
            "adamw_transition", _snapshot(observed, size), expected, gradient=gradient,
            gradient_kind=kind, bias_correction=bias, hyperparameters=settings,
            sources={"consumed": consumed["source"], "gradient": evidence["source"], "after": observed["source"]},
        )
    except _EvidenceError as exc:
        report["transition"] = _issue("adamw_transition", exc.status, str(exc))

    lineage, transition = report["lineage"], report["transition"]
    reasons = [item["boundary"] + "_mismatch" for item in (lineage, transition)
               if item["status"] == "MISMATCH"]
    if reasons:
        report.update(status="MISMATCH", reason="+".join(reasons))
    elif lineage["status"] == transition["status"] == "MATCH":
        report.update(status="MATCH", reason="both_optimizer_contracts_agree")
    else:
        issue = next(item for status in ("INVALID_EVIDENCE", "UNSUPPORTED", "INCONCLUSIVE")
                     for item in (lineage, transition) if item["status"] == status)
        report.update(status=issue["status"], reason=issue["reason"])
    return _seal(report)
