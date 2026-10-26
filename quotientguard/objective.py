"""Independent PPO derivative interpretation; does not alter v1 certificates.

Coefficients are d(negative surrogate loss)/d(current log probability).
PyTorch is imported only when an oracle or runtime backward is requested.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

from .provenance import sha256_json


@dataclass(frozen=True)
class ObjectiveSpec:
    objective: str = "ppo_clipped"
    clip_lower: float = 0.8
    clip_upper: float = 1.2
    ratio_level: str = "token"  # or exp(sum(active log ratios)) per sequence
    importance_level: str = "token"  # detached correction, outside clipping
    reduction: str = "active_mean"  # sum, active_mean, explicit
    denominator: float | None = None
    detach_old_policy: bool = True
    detach_advantage: bool = True
    detach_importance: bool = True
    dtype: str = "float32"
    gradient_scale: float = 1.0  # explicitly measured uniform post-backward scale
    atol: float | None = None
    rtol: float | None = None


@dataclass
class ObjectiveTrace:
    current_log_probs: Any
    old_log_probs: Any
    advantages: Any
    mask: Any
    importance_weights: Any = None
    observed_coefficients: Any = None
    observed_loss: Any = None
    parameter_gradients: Any = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveAuditResult:
    verdict: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe({"schema_version": "quotientguard-objective-audit-v1",
                             "verdict": self.verdict, **self.evidence})
        return {**payload, "audit_sha256": sha256_json(payload)}


def _plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _json_safe(value: Any) -> Any:
    value = _plain(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _matrix(value: Any, name: str) -> list[list[float]]:
    value = _plain(value)
    if not isinstance(value, list) or not value or not all(isinstance(r, list) for r in value):
        raise ValueError(f"{name} must be a nonempty rectangular [sequence, token] matrix")
    width = len(value[0])
    if not width or any(len(row) != width for row in value):
        raise ValueError(f"{name} must be rectangular and nonempty")
    return [[float(v) for v in row] for row in value]


def _flatten(value: Any):
    value = _plain(value)
    if isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    elif value is not None:
        yield float(value)


def _exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _inputs(spec: ObjectiveSpec, trace: ObjectiveTrace):
    current = _matrix(trace.current_log_probs, "current_log_probs")
    rows, width = len(current), len(current[0])
    data = [current]
    for name in ("old_log_probs", "advantages", "mask"):
        matrix = _matrix(getattr(trace, name), name)
        if len(matrix) != rows or any(len(r) != width for r in matrix):
            raise ValueError(f"{name} shape differs from current_log_probs")
        data.append(matrix)
    if any(v not in (0., 1.) for v in _flatten(data[3])):
        raise ValueError("mask must be binary")
    expected_width = 1 if spec.importance_level == "sequence" else width
    weights = ([[1.] * expected_width for _ in current] if trace.importance_weights is None
               else _matrix(trace.importance_weights, "importance_weights"))
    if len(weights) != rows or any(len(r) != expected_width for r in weights):
        raise ValueError("importance_weights shape must match declared importance_level")
    if spec.importance_level == "sequence":
        weights = [[r[0]] * width for r in weights]
    if any(v < 0 for v in _flatten(weights)):
        raise ValueError("importance weights must be nonnegative")
    denominator = (1. if spec.reduction == "sum" else spec.denominator
                   if spec.reduction == "explicit" else sum(_flatten(data[3])))
    return (*data, weights, denominator)


def _supported(spec: ObjectiveSpec) -> list[str]:
    reasons = []
    if spec.objective != "ppo_clipped":
        reasons.append("unsupported_objective")
    if spec.ratio_level not in {"token", "sequence_product"}:
        reasons.append("unsupported_ratio_level")
    if spec.importance_level not in {"token", "sequence"}:
        reasons.append("unsupported_importance_level")
    if spec.reduction not in {"sum", "active_mean", "explicit"}:
        reasons.append("unsupported_reduction")
    if not all((spec.detach_old_policy, spec.detach_advantage, spec.detach_importance)):
        reasons.append("unsupported_stop_gradient_boundary")
    if spec.dtype not in {"float16", "bfloat16", "float32", "float64"}:
        reasons.append("unsupported_dtype")
    if not (math.isfinite(spec.clip_lower) and math.isfinite(spec.clip_upper)
            and 0 < spec.clip_lower <= 1 <= spec.clip_upper):
        reasons.append("invalid_clip_bounds")
    if not math.isfinite(spec.gradient_scale) or not 0 <= spec.gradient_scale <= 1:
        reasons.append("invalid_uniform_gradient_scale")
    for tolerance in (spec.atol, spec.rtol):
        if tolerance is not None and (not math.isfinite(tolerance) or tolerance < 0):
            reasons.append("invalid_tolerance")
    return reasons


def _tolerances(spec):
    default = {"float16": .005, "bfloat16": .04, "float32": 2e-5, "float64": 1e-10}[spec.dtype]
    return (default if spec.atol is None else spec.atol,
            default if spec.rtol is None else spec.rtol)


def _close(a, b, spec):
    aa, bb = list(_flatten(a)), list(_flatten(b))
    atol, rtol = _tolerances(spec)
    return len(aa) == len(bb) and all(math.isfinite(x) and math.isfinite(y)
        and math.isclose(x, y, abs_tol=atol, rel_tol=rtol) for x, y in zip(aa, bb))


def audit_objective(spec: ObjectiveSpec, trace: ObjectiveTrace, *, oracle: bool = False) -> ObjectiveAuditResult:
    """Compare explicit observations with analytic PPO, optionally replaying torch.

    Manual provenance never authorizes semantic confirmation. Use
    ObjectiveRuntimeCapture for repeated, verified production backward evidence.
    """
    data = {"spec": asdict(spec), "provenance": trace.provenance,
            "coefficient_definition": "d_negative_surrogate_loss_d_current_log_prob",
            "validation_level": "MANUAL_OBSERVATION", "reasons": [],
            "limitations": ["No downstream quality or population claim.",
                "Only the declared surrogate and constant detach boundaries are supported.",
                "Uniform gradient_scale needs independent measurement; parameter clipping is not inferred."]}
    observed = _plain(trace.observed_coefficients)
    data["observed_coefficients"] = observed
    numerical = []
    for name in ("observed_coefficients", "observed_loss", "parameter_gradients"):
        value = getattr(trace, name)
        data[name] = _plain(value)
        if any(not math.isfinite(v) for v in _flatten(value)):
            numerical.append({"stage": name, "execution": "SUPPLIED_OBSERVATION"})
    reasons = _supported(spec)
    if reasons:
        data["reasons"] = reasons
        data["numerical_events"] = numerical
        return ObjectiveAuditResult("NUMERICAL_ANOMALY_OBSERVED" if numerical else "INSUFFICIENT_EVIDENCE", data)
    current, old, adv, mask, weights, denominator = _inputs(spec, trace)
    for name, values in (("current_log_probs", current), ("old_log_probs", old),
                         ("advantages", adv), ("importance_weights", weights)):
        if any(not math.isfinite(v) for v in _flatten(values)):
            numerical.append({"stage": name, "kind": "NONFINITE_INPUT",
                              "execution": "SUPPLIED_INPUT", "loss_path_unproven": not oracle})
    data.update({"inputs": {"current_log_probs": current, "old_log_probs": old,
        "advantages": adv, "mask": mask, "importance_weights": weights},
        "normalization_denominator": denominator,
        "advantage_signs": [[0 if a == 0 else 1 if a > 0 else -1 for a in row] for row in adv]})
    for name in ("current_log_probs", "old_log_probs", "advantages", "mask", "importance_weights"):
        tensor = getattr(trace, name)
        if hasattr(tensor, "dtype") and str(tensor.dtype) != "torch." + spec.dtype:
            # Boolean/integer masks do not change the floating computation precision.
            if name != "mask":
                reasons.append("tensor_dtype_differs_from_declared_precision:" + name)
    if denominator is None or not math.isfinite(denominator) or denominator <= 0:
        reasons.append("undefined_normalization_denominator")
    expected, branches, ratios = [], [], []
    maximum = {"float16": 65504., "bfloat16": 3.3895313892515355e38,
               "float32": 3.4028234663852886e38,
               "float64": 1.7976931348623157e308}[spec.dtype]
    minimum = {"float16": 2.**-24, "bfloat16": 2.**-133,
               "float32": 2.**-149, "float64": 2.**-1074}[spec.dtype]
    range_risk = False
    if not oracle:
        range_risk = any(abs(v) > maximum or 0 < abs(v) < minimum
                         for values in (current, old, adv, weights) for v in _flatten(values))
    absolute_surrogate_sum = 0.
    if not reasons:
        for xs, os, ads, ms, ws in zip(current, old, adv, mask, weights):
            log_ratios = [x - o for x, o in zip(xs, os)]
            if spec.ratio_level == "sequence_product":
                # Deliberately literal multiply: invalid padded arithmetic is not hidden.
                ratio = _exp(sum(v * m for v, m in zip(log_ratios, ms)))
                rs = [ratio] * len(xs)
            else:
                rs = [_exp(v) for v in log_ratios]
            row, bs = [], []
            for r, a, m, w in zip(rs, ads, ms, ws):
                if not oracle and (not math.isfinite(r) or r == 0 or abs(r) > maximum
                        or 0 < abs(r) < minimum
                        or abs(a) > maximum or abs(w) > maximum
                        or abs(r*a) > maximum or abs(r*a*w) > maximum
                        or 0 < abs(r*a) < minimum or 0 < abs(r*a*w) < minimum):
                    range_risk = True
                clipped_ratio = min(max(r, spec.clip_lower), spec.clip_upper)
                absolute_surrogate_sum += abs(min(r*a, clipped_ratio*a) * w * m)
                epsilon = {"float16": 2**-10, "bfloat16": 2**-7,
                           "float32": 2**-23, "float64": 2**-52}[spec.dtype]
                boundary = spec.clip_upper if a > 0 else spec.clip_lower
                if m and a != 0 and math.isfinite(r) and abs(r-boundary) <= 4*epsilon*max(1., abs(boundary)):
                    if "clip_boundary_subgradient_ambiguous" not in reasons:
                        reasons.append("clip_boundary_subgradient_ambiguous")
                branch = ("masked" if not m else "zero_advantage" if a == 0 else
                    "upper_clipped" if a > 0 and r > spec.clip_upper else
                    "lower_clipped" if a < 0 and r < spec.clip_lower else "active")
                # Boundary derivative is provisional; interpretation fails closed above.
                value = (-a * r * w / denominator * spec.gradient_scale
                         if branch == "active" else 0.)
                row.append(value)
                bs.append(branch)
            if spec.ratio_level == "sequence_product":
                row = [sum(row) * m for m in ms]
            expected.append(row)
            branches.append(bs)
            ratios.append(rs)
    if not oracle and (range_risk or absolute_surrogate_sum > maximum):
        reasons.append("numerical_range_requires_execution")
    data.update({"expected_coefficients": expected, "clip_branches": branches,
                 "importance_ratios": ratios})
    replay_allowed = all(reason in {"clip_boundary_subgradient_ambiguous",
        "undefined_normalization_denominator"} for reason in reasons)
    if oracle and replay_allowed and denominator is not None and math.isfinite(denominator):
        from .objective_torch import ppo_oracle
        replay = ppo_oracle(spec, trace)
        data["oracle"] = replay
        data["validation_level"] = "INDEPENDENT_TORCH_REPLAY"
        numerical.extend(replay["numerical_events"])
        if expected and not _close(expected, replay["coefficients"], spec) and not numerical:
            reasons.append("analytic_oracle_disagreement")
    if observed is not None and expected:
        obs = _matrix(observed, "observed_coefficients")
        if len(obs) != len(expected) or any(len(a) != len(b) for a, b in zip(obs, expected)):
            raise ValueError("observed_coefficients shape differs from current_log_probs")
        data["coefficient_differences"] = [[o-e for o, e in zip(os, es)] for os, es in zip(obs, expected)]
        data["matches_expected"] = _close(expected, observed, spec)
        data["excluded_gradients_zero"] = all(o == 0 for os, ms in zip(obs, mask)
                                                for o, m in zip(os, ms) if m == 0)
        if not data["excluded_gradients_zero"]:
            reasons.append("excluded_gradient_not_zero")
    if "oracle" in data and trace.observed_loss is not None:
        data["loss_matches_oracle"] = _close(trace.observed_loss, data["oracle"]["loss"], spec)
        if not data["loss_matches_oracle"]:
            reasons.append("observed_loss_differs_from_oracle")
    executed_or_observed_anomaly = any(event["execution"] != "SUPPLIED_INPUT" for event in numerical)
    if numerical and not executed_or_observed_anomaly:
        reasons.append("nonfinite_input_loss_path_unproven")
    if executed_or_observed_anomaly:
        verdict = "NUMERICAL_ANOMALY_OBSERVED"
    elif reasons or not data.get("matches_expected", False):
        verdict = "INSUFFICIENT_EVIDENCE"
        if observed is None:
            reasons.append("no_production_coefficient_observation")
        elif not data.get("matches_expected", False):
            reasons.append("unverified_coefficient_difference")
    else:
        verdict = "EXPECTED_OBJECTIVE_EFFECT"
    data.update({"reasons": reasons, "numerical_events": numerical})
    return ObjectiveAuditResult(verdict, data)


def objective_report(result: ObjectiveAuditResult | dict[str, Any]) -> str:
    payload = result.to_dict() if isinstance(result, ObjectiveAuditResult) else result
    return "\n".join(["# QuotientGuard objective audit", "",
        "Verdict: " + payload["verdict"], "",
        "Coefficient: d(negative PPO surrogate loss)/d(current log probability).", "",
        "Validation: " + payload.get("validation_level", "unspecified"),
        "Expected coefficients: " + str(payload.get("expected_coefficients")),
        "Observed coefficients: " + str(payload.get("observed_coefficients")),
        "Clip branches: " + str(payload.get("clip_branches")),
        "Numerical events: " + str(payload.get("numerical_events", [])),
        "Reasons: " + str(payload.get("reasons", [])), "",
        *payload.get("limitations", []), ""])
