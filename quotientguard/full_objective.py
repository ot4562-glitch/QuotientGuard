"""Explicit composite loss contracts. No framework imports or execution.

The scalar analytic engine and tensor replay are separate implementations.
Supported means this declared local contract was checked, never a Trainer or
natural-input assertion. Existing PPO APIs keep their original meaning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

from .objective import ObjectiveSpec, ObjectiveTrace, _matrix, _json_safe, _close, _supported, _flatten
from .provenance import sha256_json
from .importance_correction import ImportanceCorrectionSpec, correction_reasons


@dataclass(frozen=True)
class ReductionSpec:
    reduction: str = 'active_mean'
    denominator: float | None = None
    completion_length: int | None = None
    empty_mask_policy: str = 'undefined'
    denominator_epsilon: float = 0.
    denominator_scope: str = 'microbatch'  # explicit provenance, not inferred counts
    process_count: int = 1
    scale: float = 1.


@dataclass(frozen=True)
class KLSpec:
    estimator: str = 'k3'
    direction: str = 'reference_minus_current'
    coefficient: float = 0.
    detach_reference: bool = True
    delta_bounds: tuple[float, float] | None = None
    value_bounds: tuple[float, float] | None = None
    score_weight: bool = False
    score_detach_on_delta_clamp: bool = False
    importance_weight: str = 'unit'  # current_rollout: exp(kl_current-rollout)
    detach_rollout: bool = True
    weight_nonfinite: str = 'literal'  # zero matches explicit NeMo nan_to_num
    reduction: ReductionSpec | None = None


@dataclass(frozen=True)
class FullObjectiveSpec:
    policy: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    surrogate: str = 'ppo_clipped'
    coefficient_lower: float = .8  # absolute CISPO bounds, not epsilon offsets
    coefficient_upper: float = 1.2
    tau_positive: float = 1.
    tau_negative: float = 1.05
    ratio_delta_bounds: tuple[float, float] | None = None
    ratio_masking: str = 'inherit'  # select_delta acts after subtraction only
    kl: KLSpec = field(default_factory=KLSpec)
    entropy_coefficient: float = 0.
    entropy_reduction: ReductionSpec | None = None
    policy_reduction: ReductionSpec | None = None
    correction: str = 'none'  # weights are independent input coordinates
    correction_placement: str = 'outside_policy'
    output_masking: str = 'multiply'  # select is selection AFTER token arithmetic
    composition: str = 'after_reduction'  # or before_reduction, common reduction only
    importance_correction: ImportanceCorrectionSpec = field(default_factory=ImportanceCorrectionSpec)
    detach_rollout_policy: bool = True
    ratio_output_masking: str = 'none'  # multiply the sequence ratio by mask before surrogate

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        values['policy'] = ObjectiveSpec(**values.get('policy', {}))
        kl = dict(values.get('kl', {}))
        if kl.get('reduction') is not None:
            kl['reduction'] = ReductionSpec(**kl['reduction'])
        values['kl'] = KLSpec(**kl)
        values['importance_correction'] = ImportanceCorrectionSpec(**values.get('importance_correction',{}))
        for name in ('policy_reduction', 'entropy_reduction'):
            if values.get(name) is not None:
                values[name] = ReductionSpec(**values[name])
        return cls(**values)


@dataclass
class FullObjectiveTrace:
    policy: ObjectiveTrace
    reference_log_probs: Any = None
    kl_current_log_probs: Any = None
    rollout_log_probs: Any = None
    entropy: Any = None
    policy_filter: Any = None  # already computed, detached; denominator uses mask
    kl_mask: Any = None
    entropy_mask: Any = None
    observed_loss: Any = None
    observed_terms: dict[str, Any] = field(default_factory=dict)
    observed_gradients: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        values['policy'] = ObjectiveTrace(**values['policy'])
        return cls(**values)


@dataclass(frozen=True)
class FullObjectiveResult:
    evidence: dict[str, Any]

    def to_dict(self):
        payload = _json_safe(self.evidence)
        return {**payload, 'audit_sha256': sha256_json(payload)}


def reduction_for(spec, name):
    override = (spec.policy_reduction if name == 'policy' else spec.kl.reduction
                if name == 'kl' else spec.entropy_reduction)
    p = spec.policy
    return override or ReductionSpec(p.reduction, p.denominator, p.completion_length, p.empty_mask_policy)


def unsupported(spec):
    # Reuse only validation, not PPO math. Differentiable input boundaries are
    # implemented here; the legacy oracle intentionally still rejects them.
    from dataclasses import replace
    p = spec.policy
    reasons = _supported(replace(p, detach_old_policy=True, detach_advantage=True, detach_importance=True))
    if p.gradient_scale != 1:
        reasons.append('legacy_post_backward_gradient_scale_not_full_loss_scaling')
    if spec.surrogate not in {'ppo_clipped', 'cispo', 'sapo_smooth', 'sapo_detached'}:
        reasons.append('unsupported_surrogate')
    if spec.correction not in {'none', 'weights', 'gated_old_rollout', 'rollout_ratio'}:
        reasons.append('unsupported_correction_contract')
    if spec.correction=='rollout_ratio':
        if (spec.correction_placement!='inside_clip' or spec.surrogate!='ppo_clipped'
                or p.ratio_level!='sequence_masked_geometric_mean'):
            reasons.append('rollout_ratio_requires_inside_clip_sequence_mean_ppo')
    elif spec.correction_placement!='outside_policy':
        reasons.append('unsupported_correction_contract')
    if spec.ratio_output_masking not in {'none','multiply'}:
        reasons.append('unsupported_ratio_output_masking')
    if spec.ratio_output_masking=='multiply' and (spec.surrogate!='ppo_clipped' or p.ratio_level!='sequence_masked_geometric_mean'):
        reasons.append('masked_ratio_requires_sequence_mean_ppo')
    if spec.correction=='gated_old_rollout':
        reasons.extend(correction_reasons(spec.importance_correction))
        if spec.surrogate!='ppo_clipped' or p.ratio_level!='token':
            reasons.append('gated_old_rollout_requires_token_ppo')
        if not p.detach_importance or p.importance_level!='token':
            reasons.append('gated_old_rollout_requires_detached_token_coefficients')
    if spec.output_masking not in {'multiply', 'select'} or spec.composition not in {'after_reduction', 'before_reduction'}:
        reasons.append('unsupported_composition_or_masking')
    if spec.ratio_masking not in {'inherit', 'select_delta'}:
        reasons.append('unsupported_ratio_masking')
    if not (math.isfinite(spec.coefficient_lower) and math.isfinite(spec.coefficient_upper)
            and 0 <= spec.coefficient_lower <= spec.coefficient_upper):
        reasons.append('invalid_cispo_bounds')
    if any(not math.isfinite(t) or t <= 0 for t in (spec.tau_positive, spec.tau_negative)):
        reasons.append('invalid_sapo_temperature')
    k = spec.kl
    if k.estimator not in {'k1', 'k2', 'k3'} or k.direction not in {'reference_minus_current', 'current_minus_reference'}:
        reasons.append('unsupported_kl_formula')
    if k.importance_weight not in {'unit', 'current_rollout'} or k.weight_nonfinite not in {'literal', 'zero'}:
        reasons.append('unsupported_kl_importance_weight')
    if k.score_weight and k.importance_weight != 'unit':
        reasons.append('conflicting_kl_weight_paths')
    for bounds in (spec.ratio_delta_bounds, k.delta_bounds, k.value_bounds):
        if bounds is not None and (len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] >= bounds[1]):
            reasons.append('invalid_clamp_bounds')
    if not all(math.isfinite(v) for v in (k.coefficient, spec.entropy_coefficient)):
        reasons.append('invalid_auxiliary_coefficient')
    reductions = [reduction_for(spec, name) for name in ('policy', 'kl', 'entropy')]
    for r in reductions:
        if r.reduction not in {'sum', 'active_mean', 'explicit', 'per_sequence_active_mean', 'fixed_length_mean','nonempty_sequence_mean'}:
            reasons.append('unsupported_reduction')
        if r.empty_mask_policy not in {'zero', 'undefined'}:
            reasons.append('unsupported_empty_mask_policy')
        if r.reduction == 'explicit' and (r.denominator is None or not math.isfinite(r.denominator) or r.denominator <= 0):
            reasons.append('invalid_explicit_denominator')
        if r.reduction=='nonempty_sequence_mean' and r.denominator is not None and (
                not math.isfinite(r.denominator) or r.denominator<=0):
            reasons.append('invalid_sequence_denominator')
        if r.reduction == 'fixed_length_mean' and (type(r.completion_length) is not int or r.completion_length <= 0):
            reasons.append('invalid_completion_length')
        if (type(r.process_count) is not int or r.process_count <= 0 or not math.isfinite(r.scale)
                or not math.isfinite(r.denominator_epsilon) or r.denominator_epsilon < 0):
            reasons.append('invalid_reduction_scaling')
        if r.denominator_scope not in {'microbatch', 'batch', 'accumulation_window', 'global_batch'}:
            reasons.append('unsupported_denominator_scope')
        if r.denominator_scope != 'microbatch' and not (r.reduction=='explicit' or
                r.reduction=='nonempty_sequence_mean' and r.denominator is not None):
            reasons.append('nonlocal_denominator_requires_explicit_count')
    if spec.composition == 'before_reduction' and any(r != reductions[0] for r in reductions[1:]):
        reasons.append('before_reduction_requires_identical_reductions')
    return sorted(set(reasons))


def full_inputs(spec, trace):
    p = trace.policy
    names = {'current_log_probs': p.current_log_probs, 'old_log_probs': p.old_log_probs,
             'advantages': p.advantages, 'mask': p.mask}
    names.update({k: v for k, v in {'importance_weights': p.importance_weights,
                  'reference_log_probs': trace.reference_log_probs, 'entropy': trace.entropy,
                  'kl_current_log_probs': trace.kl_current_log_probs, 'rollout_log_probs': trace.rollout_log_probs,
                  'policy_filter': trace.policy_filter, 'kl_mask': trace.kl_mask,
                  'entropy_mask': trace.entropy_mask}.items() if v is not None})
    data = {k: _matrix(v, k) for k, v in names.items()}
    shape = (len(data['current_log_probs']), len(data['current_log_probs'][0]))
    for name, matrix in data.items():
        expected = (shape[0], 1) if name == 'importance_weights' and spec.policy.importance_level == 'sequence' else shape
        if (len(matrix), len(matrix[0])) != expected:
            raise ValueError(f'{name} shape differs from declared input shape')
        if name in {'mask', 'policy_filter', 'kl_mask', 'entropy_mask'} and any(v not in (0, 1) for row in matrix for v in row):
            raise ValueError(f'{name} must be binary')
    if spec.correction == 'weights' and 'importance_weights' not in data:
        raise ValueError('weights correction requires explicit importance_weights')
    if spec.correction in {'gated_old_rollout','rollout_ratio'} and 'rollout_log_probs' not in data:
        raise ValueError(spec.correction+' correction requires rollout_log_probs')
    if spec.kl.coefficient != 0 and 'reference_log_probs' not in data:
        raise ValueError('KL requires reference_log_probs')
    if spec.kl.coefficient != 0 and spec.kl.importance_weight == 'current_rollout' and 'rollout_log_probs' not in data:
        raise ValueError('current_rollout KL importance requires rollout_log_probs')
    if spec.entropy_coefficient != 0 and 'entropy' not in data:
        raise ValueError('entropy bonus requires entropy input')
    if any(v < 0 for row in data.get('importance_weights', []) for v in row):
        raise ValueError('importance weights must be nonnegative')
    if spec.composition == 'before_reduction':
        for name in ('kl_mask', 'entropy_mask'):
            if name in data and data[name] != data['mask']:
                raise ValueError('before_reduction requires identical term masks')
    return data


def input_states(spec, data, term=None):
    p = spec.policy
    states = {name: 'UNUSED' for name in data if name not in {'mask', 'policy_filter', 'kl_mask', 'entropy_mask'}}
    if term in (None, 'policy'):
        states['current_log_probs'] = 'CONNECTED'
        if spec.correction=='rollout_ratio':
            states['rollout_log_probs']='DETACHED' if spec.detach_rollout_policy else 'CONNECTED'
        else:
            states['old_log_probs'] = 'DETACHED' if (p.detach_old_policy or p.ratio_level == 'sequence_token'
                or spec.surrogate in {'cispo', 'sapo_detached'}) else 'CONNECTED'
        states['advantages'] = 'DETACHED' if p.detach_advantage else 'CONNECTED'
        if spec.correction == 'weights':
            states['importance_weights'] = 'DETACHED' if p.detach_importance else 'CONNECTED'
        elif spec.correction=='gated_old_rollout':
            states['rollout_log_probs']='DETACHED'
    if term in (None, 'kl') and spec.kl.coefficient != 0:
        states['kl_current_log_probs' if 'kl_current_log_probs' in data else 'current_log_probs'] = 'CONNECTED'
        states['reference_log_probs'] = 'DETACHED' if spec.kl.detach_reference else 'CONNECTED'
        if spec.kl.importance_weight == 'current_rollout':
            if states.get('rollout_log_probs')!='CONNECTED':
                states['rollout_log_probs'] = 'DETACHED' if spec.kl.detach_rollout else 'CONNECTED'
    if term in (None, 'entropy') and spec.entropy_coefficient != 0:
        states['entropy'] = 'CONNECTED'
    return states


def _observed_gradient_matches(value, expected, name, spec, data):
    if value is None or expected is None:
        return value is None and expected is None
    matrix = _matrix(value, 'observed_gradients.'+name)
    if len(matrix) != len(expected) or any(len(a) != len(b) for a,b in zip(matrix, expected)):
        return False
    mask = data['mask']
    active=[[False]*len(row) for row in mask]
    for term in ('policy','kl','entropy'):
        if input_states(spec,data,term).get(name)=='CONNECTED':
            term_mask=data.get(term+'_mask',mask) if term!='policy' else mask
            active=[[bool(v or term_mask[i][j]) for j,v in enumerate(row)] for i,row in enumerate(active)]
    if name == 'importance_weights' and spec.policy.importance_level == 'sequence':
        active = [[any(row)] for row in active]
    excluded_zero = all(v == 0 for i,row in enumerate(matrix) for j,v in enumerate(row) if not active[i][j])
    return excluded_zero and _close(matrix, expected, spec.policy)


def audit_full_objective(spec: FullObjectiveSpec, trace: FullObjectiveTrace, *, device='cpu'):
    """Compare hand derivatives with a homogeneous-dtype autograd reference.

    Input gradients are partial derivatives in the declared coordinates.
    A supplied entropy tensor or correction weight does not specify its upstream
    logits/temperature/sampling processor. Native callers must bind that graph.
    """
    from .full_objective_analytic import analytic_full_objective
    reasons = unsupported(spec)
    result = {'schema_version': 'quotientguard-full-objective-v1', 'spec': asdict(spec),
              'status': 'UNSUPPORTED', 'reasons': reasons, 'analytic': None, 'oracle': None,
              'execution': 'REFERENCE_REPLAY', 'semantic_confirmation_eligible': False,
              'trainer_executed': False, 'optimizer_step_executed': False,
              'natural_input_reachability': 'UNVERIFIED',
              'limitations': ['Partial derivatives in explicit input coordinates; no inferred logits/parameter Jacobian.',
                  'Homogeneous dtype replay; no inferred autocast, GradScaler, distributed runtime or optimizer.',
                  'Correction weights do not attest rollout logprob, temperature or sampling-processor provenance.']}
    if reasons:
        return FullObjectiveResult(result)
    data = full_inputs(spec, trace)
    from .full_objective_torch import full_objective_oracle
    analytic = analytic_full_objective(spec, data)
    replay = full_objective_oracle(spec, trace, device=device)
    result.update(analytic=analytic, oracle=replay)
    mismatches = []
    if replay['loss_finite'] and replay['gradients_finite']:
        if spec.correction=='gated_old_rollout':
            for name in ('coefficients','keep'):
                active=([[any(row)] for row in data['mask']] if name=='keep' and spec.importance_correction.level=='sequence'
                        else data['mask'])
                expected=[v for i,row in enumerate(analytic['correction'][name]) for j,v in enumerate(row) if active[i][j]]
                actual=[v for i,row in enumerate(replay['correction'][name]) for j,v in enumerate(row) if active[i][j]]
                if not _close(expected,actual,spec.policy):
                    mismatches.append('correction.'+name)
        for term in ('policy', 'kl', 'entropy', 'total'):
            if not _close(analytic['terms'][term], replay['terms'][term], spec.policy):
                mismatches.append(f'{term}.loss')
            ag, tg = analytic['term_gradients'][term], replay['term_gradients'][term]
            for name in ag:
                if ag[name]['state'] != tg[name]['state'] or (ag[name]['values'] is not None and
                        not _close(ag[name]['values'], tg[name]['values'], spec.policy)):
                    mismatches.append(f'{term}.gradient.{name}')
        if trace.observed_loss is not None and not _close(trace.observed_loss, replay['loss'], spec.policy):
            mismatches.append('observed.loss')
        if trace.policy.observed_loss is not None and not _close(trace.policy.observed_loss, replay['terms']['policy'], spec.policy):
            mismatches.append('observed.policy.loss')
        if trace.policy.observed_coefficients is not None and not _observed_gradient_matches(
                trace.policy.observed_coefficients, replay['term_gradients']['policy']['current_log_probs']['values'],
                'current_log_probs', spec, {**data, 'kl_mask': [[0]*len(row) for row in data['mask']]}):
            mismatches.append('observed.policy.gradient.current_log_probs')
        for name, value in trace.observed_terms.items():
            if name not in replay['terms'] or not _close(value, replay['terms'][name], spec.policy):
                mismatches.append(f'observed.term.{name}')
        for name, value in trace.observed_gradients.items():
            expected = replay['gradients'].get(name)
            if (expected is None or (value is None) != (expected['values'] is None)
                    or value is not None and not _observed_gradient_matches(value, expected['values'], name, spec, data)):
                mismatches.append(f'observed.gradient.{name}')
    result['mismatches'] = mismatches
    observed_nonfinite = any(not math.isfinite(v) for value in
        [trace.observed_loss, trace.policy.observed_loss, trace.policy.observed_coefficients,
         *trace.observed_terms.values(), *trace.observed_gradients.values()] for v in _flatten(value))
    correction_nonfinite_changed_gate = (any(name.startswith('correction.') for name in mismatches)
        and any(event['stage'].startswith('correction.') for event in replay['numerical_events']))
    if observed_nonfinite or not replay['loss_finite'] or not replay['gradients_finite'] or correction_nonfinite_changed_gate:
        result['status'] = 'NUMERICAL_ANOMALY_OBSERVED'
        if observed_nonfinite:
            result['reasons'] = ['nonfinite_supplied_observation']
        elif correction_nonfinite_changed_gate:
            result['reasons'] = ['nonfinite_correction_changed_coefficient']
    elif analytic['boundary_reasons']:
        result.update(status='INSUFFICIENT_EVIDENCE', reasons=analytic['boundary_reasons'])
    elif mismatches:
        result['status'] = 'CONTRACT_MISMATCH_OBSERVED'
    elif not replay['excluded_gradients_zero']:
        result['status'] = 'CONTRACT_MISMATCH_OBSERVED'
        result['reasons'] = ['nonzero_excluded_current_gradient']
    else:
        result['status'] = 'SUPPORTED'
    return FullObjectiveResult(result)
