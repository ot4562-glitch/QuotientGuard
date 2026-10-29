"""Scalar float64 closed-form derivatives; does not import Torch.

The Jacobian is accumulated by row rather than reconstructing a tensor graph.
Detach means removing a Jacobian path, not setting an observed gradient to zero.
"""
import math

from .objective import _exp
from .full_objective import input_states, reduction_for


def _divide(x, y):
    return x / y if y else (math.nan if x == 0 else math.copysign(math.inf, x))


def _clamp(x, bounds):
    return x if bounds is None or math.isnan(x) else min(bounds[1], max(bounds[0], x))


def _interior(x, bounds):
    return bounds is None or bounds[0] <= x <= bounds[1]


def _sigmoid(x):
    return 1 / (1 + _exp(-x)) if x >= 0 else _exp(x) / (1 + _exp(x))


def _row_divisors(r, mask):
    b = len(mask)
    counts = [sum(row) for row in mask]
    if r.reduction == 'per_sequence_active_mean':
        return [b * ((max(1, n) if r.empty_mask_policy == 'zero' else n) + r.denominator_epsilon) for n in counts]
    if r.reduction=='nonempty_sequence_mean':
        outer=r.denominator if r.denominator is not None else sum(n>0 for n in counts)
        if r.empty_mask_policy=='zero' and r.denominator is None:outer=max(1,outer)
        return [outer*((max(1,n) if r.empty_mask_policy=='zero' else n)+r.denominator_epsilon) for n in counts]
    denominator = (1. if r.reduction == 'sum' else r.denominator if r.reduction == 'explicit'
                   else b * r.completion_length if r.reduction == 'fixed_length_mean' else sum(counts))
    if r.reduction == 'active_mean' and r.empty_mask_policy == 'zero':
        denominator = max(1, denominator)
    return [denominator + r.denominator_epsilon] * b


def analytic_full_objective(spec, data):
    p, k = spec.policy, spec.kl
    mask = data['mask']
    b, t = len(mask), len(mask[0])
    policy_filter = data.get('policy_filter')
    enabled = {'policy': True, 'kl': k.coefficient != 0, 'entropy': spec.entropy_coefficient != 0}
    term_values, term_gradients = {}, {}
    boundaries = set()
    eps = {'float16': 2**-10, 'bfloat16': 2**-7, 'float32': 2**-23, 'float64': 2**-52}[p.dtype]
    correction=None
    if spec.correction=='gated_old_rollout':
        from .importance_correction import scalar_correction
        correction=scalar_correction(spec.importance_correction,data['old_log_probs'],data['rollout_log_probs'],mask)
        for i,row in enumerate(correction['statistic']):
            if any(math.isfinite(value) and abs(value-bound)<=4*eps*max(1,abs(bound))
                   and (any(mask[i]) if spec.importance_correction.level=='sequence' else bool(mask[i][j]))
                   and not (bound_index==0 and value==0 and bound==0 and spec.importance_correction.gating!='ratio')
                   for j,value in enumerate(row) for bound_index,bound in enumerate(
                       (spec.importance_correction.lower,spec.importance_correction.upper))):
                boundaries.add('correction_gate_boundary')

    def near(value, bounds, reason):
        if bounds is not None and any(abs(value - v) <= 4*eps*max(1, abs(v)) for v in bounds):
            boundaries.add(reason)

    for term in ('policy', 'kl', 'entropy'):
        states = input_states(spec, data, term)
        grads = {name: [[0. for _ in row] for row in data[name]] for name, state in states.items() if state == 'CONNECTED'}
        term_mask = data.get(term + '_mask', mask) if term != 'policy' else mask
        r = reduction_for(spec, term)
        divisors = _row_divisors(r, term_mask)
        total = 0.
        if enabled[term]:
            for i in range(b):
                selected = p.masking == 'select_inputs'
                xname = 'kl_current_log_probs' if term == 'kl' and 'kl_current_log_probs' in data else 'current_log_probs'
                x = [v if not selected or (mask if term == 'policy' else term_mask)[i][j] else 0.
                     for j, v in enumerate(data[xname][i])]
                if term == 'policy':
                    reference_name='rollout_log_probs' if spec.correction=='rollout_ratio' else 'old_log_probs'
                    old = [v if not selected or mask[i][j] else 0. for j, v in enumerate(data[reference_name][i])]
                    delta = [x[j] - old[j] for j in range(t)]
                    if spec.ratio_masking == 'select_delta':
                        delta = [v if mask[i][j] else 0. for j, v in enumerate(delta)]
                    if p.ratio_level == 'token':
                        logratios = delta
                    else:
                        aggregate = sum(delta[j] * mask[i][j] for j in range(t))
                        if p.ratio_level != 'sequence_product':
                            aggregate /= max(1, sum(mask[i]))
                        logratios = [aggregate] * t
                    # Aggregate all ratio sensitivities once for coupled modes.
                    coupled = 0.
                for j in range(t):
                    m = term_mask[i][j]
                    scale = _divide(r.scale * r.process_count, divisors[i])
                    if r.reduction=='nonempty_sequence_mean':scale*=float(any(term_mask[i]))
                    factor = m * scale
                    if term == 'policy':
                        a = data['advantages'][i][j] if not selected or mask[i][j] else 0.
                        raw_l = logratios[j]
                        l = _clamp(raw_l, spec.ratio_delta_bounds)
                        ratio = _exp(l)
                        if spec.ratio_output_masking=='multiply':ratio*=mask[i][j]
                        near(raw_l, spec.ratio_delta_bounds, 'ratio_clamp_boundary')
                        wj = 0 if p.importance_level == 'sequence' else j
                        w = data['importance_weights'][i][wj] if spec.correction == 'weights' else 1.
                        if correction is not None:
                            w=correction['coefficients'][i][j]
                        if selected and not mask[i][j]:
                            w = 0.
                        filt = policy_filter[i][j] if policy_filter is not None else 1.
                        direct = dr = da = 0.
                        if spec.surrogate == 'ppo_clipped':
                            if m and filt and not p.detach_advantage and a == 0 and not p.clip_lower <= ratio <= p.clip_upper:
                                boundaries.add('zero_advantage_clip_branch_ambiguous')
                            saturated = (a > 0 and ratio > p.clip_upper) or (a < 0 and ratio < p.clip_lower)
                            effective = _clamp(ratio, (p.clip_lower, p.clip_upper)) if saturated else ratio
                            value, da = -a * effective, -effective
                            dr = 0. if saturated else -a * ratio
                            if a != 0:
                                near(ratio, (p.clip_upper, p.clip_upper) if a > 0 else (p.clip_lower, p.clip_lower), 'ppo_clip_boundary')
                        elif spec.surrogate == 'cispo':
                            coefficient = _clamp(ratio, (spec.coefficient_lower, spec.coefficient_upper))
                            value, direct, da = -a*coefficient*x[j], -a*coefficient, -coefficient*x[j]
                        else:
                            if m and filt and not p.detach_advantage and a == 0 and spec.tau_positive != spec.tau_negative:
                                boundaries.add('zero_advantage_temperature_branch_ambiguous')
                            tau = spec.tau_positive if a > 0 else spec.tau_negative
                            s = _sigmoid(tau*(ratio-1))
                            gate = 4*s/tau
                            if spec.surrogate == 'sapo_detached':
                                value, direct, da = -a*gate*x[j], -a*gate, -gate*x[j]
                            else:
                                value, da = -a*gate, -gate
                                # Stable mathematical limit at ratio infinity;
                                # replay separately detects literal 0*Inf backward.
                                dr = 0. if math.isinf(ratio) and s in (0, 1) else -4*a*s*(1-s)*ratio
                        if not _interior(raw_l, spec.ratio_delta_bounds):
                            dr = 0.
                        if spec.output_masking == 'select' and not m:
                            contribution = 0.
                        else:
                            contribution = value*w*filt*m
                        total += contribution*scale
                        local_factor = factor*w*filt
                        gx = grads['current_log_probs'][i]
                        if not selected or mask[i][j]:
                            gx[j] += direct*local_factor
                        sensitivity = dr*local_factor
                        if spec.ratio_masking == 'select_delta' and not mask[i][j] and p.ratio_level == 'token':
                            sensitivity = 0.
                        if p.ratio_level in {'token', 'sequence_token'}:
                            if not selected or mask[i][j]:
                                gx[j] += sensitivity
                                if reference_name in grads:
                                    grads[reference_name][i][j] -= sensitivity
                        else:
                            coupled += sensitivity
                        if 'advantages' in grads and (not selected or mask[i][j]):
                            grads['advantages'][i][j] += da*local_factor
                        if 'importance_weights' in grads and (not selected or mask[i][j]):
                            grads['importance_weights'][i][wj] += value*filt*factor
                    elif term == 'kl':
                        ref = data['reference_log_probs'][i][j] if not selected or m else 0.
                        sign = 1 if k.direction == 'current_minus_reference' else -1
                        raw = sign * (x[j]-ref)
                        d = _clamp(raw, k.delta_bounds)
                        near(raw, k.delta_bounds, 'kl_delta_boundary')
                        if k.estimator == 'k1':
                            value, derivative = d, 1.
                        elif k.estimator == 'k2':
                            value, derivative = .5*d*d, d
                        else:
                            e = _exp(d)
                            value, derivative = e-d-1, e-1
                        weight = 1.
                        if k.importance_weight == 'current_rollout':
                            rollout = data['rollout_log_probs'][i][j] if not selected or m else 0.
                            weight = _exp(x[j]-rollout)
                            if k.weight_nonfinite == 'zero' and not math.isfinite(weight):
                                weight = 0.
                        unclamped = value*weight
                        near(unclamped, k.value_bounds, 'kl_value_boundary')
                        value = _clamp(unclamped, k.value_bounds)
                        dc = weight*derivative*sign if _interior(raw, k.delta_bounds) else 0.
                        score_active = (k.score_weight or k.importance_weight == 'current_rollout') and (
                            not k.score_detach_on_delta_clamp or _interior(raw, k.delta_bounds))
                        score_derivative = unclamped if score_active else 0.
                        dx = dc + score_derivative
                        if not _interior(unclamped, k.value_bounds):
                            dx = dc = score_derivative = 0.
                        total += (0. if spec.output_masking == 'select' and not m else value*m)*scale*k.coefficient
                        if not selected or m:
                            grads[xname][i][j] += dx*factor*k.coefficient
                            if 'reference_log_probs' in grads:
                                grads['reference_log_probs'][i][j] -= dc*factor*k.coefficient
                            if 'rollout_log_probs' in grads:
                                grads['rollout_log_probs'][i][j] -= score_derivative*factor*k.coefficient
                    else:
                        value = data['entropy'][i][j] if not selected or m else 0.
                        total += (0. if spec.output_masking == 'select' and not m else -spec.entropy_coefficient*value*m)*scale
                        if not selected or m:
                            grads['entropy'][i][j] -= spec.entropy_coefficient*factor
                if term == 'policy' and p.ratio_level not in {'token', 'sequence_token'}:
                    divisor = 1 if p.ratio_level == 'sequence_product' else max(1, sum(mask[i]))
                    for j in range(t):
                        sensitivity = coupled*mask[i][j]/divisor
                        grads['current_log_probs'][i][j] += sensitivity
                        if reference_name in grads:
                            grads[reference_name][i][j] -= sensitivity
        term_values[term] = total
        term_gradients[term] = {name: {'state': state, 'values': grads.get(name)} for name, state in states.items()}
    term_values['total'] = sum(term_values.values())
    term_gradients['total'] = {}
    for name, state in input_states(spec, data).items():
        values = None
        if state == 'CONNECTED':
            values = [[sum(term_gradients[term][name]['values'][i][j]
                           for term in ('policy', 'kl', 'entropy')
                           if term_gradients[term][name]['values'] is not None)
                       for j in range(len(row))] for i, row in enumerate(data[name])]
        term_gradients['total'][name] = {'state': state, 'values': values}
    return {'precision': 'independent_scalar_float64', 'terms': term_values,
            'term_gradients': term_gradients, 'gradients': term_gradients['total'],
            'boundary_reasons': sorted(boundaries), 'correction':correction}
