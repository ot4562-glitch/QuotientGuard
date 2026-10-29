"""Literal tensor execution, separately checked by scalar closed-form Jacobians."""
from .full_objective import full_inputs, input_states, reduction_for, unsupported
from .objective import _plain


def full_objective_oracle(spec, trace, *, device='cpu', loss_scale=1.):
    import math
    if isinstance(loss_scale,bool) or not isinstance(loss_scale,(int,float)) or not math.isfinite(loss_scale) or loss_scale<0:
        raise ValueError('loss_scale must be a finite nonnegative real')
    import torch

    reasons = unsupported(spec)
    if reasons:
        raise ValueError('unsupported full-objective replay: '+', '.join(reasons))

    data = full_inputs(spec, trace)
    p, k = spec.policy, spec.kl
    dtype = getattr(torch, p.dtype)
    controls = {'mask', 'kl_mask', 'entropy_mask', 'policy_filter'}
    inputs = {name: torch.tensor(value, dtype=dtype, device=device,
                                requires_grad=name not in controls) for name, value in data.items()}
    stages = {}

    def stage(name, value):
        stages[name] = value
        return value

    def select(value, mask):
        return torch.where(mask.bool(), value, 0.) if p.masking == 'select_inputs' else value

    def clamp(value, bounds):
        return value if bounds is None else value.clamp(*bounds)

    def reduce(value, mask, name):
        r = reduction_for(spec, name)
        masked = (torch.where(mask.bool(), value, 0.) if spec.output_masking == 'select' else value) * mask
        stage(name+'.masked_terms', masked)
        if r.reduction=='nonempty_sequence_mean':
            counts=mask.sum(-1)
            sequence_mask=(counts>0).float()
            row_counts=counts.clamp_min(1) if r.empty_mask_policy=='zero' else counts
            row_means=stage(name+'.sequence_means',masked.sum(-1)/(row_counts+r.denominator_epsilon))
            selected=stage(name+'.selected_sequence_means',row_means*sequence_mask)
            denominator=r.denominator if r.denominator is not None else sequence_mask.sum()
            if r.empty_mask_policy=='zero' and r.denominator is None:denominator=denominator.clamp_min(1)
            reduced=selected.sum()/denominator
        elif r.reduction == 'per_sequence_active_mean':
            counts = mask.sum(-1)
            if r.empty_mask_policy == 'zero':
                counts = counts.clamp_min(1)
            reduced = (masked.sum(-1)/(counts+r.denominator_epsilon)).mean()
        else:
            denominator = (1. if r.reduction == 'sum' else r.denominator if r.reduction == 'explicit'
                           else mask.shape[0]*r.completion_length if r.reduction == 'fixed_length_mean' else mask.sum())
            if r.reduction == 'active_mean' and r.empty_mask_policy == 'zero':
                denominator = denominator.clamp_min(1)
            reduced = masked.sum()/(denominator+r.denominator_epsilon)
        return stage(name+'.loss', reduced*r.process_count*r.scale)

    m = inputs['mask']
    correction=None
    correction_coefficient=None
    if spec.correction=='gated_old_rollout':
        from .importance_correction import torch_correction
        correction_coefficient,correction=torch_correction(spec.importance_correction,inputs['old_log_probs'],
                                                         inputs['rollout_log_probs'],m,stage)
    x = select(inputs['current_log_probs'], m)
    o = select(inputs['rollout_log_probs'] if spec.correction=='rollout_ratio' else inputs['old_log_probs'], m)
    a = select(inputs['advantages'], m)
    detach_reference=spec.detach_rollout_policy if spec.correction=='rollout_ratio' else p.detach_old_policy
    if detach_reference:
        o = o.detach()
    if p.detach_advantage:
        a = a.detach()
    delta = stage('policy.subtraction', x-o)
    if spec.ratio_masking == 'select_delta':
        delta = stage('policy.selected_delta', torch.where(m.bool(), delta, 0.))
    if p.ratio_level == 'token':
        log_ratio = delta
    else:
        aggregate = (delta*m).sum(-1, keepdim=True)
        if p.ratio_level != 'sequence_product':
            aggregate = aggregate/m.sum(-1, keepdim=True).clamp_min(1)
        stage('policy.sequence_aggregation', aggregate)
        log_ratio = x-x.detach()+aggregate.detach() if p.ratio_level == 'sequence_token' else aggregate
    ratio = stage('policy.exp_ratio', clamp(log_ratio, spec.ratio_delta_bounds).exp())
    if spec.ratio_output_masking=='multiply':ratio=stage('policy.masked_ratio',ratio*m)
    if spec.surrogate == 'ppo_clipped':
        # max of negative terms is independently differentiated, while the
        # scalar implementation selects the active closed-form derivative.
        raw = -torch.minimum(ratio*a, ratio.clamp(p.clip_lower, p.clip_upper)*a)
    elif spec.surrogate == 'cispo':
        coefficient = stage('policy.coefficient', ratio.clamp(spec.coefficient_lower, spec.coefficient_upper).detach())
        raw = -coefficient*a*x
    else:
        tau = torch.where(a > 0, a.new_tensor(spec.tau_positive), a.new_tensor(spec.tau_negative))
        gate = stage('policy.gate', torch.sigmoid(tau*(ratio-1))*4/tau)
        raw = -a*gate.detach()*x if spec.surrogate == 'sapo_detached' else -a*gate
    stage('policy.surrogate', raw)
    if spec.correction == 'weights':
        w = select(inputs['importance_weights'], m)
        raw = raw*(w.detach() if p.detach_importance else w)
        stage('policy.corrected_terms', raw)
    elif correction_coefficient is not None:
        raw=raw*correction_coefficient
        stage('policy.corrected_terms',raw)
    if 'policy_filter' in inputs:
        raw = raw*inputs['policy_filter']
    token_terms = {'policy': stage('policy.terms', raw)}
    masks = {'policy': m}
    if k.coefficient != 0:
        km = inputs.get('kl_mask', m)
        xx = select(inputs.get('kl_current_log_probs', inputs['current_log_probs']), km)
        ref = select(inputs['reference_log_probs'], km)
        if k.detach_reference:
            ref = ref.detach()
        raw_d = stage('kl.subtraction', ref-xx if k.direction == 'reference_minus_current' else xx-ref)
        d = stage('kl.clamped_delta', clamp(raw_d, k.delta_bounds))
        if k.estimator == 'k1':
            kl = d
        elif k.estimator == 'k2':
            kl = d.square()/2
        else:
            exponential = stage('kl.exp_delta', d.exp())
            kl = exponential-d-1
        if k.score_weight or k.importance_weight == 'current_rollout':
            if k.importance_weight == 'current_rollout':
                rollout = select(inputs['rollout_log_probs'], km)
                if k.detach_rollout:
                    rollout = rollout.detach()
                score = stage('kl.exp_importance', (xx-rollout).exp())
            else:
                score = stage('kl.exp_importance', (xx-xx.detach()).exp())
            if k.weight_nonfinite == 'zero':
                score = torch.nan_to_num(score, nan=0., posinf=0., neginf=0.)
            if k.score_detach_on_delta_clamp:
                score = torch.where(raw_d == d, score, score.detach())
            kl = kl*stage('kl.score_weight', score)
        kl = stage('kl.clamped_value', clamp(kl, k.value_bounds))
        token_terms['kl'] = kl*k.coefficient
        masks['kl'] = km
    if spec.entropy_coefficient != 0:
        em = inputs.get('entropy_mask', m)
        token_terms['entropy'] = -spec.entropy_coefficient*select(inputs['entropy'], em)
        masks['entropy'] = em
    terms = {name: reduce(value, masks[name], name) for name, value in token_terms.items()}
    zero = torch.zeros((), dtype=dtype, device=device)
    for name in ('kl', 'entropy'):
        terms.setdefault(name, zero)
    terms['total'] = (reduce(sum(token_terms.values()), m, 'policy') if spec.composition == 'before_reduction'
                      else sum(terms.values()))
    if loss_scale!=1:
        terms={name:stage(name+'.scaled_loss',value*loss_scale) for name,value in terms.items()}
    stage('total.loss', terms['total'])
    names = [name for name in inputs if name not in controls]
    targets = [inputs[name] for name in names]
    term_gradients = {}
    for term, loss in terms.items():
        gradients = torch.autograd.grad(loss, targets, allow_unused=True, retain_graph=True) if loss.requires_grad else [None]*len(targets)
        declared = input_states(spec, data, None if term == 'total' else term)
        records = {}
        for name, g in zip(names, gradients):
            state = 'CONNECTED' if g is not None else declared[name]
            # A declared connection missing from execution must remain UNUSED.
            if g is None and state == 'CONNECTED':
                state = 'UNUSED'
            records[name] = {'state': state, 'values': _plain(g) if g is not None else None}
            if g is not None:
                stage(term+'.gradient.'+name, g)
        term_gradients[term] = records
    # One device synchronization and host transfer per stage at serialization;
    # no extra isfinite().item() synchronization per stage.
    plain = {name: _plain(value) for name, value in stages.items()}
    import math
    from .objective import _flatten
    events = [{'stage': name, 'kind': 'NONFINITE', 'execution': 'REFERENCE_REPLAY'}
              for name, value in plain.items() if any(not math.isfinite(v) for v in _flatten(value))]
    total_g = term_gradients['total']
    finite_g = all(math.isfinite(v) for records in term_gradients.values() for entry in records.values()
                   for v in _flatten(entry['values']))
    excluded = all(value == 0 for i, row in enumerate(total_g['current_log_probs']['values'])
                   for j, value in enumerate(row) if data['mask'][i][j] == 0
                   and ('kl_current_log_probs' in data or k.coefficient == 0 or data.get('kl_mask', data['mask'])[i][j] == 0))
    return {'execution': 'REFERENCE_REPLAY', 'torch_version': torch.__version__, 'dtype': p.dtype,
            'device': str(device), 'loss_scale':loss_scale,'loss': _plain(terms['total']), 'loss_dtype':str(terms['total'].dtype),
            'term_dtypes':{name:str(value.dtype) for name,value in terms.items()},
            'terms': {name: _plain(value) for name, value in terms.items()},
            'gradients': total_g, 'term_gradients': term_gradients, 'stages': plain,
            'loss_finite': math.isfinite(_plain(terms['total'])), 'gradients_finite': finite_g,
            'excluded_gradients_zero': excluded, 'numerical_events': events,
            'first_nonfinite_stage': events[0]['stage'] if events else None,
            'correction':{name:_plain(value) for name,value in correction.items()} if correction is not None else None}
