"""Count-bound normalization and effective microbatch/full-batch equivalence.

All ranks and masks are supplied snapshots. No framework, producer, collective
or optimizer executes here. Explicit counts are reconstructed independently;
implicit reductions retain their own units rather than becoming nominally
identical just because two denominators happen to have the same value.
"""
from dataclasses import asdict, dataclass, field
import math
from typing import Any

from .full_objective import (FullObjectiveSpec, FullObjectiveTrace, audit_full_objective,
                            full_inputs, reduction_for, unsupported, _observed_gradient_matches)
from .objective import ObjectiveTrace, _matrix, _plain, _json_safe, _flatten, _close
from .provenance import sha256_json


@dataclass(frozen=True)
class DenominatorCountSpec:
    unit: str
    mask_stage: str = 'loss'
    declared_total: Any = None
    divide_by_process_count: bool = False
    completion_length: int | None = None
    minimum: float | None = None


@dataclass(frozen=True)
class ObjectiveWindowSpec:
    reference: FullObjectiveSpec
    counts: dict[str, DenominatorCountSpec] = field(default_factory=dict)
    process_count: int = 1
    process_reduction: str = 'mean'

    @classmethod
    def from_dict(cls, payload):
        values=dict(payload)
        values['reference']=FullObjectiveSpec.from_dict(values['reference'])
        values['counts']={name:DenominatorCountSpec(**value) for name,value in values.get('counts',{}).items()}
        return cls(**values)


@dataclass
class ObjectiveWindowBatch:
    spec: FullObjectiveSpec
    trace: FullObjectiveTrace
    rank: int = 0
    microbatch_index: int = 0
    backward_scale: float = 1.
    count_masks: dict[str, Any] = field(default_factory=dict)
    observed_backward_gradients: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls,payload):
        values=dict(payload)
        values['spec']=FullObjectiveSpec.from_dict(values['spec'])
        values['trace']=FullObjectiveTrace.from_dict(values['trace'])
        return cls(**values)


@dataclass(frozen=True)
class ObjectiveWindowResult:
    evidence: dict[str, Any]

    def to_dict(self):
        payload=_json_safe(self.evidence)
        return {**payload,'audit_sha256':sha256_json(payload)}


_UNITS={'sum':'none','active_mean':'active_tokens','per_sequence_active_mean':'sequences',
        'fixed_length_mean':'fixed_length_slots','nonempty_sequence_mean':'nonempty_sequences'}


def _enabled(spec):
    return ['policy']+(['kl'] if spec.kl.coefficient!=0 else [])+(['entropy'] if spec.entropy_coefficient!=0 else [])


def _term_mask(data,term):
    return data.get(term+'_mask',data['mask']) if term!='policy' else data['mask']


def _explicit(reduction):
    return reduction.reduction=='explicit' or reduction.reduction=='nonempty_sequence_mean' and reduction.denominator is not None


def _signature(spec):
    value=asdict(spec)
    for name in ('policy_reduction','entropy_reduction'):value.pop(name)
    value['kl'].pop('reduction')
    for name in ('reduction','denominator','completion_length','empty_mask_policy'):value['policy'].pop(name)
    return value


def _count(unit,masks,length=None):
    if unit=='none':return 1.
    if unit=='active_tokens':return float(sum(sum(row) for matrix in masks for row in matrix))
    if unit=='nonempty_sequences':return float(sum(any(row) for matrix in masks for row in matrix))
    rows=sum(len(matrix) for matrix in masks)
    return float(rows*length if unit=='fixed_length_slots' else rows)


def _merge(data,width,spec):
    return {name:[list(row)+[0.]*(target-len(row)) for item in data for row in item[name]]
            for name in data[0] for target in [1 if name=='importance_weights' and spec.policy.importance_level=='sequence' else width]}


def _trace(data):
    policy=ObjectiveTrace(*(data[name] for name in ('current_log_probs','old_log_probs','advantages','mask')),
                          importance_weights=data.get('importance_weights'))
    names=('reference_log_probs','entropy','kl_current_log_probs','rollout_log_probs','policy_filter','kl_mask','entropy_mask')
    return FullObjectiveTrace(policy,**{name:data[name] for name in names if name in data})


def _nonfinite_observations(batches,reference_trace):
    traces=[batch.trace for batch in batches]+([reference_trace] if reference_trace is not None else [])
    values=[value for trace in traces for value in (trace.observed_loss,trace.policy.observed_loss,
            trace.policy.observed_coefficients,*trace.observed_terms.values(),*trace.observed_gradients.values())]
    values.extend(value for batch in batches for value in batch.observed_backward_gradients.values())
    return any(not math.isfinite(value) for item in values for value in _flatten(item))


def _proof(reduction,loss_masks,*,count_spec=None,count_masks=None,process_count=1,reference=False):
    if _explicit(reduction):
        raw=_count(count_spec.unit,count_masks,count_spec.completion_length)
        expected=raw/(process_count if count_spec.divide_by_process_count and not reference else 1)
        if count_spec.minimum is not None:expected=max(count_spec.minimum,expected)
        actual=float(reduction.denominator)
        return {'unit':count_spec.unit,'mask_stage':count_spec.mask_stage,'raw_count':raw,
                'expected_denominator':expected,'actual_denominator':actual,
                'matches':actual==expected and (count_spec.unit!='fixed_length_slots' or all(
                    sum(row)<=count_spec.completion_length for matrix in count_masks for row in matrix)),
                'scope':reduction.denominator_scope,'epsilon':reduction.denominator_epsilon}
    unit=_UNITS[reduction.reduction]
    raw=_count(unit,loss_masks,reduction.completion_length)
    actual=max(1.,raw) if reduction.empty_mask_policy=='zero' and unit in {'active_tokens','nonempty_sequences'} else raw
    valid=(reduction.reduction!='fixed_length_mean' or all(sum(row)<=reduction.completion_length for matrix in loss_masks for row in matrix))
    return {'unit':unit,'mask_stage':'loss','raw_count':raw,'expected_denominator':actual,'actual_denominator':actual,
            'matches':valid,'scope':'derived_from_this_snapshot','epsilon':reduction.denominator_epsilon,
            'row_active_counts':[sum(row) for matrix in loss_masks for row in matrix]}


def _summary(audit):
    return {key:audit[key] for key in ('audit_sha256','status','reasons')}|{'terms':audit['oracle']['terms'] if audit['oracle'] else None}


def audit_objective_window(spec:ObjectiveWindowSpec,batches,*,reference_trace=None,device='cpu'):
    """Verify count units and weighted coordinate loss/gradient equivalence.

    Explicit reductions need a count binding. Implicit reductions derive their
    own units, allowing a sum divided once to match a global mean. Counts and
    process reduction are reconstructions/simulations over the supplied complete
    window, never attestation of a Trainer's gather, DDP or sampling producer.
    Backward observations use local post-Trainer-scale, pre-process-average units.
    """
    batches=list(batches)
    if type(spec.process_count) is not int or spec.process_count<1 or not batches:
        raise ValueError('provide a positive process_count and nonempty window')
    seen=set()
    for batch in batches:
        if (type(batch.rank) is not int or not 0<=batch.rank<spec.process_count or
                type(batch.microbatch_index) is not int or batch.microbatch_index<0 or
                isinstance(batch.backward_scale,bool) or not math.isfinite(batch.backward_scale) or batch.backward_scale<0):
            raise ValueError('invalid rank, microbatch index or backward scale')
        key=(batch.rank,batch.microbatch_index)
        if key in seen:raise ValueError('duplicate rank/microbatch index')
        seen.add(key)
    report={'schema_version':'quotientguard-objective-window-v1','status':'UNSUPPORTED','spec':asdict(spec),'reasons':[],
        'counts_reconstructed_from_supplied_masks':False,'count_producer_executed':False,'trainer_executed':False,
        'optimizer_step_executed':False,'collectives_executed':False,'semantic_confirmation_eligible':False,
        'execution':'REFERENCE_WINDOW_REPLAY','natural_input_reachability':'UNVERIFIED','count_totals':{},'count_proofs':[],
        'mismatches':[],'limitations':['All ranks/masks are caller-supplied snapshots; input coverage is not remotely attested.',
            'Process averaging is simulated; no collective or parameter Jacobian is executed.',
            'Effective loss uses declared backward scales and is not a Trainer logging-loss claim.',
            'Producer filtering/quantiles and AMP scaler state are not inferred from these coordinates.']}
    def finish(status,*reasons):
        report.update(status=status,reasons=[*report['reasons'],*reasons])
        return ObjectiveWindowResult(report)
    if _nonfinite_observations(batches,reference_trace):
        return finish('NUMERICAL_ANOMALY_OBSERVED','nonfinite_supplied_window_observation')
    reasons=unsupported(spec.reference)
    for i,batch in enumerate(batches):reasons.extend(f'batch{i}:{reason}' for reason in unsupported(batch.spec))
    if spec.process_reduction not in {'mean','sum'}:reasons.append('unsupported_process_reduction')
    for term,count in spec.counts.items():
        if term not in _enabled(spec.reference) or count.unit not in {*_UNITS.values()} or count.mask_stage not in {'loss','pre_filter'}:
            reasons.append('unsupported_count_binding:'+term)
        if count.unit=='fixed_length_slots' and (type(count.completion_length) is not int or count.completion_length<=0):
            reasons.append('invalid_fixed_count_length:'+term)
        if count.minimum is not None and (not math.isfinite(count.minimum) or count.minimum<0):
            reasons.append('invalid_count_minimum:'+term)
    if reasons:return finish('UNSUPPORTED',*reasons)
    if {batch.rank for batch in batches}!=set(range(spec.process_count)):
        return finish('INSUFFICIENT_EVIDENCE','missing_process_snapshot')
    if any(_signature(batch.spec)!=_signature(spec.reference) for batch in batches):
        return finish('UNSUPPORTED','microbatch_nonreduction_math_differs')
    data=[full_inputs(batch.spec,batch.trace) for batch in batches]
    if any(set(item)!=set(data[0]) for item in data):
        return finish('INSUFFICIENT_EVIDENCE','inconsistent_input_coordinates')
    width=max(len(item['current_log_probs'][0]) for item in data)
    merged=_merge(data,width,spec.reference)
    canonical=_trace(merged)
    if reference_trace is not None:
        if _json_safe(full_inputs(spec.reference,reference_trace))!=_json_safe(merged):
            return finish('INSUFFICIENT_EVIDENCE','reference_snapshot_inputs_differ')
        canonical=reference_trace
    count_masks={}
    for term in _enabled(spec.reference):
        binding=spec.counts.get(term)
        reductions=[reduction_for(spec.reference,term)]+[reduction_for(batch.spec,term) for batch in batches]
        if binding is None and any(_explicit(r) for r in reductions):
            return finish('INSUFFICIENT_EVIDENCE','missing_explicit_count_unit:'+term)
        matrices=[]
        for index,(batch,item) in enumerate(zip(batches,data)):
            loss_mask=_term_mask(item,term)
            if binding is not None and binding.mask_stage=='pre_filter':
                if term not in batch.count_masks:return finish('INSUFFICIENT_EVIDENCE',f'batch{index}:missing_prefilter_mask:{term}')
                mask=_matrix(batch.count_masks[term],f'count_masks.{term}')
                if len(mask)!=len(loss_mask) or any(len(a)!=len(b) for a,b in zip(mask,loss_mask)):
                    raise ValueError('count mask must match loss input shape')
                if any(v not in (0,1) for row in mask for v in row):raise ValueError('count mask must be binary')
                if any(v>mask[i][j] for i,row in enumerate(loss_mask) for j,v in enumerate(row)):
                    report['mismatches'].append(f'batch{index}:prefilter_mask_excludes_loss_tokens:{term}')
            else:
                mask=loss_mask
                if term in batch.count_masks and _matrix(batch.count_masks[term],f'count_masks.{term}')!=loss_mask:
                    report['mismatches'].append(f'batch{index}:count_mask_conflicts_with_loss_stage:{term}')
            matrices.append(mask)
        count_masks[term]=matrices
        if binding is not None:
            total=_count(binding.unit,matrices,binding.completion_length)
            declared=_plain(binding.declared_total)
            if declared is not None:
                if isinstance(declared,(list,dict,bool)):raise ValueError('declared_total must be a scalar count')
                declared=float(declared)
                if not math.isfinite(declared):return finish('NUMERICAL_ANOMALY_OBSERVED','nonfinite_declared_count:'+term)
                if declared!=total:report['mismatches'].append('declared_total:'+term)
            report['count_totals'][term]={'value':total,'unit':binding.unit,'mask_stage':binding.mask_stage,'declared_total':declared}
        else:
            reference_reduction=reduction_for(spec.reference,term)
            unit=_UNITS[reference_reduction.reduction]
            report['count_totals'][term]={'value':_count(unit,[_term_mask(merged,term)],reference_reduction.completion_length),
                                         'unit':unit,'mask_stage':'loss'}
        proof=_proof(reductions[0],[_term_mask(merged,term)],count_spec=binding,count_masks=matrices,reference=True)
        report['count_proofs'].append({'batch':'reference','term':term,**proof})
        if not proof['matches']:report['mismatches'].append('reference.denominator:'+term)
        for index,(batch,item,reduction) in enumerate(zip(batches,data,reductions[1:])):
            if _explicit(reduction):
                if reduction.denominator_scope=='microbatch':selected=[matrices[index]]
                elif reduction.denominator_scope=='batch':
                    positions=[i for i,b in enumerate(batches) if b.microbatch_index==batch.microbatch_index]
                    if {batches[i].rank for i in positions}!=set(range(spec.process_count)):
                        return finish('INSUFFICIENT_EVIDENCE','incomplete_cross_rank_batch_count:'+term)
                    selected=[matrices[i] for i in positions]
                else:selected=matrices
            else:selected=None
            proof=_proof(reduction,[_term_mask(item,term)],count_spec=binding,count_masks=selected,process_count=spec.process_count)
            report['count_proofs'].append({'batch':index,'rank':batch.rank,'term':term,**proof})
            if not proof['matches']:report['mismatches'].append(f'batch{index}.denominator:{term}')
    if report['mismatches']:return finish('COUNT_MISMATCH_OBSERVED','count_binding_does_not_match_masks')
    report['counts_reconstructed_from_supplied_masks']=True
    reference=audit_full_objective(spec.reference,canonical,device=device).to_dict()
    report['reference_audit']=_summary(reference)
    report['reference_input_sha256']=sha256_json(_json_safe(merged))
    if reference['status']!='SUPPORTED':return finish(reference['status'],'reference_objective_not_supported')
    report['reference_terms']=reference['oracle']['terms']
    report['reference_gradients']=reference['oracle']['gradients']
    effective_terms={name:0. for name in ('policy','kl','entropy','total')}
    effective_gradients={term:{name:{'state':entry['state'],'values':[] if entry['values'] is not None else None}
                              for name,entry in entries.items()} for term,entries in reference['oracle']['term_gradients'].items()}
    report['microbatches']=[]
    from .full_objective_torch import full_objective_oracle
    rank_factor=1/spec.process_count if spec.process_reduction=='mean' else 1.
    for index,(batch,item) in enumerate(zip(batches,data)):
        audit=audit_full_objective(batch.spec,batch.trace,device=device).to_dict()
        report['microbatches'].append({'rank':batch.rank,'microbatch_index':batch.microbatch_index,
            'backward_scale':batch.backward_scale,'simulated_process_scale':rank_factor,
            'input_sha256':sha256_json(_json_safe(item)),**_summary(audit)})
        if audit['status']!='SUPPORTED':return finish(audit['status'],f'batch{index}:objective_not_supported')
        local=(audit['oracle'] if batch.backward_scale==1 else
               full_objective_oracle(batch.spec,batch.trace,device=device,loss_scale=batch.backward_scale))
        report['microbatches'][-1]['local_backward']={key:local[key] for key in
            ('terms','loss_finite','gradients_finite','first_nonfinite_stage','numerical_events')}
        if not local['loss_finite'] or not local['gradients_finite']:
            return finish('NUMERICAL_ANOMALY_OBSERVED',f'batch{index}:local_backward_nonfinite')
        # Average the already rounded local coordinate gradients, as a process
        # reduction would. Moving this factor into the loss changes fp16/bf16
        # backward rounding and can hide local overflow/underflow.
        import torch
        for term,value in local['terms'].items():effective_terms[term]+=value*rank_factor
        for term,entries in local['term_gradients'].items():
            for name,entry in entries.items():
                target=effective_gradients[term][name]
                if target['state']!=entry['state']:report['mismatches'].append(f'batch{index}:{term}.state.{name}')
                if entry['values'] is not None:
                    target_width=1 if name=='importance_weights' and spec.reference.policy.importance_level=='sequence' else width
                    values=entry['values'] if rank_factor==1 else _plain(
                        torch.tensor(entry['values'],dtype=getattr(torch,batch.spec.policy.dtype),device=device)*rank_factor)
                    target['values'].extend([list(row)+[0.]*(target_width-len(row)) for row in values])
        if batch.observed_backward_gradients:
            for name,value in batch.observed_backward_gradients.items():
                expected=local['gradients'].get(name)
                if expected is None or not _observed_gradient_matches(value,expected['values'],name,batch.spec,item):
                    report['mismatches'].append(f'batch{index}.backward.{name}')
    report.update(effective_terms=effective_terms,effective_gradients=effective_gradients['total'])
    for term in effective_terms:
        if not _close(effective_terms[term],reference['oracle']['terms'][term],spec.reference.policy):
            report['mismatches'].append('effective_loss:'+term)
        for name,entry in effective_gradients[term].items():
            expected=reference['oracle']['term_gradients'][term][name]
            if (entry['state']!=expected['state'] or (entry['values'] is None)!=(expected['values'] is None) or
                    entry['values'] is not None and not _close(entry['values'],expected['values'],spec.reference.policy)):
                report['mismatches'].append(f'effective_gradient:{term}.{name}')
    return finish('CONTRACT_MISMATCH_OBSERVED' if report['mismatches'] else 'SUPPORTED')
