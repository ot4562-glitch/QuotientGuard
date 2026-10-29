"""Explicit gated old/rollout density correction used by the pinned PPO source.

Sequence-level gates retain per-token weights. Neither sampled-token binary
KL nor TV is a categorical-distribution divergence certificate. Inputs denote
log densities in the caller's declared distributions, not inferred sampler
probabilities. No framework or sampler is imported here.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ImportanceCorrectionSpec:
    mode: str = 'mask'
    level: str = 'token'
    gating: str = 'ratio'
    lower: float = .5
    upper: float = 5.


def correction_reasons(spec):
    reasons=[]
    if spec.mode not in {'mask','clip'} or spec.level not in {'token','sequence'} or spec.gating not in {'ratio','binary_kl','tv'}:
        reasons.append('unsupported_importance_gate')
    if not math.isfinite(spec.lower) or not math.isfinite(spec.upper) or not 0<=spec.lower<=spec.upper:
        reasons.append('invalid_importance_thresholds')
    if spec.mode=='clip' and (spec.level!='token' or spec.gating!='ratio'):
        reasons.append('clip_correction_requires_token_ratio')
    if spec.gating!='ratio' and spec.lower>0:
        reasons.append('divergence_gate_requires_zero_lower_bound')
    return reasons


def _exp(value):
    try:return math.exp(value)
    except OverflowError:return math.inf


def _probability(value):
    value=_exp(value)
    return value if math.isnan(value) else max(1e-6,min(1-1e-6,value))


def scalar_correction(spec,old,rollout,mask):
    """Independent scalar forward coefficient; every coefficient path is detached."""
    delta=[[a-b for a,b in zip(o,r)] for o,r in zip(old,rollout)]
    token_ratios=[[_exp(v) for v in row] for row in delta]
    statistic=[]
    for i,row in enumerate(delta):
        if spec.gating=='ratio':
            values=row
        else:
            values=[]
            for old_value,rollout_value in zip(old[i],rollout[i]):
                p,q=_probability(rollout_value),_probability(old_value)
                # log differences are independent of the tensor implementation's
                # log of probability ratios; both define the same binary KL.
                values.append(abs(p-q) if spec.gating=='tv' else
                    p*(math.log(p)-math.log(q))+(1-p)*(math.log1p(-p)-math.log1p(-q)))
        if spec.level=='sequence':
            denominator=sum(mask[i])
            values=[sum(v*m for v,m in zip(values,mask[i]))/denominator if denominator else math.nan]
        if spec.gating=='ratio':
            values=[_exp(v) for v in values]
        statistic.append(values)
    coefficients=[]
    keep=[]
    for i,row in enumerate(token_ratios):
        row_coefficients=[]
        row_keep=[]
        for j,ratio in enumerate(row):
            value=statistic[i][0 if spec.level=='sequence' else j]
            inside=spec.lower<=value<=spec.upper
            if spec.mode=='clip':
                coefficient=value if math.isnan(value) else max(spec.lower,min(spec.upper,value))
                inside=not (value<spec.lower or value>spec.upper)
            else:
                coefficient=ratio if inside else 0.
            row_coefficients.append(coefficient)
            row_keep.append(inside)
        coefficients.append(row_coefficients)
        keep.append(row_keep if spec.level=='token' else row_keep[:1])
    return {'coefficients':coefficients,'statistic':statistic,'keep':keep,
            'token_ratios':token_ratios,'log_ratio':delta}


def torch_correction(spec,old,rollout,mask,stage):
    """Execute literal source arithmetic; exposed stages precede policy terms."""
    import torch

    log_ratio=stage('correction.subtraction',(old-rollout).detach())
    token_ratios=stage('correction.exp_ratio',log_ratio.exp())
    if spec.gating=='ratio':
        statistic=log_ratio
    else:
        p=stage('correction.rollout_probability',rollout.exp().clamp(1e-6,1-1e-6))
        q=stage('correction.old_probability',old.exp().clamp(1e-6,1-1e-6))
        statistic=(p-q).abs() if spec.gating=='tv' else p*(p/q).log()+(1-p)*((1-p)/(1-q)).log()
        stage('correction.token_statistic',statistic)
    if spec.level=='sequence':
        statistic=stage('correction.sequence_statistic',(statistic*mask).sum(-1,keepdim=True)/mask.sum(-1,keepdim=True))
    if spec.gating=='ratio':
        statistic=statistic.exp()
    stage('correction.statistic',statistic)
    if spec.mode=='clip':
        coefficient=statistic.clamp(spec.lower,spec.upper)
        keep=~((statistic<spec.lower)|(statistic>spec.upper))
    else:
        keep=(statistic>=spec.lower)&(statistic<=spec.upper)
        coefficient=torch.where(keep,token_ratios,0.)
    coefficient=stage('correction.coefficient',coefficient.detach())
    return coefficient,{'coefficients':coefficient,'statistic':statistic,'keep':keep,
                        'token_ratios':token_ratios,'log_ratio':log_ratio}
