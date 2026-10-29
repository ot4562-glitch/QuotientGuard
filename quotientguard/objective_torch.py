"""Literal torch oracle, independent of the Python derivative calculation."""
from __future__ import annotations

from .objective import _inputs, _plain


def full_objective_oracle(spec, trace, *, device='cpu'):
    """Explicit composite replay, separate from the legacy PPO oracle."""
    from .full_objective_torch import full_objective_oracle as replay
    return replay(spec, trace, device=device)


def ppo_oracle(spec, trace):
    import torch

    current, old, advantage, mask, weights, denominator = _inputs(spec, trace)
    dtype = getattr(torch, spec.dtype)
    x = torch.tensor(current, dtype=dtype, requires_grad=True)
    o = torch.tensor(old, dtype=dtype)
    a = torch.tensor(advantage, dtype=dtype)
    m = torch.tensor(mask, dtype=dtype)
    w = torch.tensor(weights, dtype=dtype)
    selected_x = torch.where(m.bool(), x, 0.) if spec.masking == "select_inputs" else x
    if spec.masking == "select_inputs":
        o = torch.where(m.bool(), o, 0.)
        a = torch.where(m.bool(), a, 0.)
        w = torch.where(m.bool(), w, 0.)
    delta = selected_x - o
    aggregate = None
    if spec.ratio_level != "token":
        aggregate = (delta * m).sum(-1, keepdim=True)
        if spec.ratio_level != "sequence_product":
            aggregate = aggregate / m.sum(-1, keepdim=True).clamp_min(1)
        log_ratio = (selected_x - selected_x.detach() + aggregate.detach()
                     if spec.ratio_level == "sequence_token" else aggregate)
    else:
        log_ratio = delta
    ratio = torch.exp(log_ratio)
    unclipped = ratio * a
    clipped = ratio.clamp(spec.clip_lower, spec.clip_upper) * a
    surrogate = torch.minimum(unclipped, clipped)
    weighted = surrogate * w
    masked = weighted * m
    if spec.reduction == "per_sequence_active_mean":
        counts = m.sum(-1)
        if spec.empty_mask_policy == "zero":
            counts = counts.clamp_min(1)
        # Preserve the declared operation order. B*count can overflow half even
        # when the two successive mean operations are entirely representable.
        loss = -(masked.sum(-1) / counts).mean()
    else:
        loss = -masked.sum() / denominator
    gradient, = torch.autograd.grad(loss, x)
    scaled = gradient * spec.gradient_scale
    stages = {"subtraction": delta, "log_ratio": log_ratio, "exp_ratio": ratio, "unclipped_product": unclipped,
              "clipped_product": clipped, "weighted_surrogate": weighted,
              "masked_surrogate": masked, "loss": loss, "gradient": gradient,
              "scaled_gradient": scaled}
    if aggregate is not None:
        stages["sequence_aggregation"] = aggregate
    events = []
    for name, tensor in stages.items():
        if not torch.isfinite(tensor).all().item():
            events.append({"stage": name, "kind": "NONFINITE", "execution": "REFERENCE_REPLAY",
                           "indices": (~torch.isfinite(tensor)).nonzero().tolist()})
    underflow = (ratio == 0) & torch.isfinite(log_ratio)
    if underflow.any().item():
        events.append({"stage": "exp_ratio", "kind": "UNDERFLOW_TO_ZERO",
                       "execution": "REFERENCE_REPLAY", "indices": underflow.nonzero().tolist()})
    return {"execution": "REFERENCE_REPLAY", "torch_version": torch.__version__,
            "dtype": spec.dtype, "loss": _plain(loss), "coefficients": _plain(scaled),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "gradient_finite": bool(torch.isfinite(gradient).all().item()),
            "excluded_gradients_zero": bool((gradient[m == 0] == 0).all().item()),
            "stages": {k: _plain(v) for k, v in stages.items()}, "numerical_events": events}
