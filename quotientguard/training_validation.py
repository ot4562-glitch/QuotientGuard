"""Controlled, explicit backward/AMP/accumulation/SGD validation.

This API intentionally executes a step on the caller's model and optimizer.
Use disposable models in isolated adapters. It never imports a framework and
does not infer Trainer, distributed, natural-input or source-identity evidence.
"""
import math

from .native_validation import _seal


def validate_training_step(loss_builder, parameters, microbatches, *, optimizer,
                           accumulation_weights=None, scaler=None,
                           max_grad_norm=None, autocast_dtype=None, device_type='cpu'):
    """Execute one accumulation window, actual unscale, optional clip and SGD.

    Only plain SGD (zero momentum/decay) has an update oracle. Unused parameters
    stay None. Existing .grad is cleared, then actual post-clip .grad is retained.
    Autograd partials before backward independently check accumulation/unscale;
    they do not independently establish the loss formula. Use a full-objective
    native audit for that separate contract.
    """
    import torch

    report = {'schema_version': 'quotientguard-training-validation-v1', 'status': 'UNSUPPORTED',
              'execution_scope': 'CONTROLLED_OPTIMIZER_STEP', 'trainer_executed': False,
              'semantic_confirmation_eligible': False, 'natural_input_reachability': 'UNVERIFIED',
              'optimizer_step_executed': False, 'optimizer_step_skipped': False,
              'torch_version': torch.__version__, 'limitations': [
                  'Loss-builder math and source identity are separate contracts.',
                  'No DDP/DeepSpeed collective execution or natural rollout claim.',
                  'Update oracle supports plain SGD only; AMP applies around the supplied loss builder.']}
    groups = optimizer.param_groups
    if (type(optimizer) is not torch.optim.SGD or any(group.get('momentum', 0) != 0 or
            group.get('weight_decay', 0) != 0 or group.get('differentiable', False) for group in groups)):
        report['reason'] = 'unsupported_optimizer_update_contract'
        return _seal(report)
    names, params = list(parameters), list(parameters.values())
    if (not params or any(not isinstance(name, str) or not name for name in names)
            or any(not isinstance(p, torch.Tensor) or not p.is_leaf or not p.requires_grad for p in params)
            or len({id(p) for p in params}) != len(params)):
        raise ValueError('parameters require distinct named floating trainable leaves')
    optimizer_params = [p for group in groups for p in group['params']]
    if len(optimizer_params) != len(params) or {id(p) for p in optimizer_params} != {id(p) for p in params}:
        raise ValueError('parameters must exactly cover optimizer parameters')
    batches = list(microbatches)
    weights = [1.]*len(batches) if accumulation_weights is None else list(accumulation_weights)
    if not batches or len(weights) != len(batches) or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('provide nonempty microbatches and finite nonnegative accumulation weights')
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm < 0):
        raise ValueError('max_grad_norm must be finite and nonnegative')
    if autocast_dtype not in {None, 'float16', 'bfloat16'} or device_type not in {'cpu', 'cuda'}:
        raise ValueError('unsupported autocast dtype/device')
    if any(p.device.type != device_type for p in params):
        raise ValueError('parameters must be on the declared device')
    lrs = {id(p): float(group['lr'])*(-1 if group.get('maximize', False) else 1)
           for group in groups for p in group['params']}
    if any(not math.isfinite(lr) for lr in lrs.values()):
        raise ValueError('learning rates must be finite')
    before = [p.detach().clone() for p in params]
    expected = [None]*len(params)
    optimizer.zero_grad(set_to_none=True)
    enabled_scaler = scaler is not None and scaler.is_enabled()
    scale_before = scaler.get_scale() if enabled_scaler else 1.
    losses, weighted_losses, scaled_losses, loss_dtypes = [], [], [], []
    for batch, weight in zip(batches, weights):
        with torch.autocast(device_type=device_type, enabled=autocast_dtype is not None,
                            dtype=getattr(torch, autocast_dtype) if autocast_dtype else None):
            loss = loss_builder(batch)
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not loss.requires_grad:
            raise ValueError('loss_builder must return a connected scalar tensor')
        weighted = loss*weight
        gradient = torch.autograd.grad(weighted, params, allow_unused=True, retain_graph=True)
        for index, g in enumerate(gradient):
            if g is not None:
                expected[index] = g.detach().clone() if expected[index] is None else expected[index]+g.detach()
        scaled = scaler.scale(weighted) if enabled_scaler else weighted
        scaled.backward()
        losses.append(loss.detach())
        weighted_losses.append(weighted.detach())
        scaled_losses.append(scaled.detach())
        loss_dtypes.append(str(loss.dtype))
    has_gradients = any(p.grad is not None for p in params)
    if enabled_scaler and has_gradients:
        scaler.unscale_(optimizer)
    unscaled = [p.grad.detach().clone() if p.grad is not None else None for p in params]
    finite = all(g is None or bool(torch.isfinite(g).all()) for g in unscaled)
    loss_finite = all(bool(torch.isfinite(loss).all()) for loss in losses)

    def close(a, b):
        if a is None or b is None:
            return a is None and b is None
        tolerance = .05 if a.dtype in (torch.float16, torch.bfloat16) else 2e-5 if a.dtype == torch.float32 else 1e-10
        return bool(torch.allclose(a, b, atol=tolerance, rtol=tolerance))

    accumulation_match = all(close(a, b) for a, b in zip(expected, unscaled)) if finite else None
    norm = None
    if max_grad_norm is not None:
        norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    clipped = [p.grad.detach().clone() if p.grad is not None else None for p in params]
    clip_match = None
    if finite:
        # Independent float64 scalar L2 norm and clipping coefficient.
        norm64 = 0.
        if max_grad_norm is not None:
            for g in unscaled:
                if g is not None:
                    for value in g.double().reshape(-1).tolist():
                        norm64 = math.hypot(norm64, value)
        coefficient = 1. if max_grad_norm is None else min(1., max_grad_norm/(norm64+1e-6))
        clip_match = all(close(None if g is None else g*coefficient, c) for g, c in zip(unscaled, clipped))
    if not has_gradients:
        skipped = True
    elif enabled_scaler:
        # GradScaler's real step path skips on the found-inf state from unscale.
        scaler.step(optimizer)
        scaler.update()
        skipped = not finite
    else:
        optimizer.step()
        skipped = False
    after = [p.detach().clone() for p in params]
    parameters_finite = all(bool(torch.isfinite(value).all()) for value in after)
    weighted_finite = all(bool(torch.isfinite(value).all()) for value in weighted_losses)
    scaled_finite = all(bool(torch.isfinite(value).all()) for value in scaled_losses)
    norm_finite = norm is None or bool(torch.isfinite(norm))
    clipped_finite = all(g is None or bool(torch.isfinite(g).all()) for g in clipped)
    numerical_stage = ('LOSS' if not loss_finite else 'WEIGHTED_LOSS' if not weighted_finite else
        'SCALED_LOSS' if not scaled_finite else 'UNSCALED_BACKWARD' if not finite else
        'GRADIENT_CLIP_NORM' if not norm_finite else 'CLIPPED_GRADIENT' if not clipped_finite else
        'PARAMETER_UPDATE' if not parameters_finite else 'NONE')
    update_match = (all(close(old if g is None else old-lrs[id(p)]*g, new)
                        for old, g, p, new in zip(before, clipped, params, after)) if finite else None)
    changed = any(not torch.equal(old, new) for old, new in zip(before, after))
    no_update = []
    if not changed and finite:
        if all(lr == 0 for lr in lrs.values()):
            no_update.append('zero_learning_rate')
        elif all(g is None or bool((g == 0).all()) for g in clipped):
            no_update.append('zero_or_unused_gradient')
        elif update_match:
            no_update.append('update_rounded_to_same_parameter_value')
    report.update(status='NUMERICAL_ANOMALY_OBSERVED' if numerical_stage!='NONE' else
                  'PASS' if accumulation_match and clip_match and update_match else 'CONTRACT_MISMATCH_OBSERVED',
                  optimizer_step_executed=not skipped, optimizer_step_skipped=skipped,
                  grad_scaler_enabled=enabled_scaler, scale_before=scale_before,
                  scale_after=scaler.get_scale() if enabled_scaler else 1.,
                  autocast_enabled=autocast_dtype is not None, autocast_dtype=autocast_dtype,
                  device_type=device_type, autocast_location='loss_builder_only',
                  microbatch_count=len(batches), accumulation_weights=weights,
                  microbatch_losses=losses, weighted_losses=weighted_losses, scaled_losses=scaled_losses,
                  loss_dtypes=loss_dtypes, loss_finite=loss_finite, unscaled_gradients_finite=finite,
                  accumulation_matches=accumulation_match, clip_matches=clip_match,
                  max_grad_norm=max_grad_norm, observed_preclip_norm=norm,
                  expected_unscaled_gradients=dict(zip(names, expected)),
                  unscaled_gradients=dict(zip(names, unscaled)), clipped_gradients=dict(zip(names, clipped)),
                  parameters_before=dict(zip(names, before)), parameters_after=dict(zip(names, after)),
                  parameter_states={name: 'UNUSED' if g is None else 'CONNECTED' for name, g in zip(names, unscaled)},
                  parameter_update_matches=update_match, parameters_changed=changed,
                  parameters_finite=parameters_finite,
                  numerical_failure_stage=numerical_stage,
                  normal_no_update_reasons=no_update)
    return _seal(report)
