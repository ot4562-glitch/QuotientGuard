"""Compare supplied parameter graphs using independent loss-coordinate partials."""
from dataclasses import replace

from .native_validation import _seal
from .full_objective import audit_full_objective


def validate_parameter_graph(spec, trace, loss, parameters):
    """Read parameter gradients using a measured model Jacobian.

    Loss coefficients are independently analytic; the upstream model Jacobian
    is measured by Torch, not independently certified. Coordinates must be
    distinct sibling tensors: ancestor/descendant coordinates are unsupported
    to avoid double-counting a shared path. Retained named nonleaves are rejected
    before probes. Parameter .grad is not accumulated; arbitrary upstream hooks
    and unlisted retained intermediates remain the caller's responsibility.
    No callable or optimizer executes.
    """
    import torch

    report = {'schema_version':'quotientguard-parameter-validation-v1', 'status':'UNSUPPORTED',
              'execution_scope':'SUPPLIED_PARAMETER_GRAPH', 'reasons':[],
              'semantic_confirmation_eligible':False, 'trainer_executed':False,
              'optimizer_step_executed':False, 'model_jacobian':'MEASURED_AUTOGRAD_VJP',
              'limitations':['The supplied graph is not source-attested by this API.',
                  'The loss-coordinate partials are independent; the model Jacobian uses the actual supplied graph.',
                  'Only independent sibling input coordinates without retained named nonleaf grads are supported.',
                  'Autograd hooks and unlisted retained upstream intermediate gradients can execute during VJP probes.']}
    names, params = list(parameters), list(parameters.values())
    if not params or len({id(p) for p in params}) != len(params) or any(
            not isinstance(p,torch.Tensor) or not p.requires_grad or not p.is_leaf for p in params):
        raise ValueError('parameters require distinct named trainable leaves')
    if not isinstance(loss,torch.Tensor) or loss.numel()!=1 or not loss.requires_grad:
        raise ValueError('loss must be a connected scalar tensor')
    p = trace.policy
    raw = {'current_log_probs':p.current_log_probs,'old_log_probs':p.old_log_probs,
           'advantages':p.advantages,'importance_weights':p.importance_weights,
           'reference_log_probs':trace.reference_log_probs,'entropy':trace.entropy,
           'kl_current_log_probs':trace.kl_current_log_probs,'rollout_log_probs':trace.rollout_log_probs}
    all_named = [value for value in raw.values() if isinstance(value,torch.Tensor) and value.requires_grad]
    if any(not value.is_leaf and value.retains_grad for value in all_named):
        report['reasons']=['retained_coordinate_gradient']
        return _seal(report)
    audit = audit_full_objective(spec,replace(trace,observed_loss=loss),device=str(loss.device)).to_dict()
    report['full_objective'] = audit
    if audit['status']!='SUPPORTED':
        report.update(status=audit['status'],reasons=['full_objective_not_supported'])
        return _seal(report)
    entries = audit['analytic']['gradients']
    coordinates, coordinate_names, coefficients = [], [], []
    frozen = []
    for name,entry in entries.items():
        value = raw[name]
        if entry['state']=='CONNECTED':
            if not isinstance(value,torch.Tensor) or not value.requires_grad:
                report['reasons'].append('missing_live_coordinate:'+name)
                continue
            if value.dtype != getattr(torch,spec.policy.dtype) or value.device != loss.device:
                report['reasons'].append('coordinate_precision_or_device_mismatch:'+name)
                continue
            coordinate_names.append(name)
            coordinates.append(value)
            coefficients.append(torch.tensor(entry['values'],dtype=value.dtype,device=value.device))
        elif isinstance(value,torch.Tensor) and value.requires_grad:
            frozen.append((name,value))
    if report['reasons']:
        return _seal(report)
    if len({id(value) for value in all_named})!=len(all_named):
        report['reasons']=['aliased_input_coordinates']
        return _seal(report)
    for index,value in enumerate(all_named):
        others=[other for j,other in enumerate(all_named) if j!=index]
        if others and any(g is not None for g in torch.autograd.grad(value,others,
                grad_outputs=torch.ones_like(value),retain_graph=True,allow_unused=True)):
            report['reasons']=['dependent_input_coordinates']
            return _seal(report)
    expected = torch.autograd.grad(coordinates,params,grad_outputs=coefficients,allow_unused=True,retain_graph=True)
    observed = torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
    partials = torch.autograd.grad(loss,coordinates,allow_unused=True,retain_graph=True)
    frozen_gradients = (torch.autograd.grad(loss,[value for _,value in frozen],allow_unused=True,retain_graph=True)
                        if frozen else [])
    observations = dict(zip(coordinate_names,partials))
    observations.update({name:g for (name,_),g in zip(frozen,frozen_gradients)})
    final_audit = audit_full_objective(spec,replace(trace,observed_loss=loss,observed_gradients=observations),device=str(loss.device)).to_dict()
    report['full_objective'] = final_audit
    tolerance = {'float16':.005,'bfloat16':.04,'float32':2e-5,'float64':1e-10}[spec.policy.dtype]

    def close(a,b):
        return a is b if a is None or b is None else bool(torch.allclose(a,b,atol=tolerance,rtol=tolerance))

    finite=all(g is None or bool(torch.isfinite(g).all()) for g in (*expected,*observed,*partials,*frozen_gradients))
    partial_match=final_audit['status']=='SUPPORTED'
    parameter_match=all(close(a,b) for a,b in zip(expected,observed))
    frozen_match=all(g is None for g in frozen_gradients)
    report.update(status='NUMERICAL_ANOMALY_OBSERVED' if not finite else
                  'PASS' if partial_match and parameter_match and frozen_match else 'CONTRACT_MISMATCH_OBSERVED',
                  expected_gradients=dict(zip(names,expected)),observed_gradients=dict(zip(names,observed)),
                  coordinate_names=coordinate_names,coordinate_partials_match=partial_match,
                  parameter_gradients_match=parameter_match,frozen_coordinates_detached=frozen_match,
                  parameter_states={name:'UNUSED' if g is None else 'CONNECTED' for name,g in zip(names,observed)},
                  gradients_finite=finite)
    return _seal(report)
