"""Source-bound controlled callable execution, never a training-reachability claim.

This module does not import frameworks, download code, or install dependencies.
Integrators review and import a callable first, then adapt its arguments with
``invoke(callable, inputs)``. Run untrusted integrations in a separate process.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import fields, is_dataclass
from pathlib import Path
import sys
from types import CodeType
from typing import Any, Callable, Mapping

from .provenance import sha256_json


_GRADES = {
    'INDEPENDENT_TORCH': 'INDEPENDENT_TORCH_CONTROLLED',
    'SOURCE_EQUIVALENT': 'SOURCE_EQUIVALENT_CONTROLLED',
    'AST_EXTRACTED': 'AST_EXTRACTED_CONTROLLED',
    'NATIVE_IMPORTED': 'UNVERIFIED_CALLABLE_CONTROLLED',
}
_LIMITATIONS = [
    'Controlled inputs do not establish naturally encountered training failures or input reachability.',
    'No semantic mismatch confirmation or autonomous discovery is established by this report.',
    'Execution kind, commit identity, input validity and adapter correctness are integrator assertions.',
    'Source hashes bind recorded files, not transitive dependencies or an entire installed framework.',
    'Python call observation is local evidence, not remote attestation or a security sandbox.',
    'Metamorphic results cover only the explicitly executed branch, dtype, device and mutations.',
]


def _json_value(value: Any) -> Any:
    if hasattr(value, 'detach'):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value


def _seal(report: dict[str, Any]) -> dict[str, Any]:
    report = _json_value(report)
    report['report_sha256'] = sha256_json(report)
    return report


def _loaded_code_matches_source(code, path):
    """Compile without executing; reject stale/replaced code at the source gate."""
    if code is None or not path:
        return False
    try:
        content = Path(path).read_bytes()
        if len(content) > 2_000_000:
            return False
        compiled = compile(content, str(path), 'exec', dont_inherit=True, optimize=sys.flags.optimize)
        pending = [compiled]
        while pending:
            candidate = pending.pop()
            if candidate == code:
                return True
            pending.extend(item for item in candidate.co_consts if isinstance(item, CodeType))
    except (OSError, ValueError, SyntaxError, RecursionError):
        return False
    return False


def _returned_tensors(value, torch, depth=0):
    """Inspect standard return containers only; never evaluate arbitrary properties."""
    if isinstance(value, torch.Tensor):
        return [value]
    if depth >= 8:
        return []
    if isinstance(value, Mapping):
        values = list(value.values())[:128]
    elif isinstance(value, (tuple, list)):
        values = value[:128]
    elif is_dataclass(value) and not isinstance(value, type):
        values = [getattr(value, field.name) for field in fields(value)[:128]]
    else:
        return []
    result = []
    for item in values:
        result.extend(_returned_tensors(item, torch, depth + 1))
        if len(result) >= 128:
            break
    return result[:128]


def _loss_origin(loss, returned):
    """Require the audited loss to descend from an observed callable return.

    Sharing an input leaf with a discarded native result is insufficient. Keep
    graph node objects alive during traversal so Python object IDs cannot alias.
    This verifies a local autograd path, not objective semantics or input origin.
    """
    if any(loss is tensor for tensor in returned):
        return 'DIRECT_RETURN'
    targets = {tensor.grad_fn for tensor in returned if tensor.grad_fn is not None}
    leaves = [tensor for tensor in returned if tensor.requires_grad and tensor.is_leaf]
    pending = [loss.grad_fn] if loss.grad_fn is not None else []
    visited = set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        if node in targets or any(getattr(node, 'variable', None) is leaf for leaf in leaves):
            return 'AUTOGRAD_DESCENDANT'
        if len(visited) >= 10_000:
            return 'GRAPH_LIMIT'
        pending.extend(parent for parent, _ in node.next_functions if parent is not None)
    return 'UNVERIFIED'


def not_executed_report(*, framework: str, path_id: str, reason: str,
                        source_commit: str | None = None,
                        source_sha256: str | None = None) -> dict[str, Any]:
    """Record an unavailable requested path without substituting a replay for it."""
    if not framework.strip() or not path_id.strip() or not reason.strip():
        raise ValueError('framework, path_id and reason are required')
    return _seal({'schema_version': 'quotientguard-native-validation-v1',
        'framework': framework, 'evidence_grade': 'NOT_EXECUTED',
        'execution_scope': 'LOSS_CALLABLE', 'trainer_executed': False,
        'controlled_inputs': True, 'semantic_confirmation_eligible': False,
        'source': {'commit': source_commit, 'expected_sha256': source_sha256},
        'paths': [{'path_id': path_id, 'status': 'NOT_EXECUTED', 'reason': reason}],
        'runs': [], 'limitations': list(_LIMITATIONS)})


def validate_callable(loss_callable: Callable, inputs: Mapping[str, Any], *,
                      framework: str, path_id: str,
                      invoke: Callable | None = None,
                      execution_kind: str = 'INDEPENDENT_TORCH',
                      source_sha256: str | None = None,
                      source_commit: str | None = None,
                      mutations: Mapping[str, Mapping[str, float]] | None = None,
                      gradient_inputs: tuple[str, ...] = (),
                      objective_contract=None,
                      raw_tensors: Mapping[str, Any] | None = None,
                      sanitized_tensors: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Execute a baseline and excluded-token mutations, capturing actual derivatives.

    ``inputs`` requires same-shaped floating ``current_log_probs`` and binary
    ``mask`` tensors. Other tensor arguments are cloned and detached for every
    run. Current log probabilities and explicitly named ``gradient_inputs``
    require gradients. Extra inputs permit observation of differentiable
    corrections and auxiliary branches without declaring their math supported.
    Their gradients use the supplied input's units, not logprob coefficients.
    Missing gradients remain missing; original tensors and .grad are untouched.
    The default call is
    ``loss_callable(inputs)``. An adapter may return a scalar, a tuple whose first
    element is the loss, or a dictionary containing ``loss`` and optional raw
    and sanitized tensor observations. Observations are supplied by the adapter;
    their graph provenance is not automatically verified.

    Mutation values replace only excluded positions of named, same-shaped input
    tensors. Metamorphic invariance is exact, including zero excluded gradients.
    All results remain controlled evidence, including source-pinned imports.
    """
    import torch

    if objective_contract is not None:
        from .full_objective import FullObjectiveSpec
        if not isinstance(objective_contract, FullObjectiveSpec):
            raise ValueError('objective_contract must be FullObjectiveSpec')
        # Observe detached as well as live coordinates, so a connected zero is
        # not silently accepted as an absent path. Existing callers are unchanged.
        contract_inputs = ('old_log_probs', 'advantages', 'importance_weights', 'reference_log_probs', 'entropy',
                           'kl_current_log_probs', 'rollout_log_probs')
        gradient_inputs = tuple(dict.fromkeys((*gradient_inputs, *(name for name in contract_inputs if name in inputs))))

    if not framework.strip() or not path_id.strip():
        raise ValueError('framework and path_id are required')
    if execution_kind not in _GRADES:
        raise ValueError('unsupported execution_kind')
    x, mask = inputs.get('current_log_probs'), inputs.get('mask')
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise ValueError('current_log_probs must be a floating tensor')
    if (not isinstance(mask, torch.Tensor) or mask.shape != x.shape or
            mask.device != x.device or not bool(((mask == 0) | (mask == 1)).all())):
        raise ValueError('mask must be binary and match current_log_probs shape/device')
    if (not isinstance(gradient_inputs, (tuple, list)) or
            any(not isinstance(name, str) for name in gradient_inputs) or
            len(set(gradient_inputs)) != len(gradient_inputs)):
        raise ValueError('gradient_inputs must contain distinct auxiliary tensor names')
    for name in gradient_inputs:
        tensor = inputs.get(name)
        if (name in ('mask', 'current_log_probs') or not isinstance(tensor, torch.Tensor)
                or not tensor.is_floating_point()):
            raise ValueError('gradient_inputs require named floating tensors other than policy logprobs/mask')
    mutation_items = list((mutations or {}).items())
    if any(name == 'baseline' for name, _ in mutation_items):
        raise ValueError('baseline is a reserved mutation name')
    for _, replacements in mutation_items:
        for name in replacements:
            tensor = inputs.get(name)
            if (name == 'mask' or not isinstance(tensor, torch.Tensor) or
                    tensor.shape != x.shape or not tensor.is_floating_point()):
                raise ValueError('mutations require same-shaped floating input tensors, excluding mask')

    target = getattr(loss_callable, '__func__', loss_callable)
    if not inspect.isfunction(target):
        target = getattr(getattr(loss_callable, 'forward', None), '__func__', None)
    code = getattr(target, '__code__', None)
    path = inspect.getsourcefile(target) if target is not None else None
    actual_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None
    module = sys.modules.get(getattr(target, '__module__', ''))
    module_path = getattr(module, '__file__', None)
    imported_source = bool(path and module_path and Path(path).resolve() == Path(module_path).resolve())
    declared = module
    for component in getattr(target, '__qualname__', '').split('.'):
        declared = getattr(declared, component, None)
    callable_identity_verified = target is not None and declared is target
    source = {'path': str(Path(path).resolve()) if path else None,
        'callable': getattr(target, '__qualname__', type(loss_callable).__qualname__),
        'module': getattr(target, '__module__', None), 'commit': source_commit,
        'expected_sha256': source_sha256, 'actual_sha256': actual_hash,
        'hash_verified': bool(source_sha256 and actual_hash == source_sha256),
        'loaded_code_matches_source': _loaded_code_matches_source(code, path),
        'imported_source_module': imported_source,
        'callable_identity_verified': callable_identity_verified}
    # Record explicit decorator links, without relaxing the entry-point gate.
    chain = []
    member = target
    while inspect.isfunction(member) and all(member is not item for item in chain) and len(chain) < 16:
        chain.append(member)
        member = getattr(member, '__wrapped__', None)
    source['wrapper_chain'] = [
        {'callable': item.__qualname__, 'code_qualname': getattr(item.__code__, 'co_qualname', item.__code__.co_name),
         'module': item.__module__, 'path': inspect.getsourcefile(item),
         'loaded_code_matches_source': _loaded_code_matches_source(item.__code__, inspect.getsourcefile(item))}
        for item in chain]

    active = mask != 0
    runs = []
    baseline_loss = baseline_gradient = None
    for name, replacements in [('baseline', {}), *mutation_items]:
        cloned = {key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                  for key, value in inputs.items()}
        for key, replacement in replacements.items():
            cloned[key] = cloned[key].masked_fill(~active, replacement)
        current = cloned['current_log_probs'].requires_grad_(True)
        extra_targets = [cloned[key].requires_grad_(True) for key in gradient_inputs]
        active_unchanged = all(torch.equal(cloned[key][active], value.detach()[active])
            for key, value in inputs.items() if isinstance(value, torch.Tensor) and value.shape == x.shape)
        called = False
        chain_calls = {item.__code__: 0 for item in chain}
        returned = []
        previous_profile = sys.getprofile()

        def observe(frame, event, arg):
            nonlocal called
            if event == 'call' and frame.f_code is code:
                called = True
            if event == 'call' and frame.f_code in chain_calls:
                chain_calls[frame.f_code] += 1
            if event == 'return' and frame.f_code is code:
                returned.extend(_returned_tensors(arg, torch))
            if previous_profile is not None:
                previous_profile(frame, event, arg)

        run = {'name': name, 'mutations': dict(replacements),
               'inputs': _json_value(cloned), 'active_inputs_unchanged': active_unchanged,
               'dtype': str(current.dtype), 'device': str(current.device)}
        try:
            sys.setprofile(observe)
            try:
                output = invoke(loss_callable, cloned) if invoke is not None else loss_callable(cloned)
            finally:
                sys.setprofile(previous_profile)
            observations = output if isinstance(output, dict) else {}
            loss = observations['loss'] if observations else (output[0] if isinstance(output, tuple) else output)
            if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                raise ValueError('callable must return a scalar tensor loss')
            loss_origin = _loss_origin(loss, returned)
            gradients = (torch.autograd.grad(loss, (current, *extra_targets), allow_unused=True)
                         if loss.requires_grad else (None,) * (1 + len(extra_targets)))
            gradient = gradients[0]
            if current.is_cuda:
                torch.cuda.synchronize(current.device)
            finite_loss = bool(torch.isfinite(loss).all())
            finite_gradient = bool(torch.isfinite(gradient).all()) if gradient is not None else None
            input_gradients = {
                key: {'gradient': value, 'graph_connected': value is not None,
                      'finite': bool(torch.isfinite(value).all()) if value is not None else None,
                      'dtype': str(cloned[key].dtype), 'device': str(cloned[key].device)}
                for key, value in zip(gradient_inputs, gradients[1:])}
            monitored_states = [finite_gradient, *(item['finite'] for item in input_gradients.values())]
            finite_monitored = (False if False in monitored_states else
                                None if None in monitored_states else True)
            nonfinite_backward = finite_monitored is False
            excluded_zero = bool((gradient[~active] == 0).all()) if gradient is not None else None
            if name == 'baseline':
                baseline_loss = loss.detach().clone()
                baseline_gradient = gradient.detach().clone() if gradient is not None else None
            loss_same = baseline_loss is not None and bool(torch.equal(loss.detach(), baseline_loss))
            active_same = (gradient is not None and baseline_gradient is not None and
                           bool(torch.equal(gradient[active], baseline_gradient[active])))
            run.update(status='EXECUTED', loss=loss, current_gradient=gradient,
                backward_executed=gradient is not None, graph_connected=gradient is not None,
                loss_origin=loss_origin,
                loss_derived_from_callable_return=loss_origin in ('DIRECT_RETURN', 'AUTOGRAD_DESCENDANT'),
                finite_loss=finite_loss, finite_current_gradient=finite_gradient,
                input_gradients=input_gradients, finite_monitored_gradients=finite_monitored,
                excluded_gradients_exactly_zero=excluded_zero,
                loss_unchanged=loss_same, active_gradients_unchanged=active_same,
                mask_invariance_passed=bool(finite_loss and finite_gradient and finite_monitored is not False and excluded_zero and
                                          active_unchanged and loss_same and active_same),
                numerical_status=('NONFINITE_OBSERVED' if not finite_loss or nonfinite_backward
                                  else ('FINITE_OBSERVED' if finite_gradient else 'BACKWARD_MISSING')),
                numerical_failure_stage=('FORWARD_AND_BACKWARD' if not finite_loss and nonfinite_backward
                    else 'FORWARD' if not finite_loss else 'BACKWARD_ONLY' if nonfinite_backward
                    else 'BACKWARD_MISSING' if finite_gradient is None else 'NONE'),
                raw_tensors=observations.get('raw_tensors', raw_tensors or {}),
                sanitized_tensors=observations.get('sanitized_tensors', sanitized_tensors or {}))
            if objective_contract is not None:
                from .objective import ObjectiveTrace
                from .full_objective import FullObjectiveTrace, audit_full_objective
                policy_trace = ObjectiveTrace(*(cloned[key] for key in
                    ('current_log_probs', 'old_log_probs', 'advantages', 'mask')),
                    importance_weights=cloned.get('importance_weights'))
                full_trace = FullObjectiveTrace(policy_trace,
                    **{key: cloned[key] for key in ('reference_log_probs', 'entropy', 'policy_filter', 'kl_mask', 'entropy_mask',
                        'kl_current_log_probs', 'rollout_log_probs') if key in cloned},
                    observed_loss=loss, observed_terms=observations.get('loss_terms', {}),
                    observed_gradients={'current_log_probs': gradient,
                        **{key: item['gradient'] for key, item in input_gradients.items() if key in
                           ('old_log_probs', 'advantages', 'importance_weights', 'reference_log_probs', 'entropy',
                            'kl_current_log_probs', 'rollout_log_probs')}})
                comparison = audit_full_objective(objective_contract, full_trace, device=str(current.device))
                # Input-coordinate replay can convert values, but that cannot
                # attest a native homogeneous-precision contract with other
                # floating dtypes. Masks/filters are nondifferentiable controls.
                actual_dtypes = {key: str(cloned[key].dtype) for key in
                    ('current_log_probs', 'old_log_probs', 'advantages', 'importance_weights', 'reference_log_probs', 'entropy',
                     'kl_current_log_probs', 'rollout_log_probs')
                    if key in cloned and isinstance(cloned[key], torch.Tensor)}
                if any(value != 'torch.'+objective_contract.policy.dtype for value in actual_dtypes.values()):
                    from .full_objective import FullObjectiveResult
                    comparison = FullObjectiveResult({**comparison.evidence, 'status': 'UNSUPPORTED',
                        'reasons': [*comparison.evidence['reasons'], 'native_input_dtype_differs_from_contract'],
                        'native_input_dtypes': actual_dtypes})
                run['full_objective'] = comparison.to_dict()
        except Exception as exc:
            run.update(status='EXECUTION_ERROR', error_type=type(exc).__name__, error=str(exc),
                       backward_executed=False, mask_invariance_passed=False)
        run['callable_executed'] = called
        run['callable_chain'] = [
            {'code_qualname': getattr(item.__code__, 'co_qualname', item.__code__.co_name),
             'call_count': chain_calls[item.__code__], 'executed': chain_calls[item.__code__] > 0}
            for item in chain]
        runs.append(_json_value(run))

    grade = _GRADES[execution_kind]
    if (execution_kind == 'NATIVE_IMPORTED' and source['hash_verified'] and source_commit and
            source['loaded_code_matches_source'] and
            imported_source and callable_identity_verified and
            all(run['callable_executed'] and run['status'] == 'EXECUTED' and
                run['backward_executed'] and run['loss_derived_from_callable_return'] for run in runs)):
        grade = 'NATIVE_CONTROLLED_REPRODUCED'
    status = 'EXECUTED' if all(run['status'] == 'EXECUTED' for run in runs) else 'EXECUTION_ERROR'
    return _seal({'schema_version': 'quotientguard-native-validation-v1',
        'framework': framework, 'execution_kind': execution_kind, 'source': source,
        'execution_scope': 'LOSS_CALLABLE', 'trainer_executed': False,
        'evidence_grade': grade, 'controlled_inputs': True, 'semantic_confirmation_eligible': False,
        'gradient_inputs': list(gradient_inputs),
        'paths': [{'path_id': path_id, 'status': status}], 'runs': runs,
        'environment': {'torch_version': torch.__version__, 'python_version': sys.version.split()[0],
                        'cuda_available': torch.cuda.is_available()},
        'limitations': [*_LIMITATIONS,
            'Autograd ancestry verifies a local return-to-loss path, not the declared objective, raw coefficients, parameter updates or training reachability.',
            'Raw/sanitized observations are adapter-supplied and not automatically proven to be graph intermediates.']})


def write_native_report(report: Mapping[str, Any], path: str | Path) -> None:
    """Write strict JSON only after checking the report's content-addressed hash."""
    payload = {key: value for key, value in report.items() if key != 'report_sha256'}
    if report.get('report_sha256') != sha256_json(payload):
        raise ValueError('native validation report hash mismatch')
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def validate_training_step(*args, **kwargs):
    """Execute controlled AMP/accumulation/clipping/plain SGD; mutates the model.

    The separate helper is lazy so ordinary callable validation imports no new
    dependency or optimizer runtime. See training_validation for the contract.
    """
    from .training_validation import validate_training_step as validate
    return validate(*args, **kwargs)


def validate_parameter_graph(*args, **kwargs):
    """Check analytic partials through a supplied model Jacobian without accumulating parameter .grad."""
    from .parameter_validation import validate_parameter_graph as validate
    return validate(*args, **kwargs)
