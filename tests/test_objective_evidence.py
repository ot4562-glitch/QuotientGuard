import hashlib
import unittest
from pathlib import Path

import quotientguard as qg

try:
    import torch
except ImportError:
    torch = None


def local_loss(trace):
    # Explicitly a synthetic local control, never an external defect claim.
    return -2 * trace.current_log_probs.exp().mean()


@unittest.skipIf(torch is None, 'install quotientguard[objective]')
class ObjectiveEvidenceTests(unittest.TestCase):
    def test_detached_loss_is_insufficient_instead_of_backward_exception(self):
        trace = self.trace()
        capture = qg.ObjectiveRuntimeCapture(policy_snapshot_id='p', model_version='m',
            code_version='c', framework='fixture')
        result = capture.record_backward(qg.ObjectiveSpec(dtype='float64'), trace,
            trace.current_log_probs.detach().sum(), reproduction_id='r1')
        self.assertEqual(result.verdict, 'INSUFFICIENT_EVIDENCE')
        self.assertFalse(result.to_dict()['runtime_validation']['actual_backward'])
        self.assertIsNone(result.to_dict()['observed_coefficients'])

    def test_undeclared_old_policy_gradient_prevents_expected_verdict(self):
        trace = self.trace()
        trace.old_log_probs.requires_grad_(True)
        loss = -(trace.current_log_probs - trace.old_log_probs).exp().mean()
        capture = qg.ObjectiveRuntimeCapture(policy_snapshot_id='p', model_version='m',
            code_version='c', framework='fixture')
        result = capture.record_backward(qg.ObjectiveSpec(dtype='float64'), trace,
            loss, reproduction_id='r1')
        self.assertEqual(result.verdict, 'INSUFFICIENT_EVIDENCE')
        self.assertIn('declared_detach_boundary_violated:old_log_probs', result.to_dict()['reasons'])

    def trace(self):
        x = torch.tensor([[0., 0.]], dtype=torch.float64, requires_grad=True)
        return qg.ObjectiveTrace(x, torch.zeros_like(x), torch.ones_like(x), torch.ones_like(x))

    def test_source_verified_synthetic_still_never_confirms(self):
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        capture = qg.ObjectiveRuntimeCapture(policy_snapshot_id='p', model_version='m',
            code_version='c', framework='local-fixture', source_sha256=digest,
            execution_kind='FRAMEWORK_INTEGRATION', synthetic_inputs=True)
        trace = self.trace()
        for identifier in ('r1', 'r2', 'r3'):
            result = capture.record_backward(qg.ObjectiveSpec(dtype='float64'), trace,
                reproduction_id=identifier, production_loss=local_loss,
                reconstruction_loss=lambda: local_loss(trace),
                equivalence_contract={'id': 'fixture', 'version': '1', 'claim': 'PPO coefficient'},
                contract_verifier=lambda t: True)
            self.assertEqual(result.verdict, 'INSUFFICIENT_EVIDENCE')
            self.assertTrue(result.to_dict()['runtime_validation']['source_verified'])

    def test_unpinned_source_cannot_confirm_even_declared_real(self):
        capture = qg.ObjectiveRuntimeCapture(policy_snapshot_id='p', model_version='m',
            code_version='c', framework='fixture', execution_kind='FRAMEWORK_INTEGRATION',
            synthetic_inputs=False, source_sha256='0'*64)
        trace = self.trace()
        for identifier in ('r1', 'r2'):
            result = capture.record_backward(qg.ObjectiveSpec(dtype='float64'), trace,
                reproduction_id=identifier, production_loss=local_loss,
                reconstruction_loss=lambda: local_loss(trace),
                equivalence_contract={'id': 'fixture', 'version': '1', 'claim': 'PPO coefficient'},
                contract_verifier=lambda t: True)
            self.assertEqual(result.verdict, 'INSUFFICIENT_EVIDENCE')
            self.assertFalse(result.to_dict()['runtime_validation']['source_verified'])

    def test_capture_preserves_trainer_backward(self):
        trace = self.trace()
        x = trace.current_log_probs
        loss = -x.exp().mean()
        capture = qg.ObjectiveRuntimeCapture(policy_snapshot_id='p', model_version='m',
            code_version='c', framework='fixture')
        result = capture.record_backward(qg.ObjectiveSpec(dtype='float64'), trace,
            loss, reproduction_id='r1')
        self.assertIsNone(x.grad)
        loss.backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([[-.5, -.5]], dtype=x.dtype)))
        self.assertEqual(result.verdict, 'EXPECTED_OBJECTIVE_EFFECT')

    def test_finite_forward_nonfinite_backward(self):
        trace = self.trace()
        x = trace.current_log_probs
        loss = torch.sqrt(x).sum()
        self.assertTrue(torch.isfinite(loss).item())
        capture = qg.ObjectiveRuntimeCapture(policy_snapshot_id='p', model_version='m',
            code_version='c', framework='fixture')
        result = capture.record_backward(qg.ObjectiveSpec(dtype='float64'), trace,
                                        loss, reproduction_id='r1')
        self.assertEqual(result.verdict, 'NUMERICAL_ANOMALY_OBSERVED')

    def test_disconnected_nonfinite_loss_keeps_numerical_priority(self):
        trace = self.trace()
        other = torch.tensor(float('nan'), requires_grad=True)
        capture = qg.ObjectiveRuntimeCapture(policy_snapshot_id='p', model_version='m',
            code_version='c', framework='fixture')
        result = capture.record_backward(qg.ObjectiveSpec(dtype='float64'), trace,
                                        other * 2, reproduction_id='r1')
        self.assertEqual(result.verdict, 'NUMERICAL_ANOMALY_OBSERVED')

    def test_real_global_norm_clipping_scale_is_explained(self):
        trace = self.trace()
        x = trace.current_log_probs
        loss = -x.exp().mean()
        loss.backward()
        norm_before = x.grad.norm().item()
        torch.nn.utils.clip_grad_norm_([x], max_norm=.1)
        scale = x.grad.norm().item() / norm_before
        trace.observed_coefficients = x.grad
        trace.observed_loss = loss
        result = qg.audit_objective(qg.ObjectiveSpec(dtype='float64', gradient_scale=scale),
                                    trace, oracle=True)
        self.assertEqual(result.verdict, 'EXPECTED_OBJECTIVE_EFFECT')

    def test_nonfinite_parameter_gradients_are_not_hidden_by_surrogate(self):
        trace = self.trace()
        trace.observed_coefficients = [[-.5, -.5]]
        trace.parameter_gradients = [0., float('nan')]
        result = qg.audit_objective(qg.ObjectiveSpec(dtype='float64'), trace, oracle=True)
        self.assertEqual(result.verdict, 'NUMERICAL_ANOMALY_OBSERVED')

    def test_mixed_precision_and_differentiable_weights_fail_closed(self):
        trace = self.trace()
        trace.observed_coefficients = [[-.5, -.5]]
        result = qg.audit_objective(qg.ObjectiveSpec(dtype='float32'), trace, oracle=True)
        self.assertEqual(result.verdict, 'INSUFFICIENT_EVIDENCE')
        result = qg.audit_objective(qg.ObjectiveSpec(dtype='float64', detach_importance=False), trace)
        self.assertEqual(result.verdict, 'INSUFFICIENT_EVIDENCE')


if __name__ == '__main__':
    unittest.main()
