import json
import math
from pathlib import Path
import tempfile
import unittest

import quotientguard as qg

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'install quotientguard[objective]')
class ObjectiveRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(qg, 'ObjectiveRuntimeCapture'), 'runtime backward capture is missing')

    def capture(self):
        return qg.ObjectiveRuntimeCapture(policy_snapshot_id='p1', model_version='m1',
            code_version='local-test-source', framework='synthetic-runtime')

    def record(self, capture, *, factor=1., reproduction_id='r1', validated=False):
        x = torch.tensor([[0., 0.]], dtype=torch.float64, requires_grad=True)
        loss = -factor * x.exp().mean()
        trace = qg.ObjectiveTrace(x, torch.zeros_like(x), torch.ones_like(x), torch.ones_like(x))
        return capture.record_backward(qg.ObjectiveSpec(dtype='float64'), trace, loss,
            reproduction_id=reproduction_id,
            equivalence_contract={'id': 'same-logprob-derivative', 'version': '1',
                'claim': 'declared PPO derivative is the optimization coefficient'},
            contract_verifier=(lambda t: bool(torch.equal(t.current_log_probs,
                t.old_log_probs))) if validated else None,
            reconstruction_loss=(lambda: -factor * x.exp().mean()) if validated else None)

    def test_synthetic_backward_never_confirms_framework_semantics(self):
        capture = self.capture()
        first = self.record(capture, factor=2., validated=True)
        self.assertEqual(first.verdict, 'INSUFFICIENT_EVIDENCE')
        second = self.record(capture, factor=2., validated=True, reproduction_id='r2')
        self.assertEqual(second.verdict, 'INSUFFICIENT_EVIDENCE')
        self.assertTrue(second.to_dict()['runtime_validation']['reconstruction_validation'])

    def test_unverified_and_duplicate_repetitions_never_confirm(self):
        for validated in (True, False):
            capture = self.capture()
            self.record(capture, factor=2., validated=validated)
            result = self.record(capture, factor=2., validated=validated)
            self.assertEqual(result.verdict, 'INSUFFICIENT_EVIDENCE')

    def test_expected_clip_difference_preserves_v1_status(self):
        capture = self.capture()
        x = torch.tensor([[math.log(1.5), 0.]], dtype=torch.float64, requires_grad=True)
        ratio = x.exp()
        loss = -torch.minimum(ratio, ratio.clamp(.8, 1.2)).mean()
        trace = qg.ObjectiveTrace(x, torch.zeros_like(x), torch.ones_like(x), torch.ones_like(x))
        result = capture.record_backward(qg.ObjectiveSpec(dtype='float64'), trace, loss,
                                        reproduction_id='r1')
        self.assertEqual(result.verdict, 'EXPECTED_OBJECTIVE_EFFECT')
        legacy = qg.CallbackRuntimeCapture(framework='custom', framework_version='1',
                                           policy_snapshot_id='p1')
        for index, coeff in enumerate((0., -.5)):
            legacy.record(batch_id='b', sample_index=index, class_id='same',
                trajectory_id=str(index), realized_coefficient=coeff,
                state_consuming_point='gradient_contribution')
        before = legacy.certificate()
        legacy.attach_objective_audit(result)
        after = legacy.certificate()
        self.assertEqual(before['classes'], after['classes'])
        self.assertEqual(after['classes'][0]['compatibility_status'], 'COEFFICIENT_MISMATCH_OBSERVED')
        self.assertIn('objective_interpretation', after)
        self.assertFalse(after['estimator']['reconstruction_validation'])
        from quotientguard.provenance import sha256_json
        self.assertEqual(after['certificate_sha256'], sha256_json(
            {k: v for k, v in after.items() if k != 'certificate_sha256'}))
        original_verdict = after['objective_interpretation']['audits'][0]['verdict']
        after['objective_interpretation']['audits'][0]['verdict'] = 'SEMANTIC_MISMATCH_CONFIRMED'
        self.assertEqual(legacy.certificate()['objective_interpretation']['audits'][0]['verdict'],
                         original_verdict)

    def test_actual_runtime_nan_gradient_and_zero_mask(self):
        capture = self.capture()
        x = torch.tensor([[1000., 0.]], requires_grad=True)
        mask = torch.tensor([[0., 1.]])
        loss = -(x.exp() * mask).sum()
        result = capture.record_backward(qg.ObjectiveSpec(reduction='sum'),
            qg.ObjectiveTrace(x, torch.zeros_like(x), torch.ones_like(x), mask),
            loss, reproduction_id='r1')
        self.assertEqual(result.verdict, 'NUMERICAL_ANOMALY_OBSERVED')
        self.assertFalse(result.to_dict()['excluded_gradients_zero'])
        json.dumps(result.to_dict(), allow_nan=False)


class ObjectiveCLITests(unittest.TestCase):
    def test_json_and_report_cli(self):
        from quotientguard.cli import main
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / 'input.json'
            fixture.write_text(json.dumps({'spec': {'reduction': 'sum'}, 'trace': {
                'current_log_probs': [[0]], 'old_log_probs': [[0]],
                'advantages': [[1]], 'mask': [[1]], 'observed_coefficients': [[-1]]}}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['objective-audit', str(fixture), '--out', str(root/'out.json'),
                                       '--markdown', str(root/'report.md')]), 0)
            self.assertEqual(json.loads((root/'out.json').read_text())['verdict'],
                             'EXPECTED_OBJECTIVE_EFFECT')
            self.assertIn('EXPECTED_OBJECTIVE_EFFECT', (root/'report.md').read_text())


if __name__ == '__main__':
    unittest.main()
