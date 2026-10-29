import json
import math
import unittest

import quotientguard as qg


class ObjectiveAnalyticTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(qg, 'audit_objective'), 'objective-aware API is missing')

    def trace(self, ratios, advantages, **kwargs):
        return qg.ObjectiveTrace(
            current_log_probs=[[math.log(r) for r in ratios]],
            old_log_probs=[[0.0] * len(ratios)],
            advantages=[advantages], mask=[[1] * len(ratios)], **kwargs)

    def test_clip_signs_and_zero_gradients(self):
        result = qg.audit_objective(qg.ObjectiveSpec(reduction='sum'),
            self.trace([1.5, .5, 1.5, .5], [1, 1, -1, -1],
                       observed_coefficients=[[0, -.5, 1.5, 0]]))
        self.assertEqual(result.verdict, 'EXPECTED_OBJECTIVE_EFFECT')
        self.assertEqual(result.to_dict()['expected_coefficients'], [[0, -.5, 1.5, 0]])
        self.assertEqual(result.to_dict()['clip_branches'],
                         [['upper_clipped', 'active', 'active', 'lower_clipped']])

    def test_manual_mismatch_never_confirms_semantics(self):
        result = qg.audit_objective(qg.ObjectiveSpec(),
            self.trace([1.0], [1.0], observed_coefficients=[[99.0]],
                       provenance={'kind': 'FRAMEWORK_RUNTIME', 'reconstruction_validation': True}))
        self.assertEqual(result.verdict, 'INSUFFICIENT_EVIDENCE')

    def test_no_observation_is_insufficient(self):
        self.assertEqual(qg.audit_objective(qg.ObjectiveSpec(),
            self.trace([1], [1])).verdict, 'INSUFFICIENT_EVIDENCE')

    def test_unsupported_detach_and_objective_fail_closed(self):
        for spec in [qg.ObjectiveSpec(objective='grpo'),
                     qg.ObjectiveSpec(detach_advantage=False),
                     qg.ObjectiveSpec(detach_old_policy=False)]:
            with self.subTest(spec=spec):
                self.assertEqual(qg.audit_objective(spec, self.trace([1], [1],
                    observed_coefficients=[[-1]])).verdict, 'INSUFFICIENT_EVIDENCE')

    def test_strict_json_nonfinite(self):
        result = qg.audit_objective(qg.ObjectiveSpec(),
            self.trace([1], [1], observed_coefficients=[[float('nan')]]))
        self.assertEqual(result.verdict, 'NUMERICAL_ANOMALY_OBSERVED')
        json.dumps(result.to_dict(), allow_nan=False)

    def test_mask_normalization_importance_and_gradient_scale(self):
        trace = qg.ObjectiveTrace(current_log_probs=[[0, 0, 0]],
            old_log_probs=[[0, 0, 0]], advantages=[[2, -2, 3]], mask=[[1, 1, 0]],
            importance_weights=[[2, 3, 4]], observed_coefficients=[[-.5, .75, 0]])
        result = qg.audit_objective(qg.ObjectiveSpec(denominator=4,
            reduction='explicit', gradient_scale=.5), trace)
        self.assertEqual(result.verdict, 'EXPECTED_OBJECTIVE_EFFECT')

    def test_all_masked_and_sequence_ratio(self):
        trace = qg.ObjectiveTrace(current_log_probs=[[0, 0]], old_log_probs=[[0, 0]],
            advantages=[[1, 1]], mask=[[0, 0]], observed_coefficients=[[0, 0]])
        self.assertEqual(qg.audit_objective(qg.ObjectiveSpec(), trace).verdict,
                         'INSUFFICIENT_EVIDENCE')
        self.assertEqual(qg.audit_objective(qg.ObjectiveSpec(reduction='sum'), trace).verdict,
                         'EXPECTED_OBJECTIVE_EFFECT')


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'install quotientguard[objective] for the torch oracle')
class ObjectiveTorchTests(unittest.TestCase):
    def test_all_dtypes_against_independent_backward(self):
        for dtype in ('float16', 'bfloat16', 'float32', 'float64'):
            with self.subTest(dtype=dtype):
                x = torch.tensor([[math.log(1.6), math.log(.4), 0, 0]],
                                 dtype=getattr(torch, dtype), requires_grad=True)
                adv = torch.tensor([[2., -2., 1., -1.]], dtype=x.dtype)
                ratio = x.exp()
                loss = -torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv).mean()
                loss.backward()
                trace = qg.ObjectiveTrace(x, torch.zeros_like(x), adv,
                    torch.ones_like(x), observed_coefficients=x.grad, observed_loss=loss)
                result = qg.audit_objective(qg.ObjectiveSpec(dtype=dtype), trace, oracle=True)
                self.assertEqual(result.verdict, 'EXPECTED_OBJECTIVE_EFFECT')
                self.assertTrue(result.to_dict()['oracle']['gradient_finite'])

    def test_actual_nonfinite_paths(self):
        for dtype in ('float16', 'bfloat16', 'float32', 'float64'):
            cases = [([[1000., 0]], [[0., 0]], [[0, 1]], [[1., 1.]]),
                     ([[float('-inf'), 0]], [[float('-inf'), 0]], [[0, 1]], [[1., 1.]]),
                     ([[80., 0]], [[0., 0]], [[1, 1]], [[-1e30, 1.]])]
            for current, old, mask, adv in cases:
                with self.subTest(dtype=dtype, current=current):
                    result = qg.audit_objective(qg.ObjectiveSpec(dtype=dtype),
                        qg.ObjectiveTrace(current, old, adv, mask), oracle=True)
                    # The finite-product example overflows only below float64.
                    expected = ('INSUFFICIENT_EVIDENCE' if dtype == 'float64'
                        and current[0][0] == 80 else 'NUMERICAL_ANOMALY_OBSERVED')
                    self.assertEqual(result.verdict, expected)

    def test_sequence_product_derivative_and_padding_zero(self):
        x = torch.tensor([[.1, -.1, 4.], [.1, .1, 3.]], dtype=torch.float64,
                         requires_grad=True)
        mask = torch.tensor([[1., 1., 0.], [1., 1., 0.]], dtype=x.dtype)
        adv = torch.tensor([[2., 2., 2.], [-1., -1., -1.]], dtype=x.dtype)
        ratio = (x * mask).sum(-1, keepdim=True).exp().expand_as(x)
        loss = -(torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv) * mask).sum()/4
        loss.backward()
        result = qg.audit_objective(qg.ObjectiveSpec(ratio_level='sequence_product',
            dtype='float64'), qg.ObjectiveTrace(x, torch.zeros_like(x), adv, mask,
            observed_coefficients=x.grad, observed_loss=loss), oracle=True)
        self.assertEqual(result.verdict, 'EXPECTED_OBJECTIVE_EFFECT')
        self.assertTrue(result.to_dict()['oracle']['excluded_gradients_zero'])


if __name__ == '__main__':
    unittest.main()
