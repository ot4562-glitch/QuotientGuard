"""Independent negative-surrogate controls for explicit development contracts."""
import math
import unittest
from dataclasses import replace

from quotientguard import ObjectiveSpec, ObjectiveTrace, audit_objective

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'install quotientguard[objective]')
class ObjectiveV12Tests(unittest.TestCase):
    def control(self, spec, *, empty=False, sentinel=4.):
        dtype = getattr(torch, spec.dtype)
        x = torch.tensor([[.1, -.03, sentinel], [.1, .04, -.02]], dtype=dtype,
                         requires_grad=True)
        old = torch.zeros_like(x)
        mask = torch.tensor([[1, 1, 0], [0, 0, 0] if empty else [1, 1, 1]], dtype=dtype)
        adv = torch.tensor([[1, -2, 3], [-1, 0, 2]], dtype=dtype)
        weights = torch.tensor([[.7, 1.3, 2], [1, .5, 1.5]], dtype=dtype)
        # Independent graph uses max of negative terms rather than oracle min.
        selected_x = torch.where(mask.bool(), x, 0.) if spec.masking == 'select_inputs' else x
        selected_o = torch.where(mask.bool(), old, 0.) if spec.masking == 'select_inputs' else old
        delta = selected_x - selected_o
        if spec.ratio_level == 'token':
            log_ratio = delta
        else:
            total = (delta * mask).sum(-1, keepdim=True)
            if spec.ratio_level != 'sequence_product':
                total = total / mask.sum(-1, keepdim=True).clamp_min(1)
            log_ratio = (selected_x - selected_x.detach() + total.detach()
                         if spec.ratio_level == 'sequence_token' else total)
        ratio = log_ratio.exp()
        loss_terms = torch.maximum(-ratio * adv, -ratio.clamp(.8, 1.2) * adv) * weights * mask
        if spec.reduction == 'per_sequence_active_mean':
            loss = (loss_terms.sum(-1) / mask.sum(-1).clamp_min(1)).mean()
        else:
            divisor = {'sum': 1., 'active_mean': mask.sum().item(),
                       'fixed_length_mean': x.shape[0] * (spec.completion_length or 1),
                       'explicit': spec.denominator or 1.}[spec.reduction]
            if divisor == 0 and spec.empty_mask_policy == 'zero':
                divisor = 1.
            loss = loss_terms.sum() / divisor
        gradient, = torch.autograd.grad(loss, x)
        trace = ObjectiveTrace(x, old, adv, mask, importance_weights=weights,
                               observed_coefficients=gradient, observed_loss=loss)
        return audit_objective(spec, trace, oracle=True)

    def test_independent_gradients_all_ratios_reductions_and_cpu_dtypes(self):
        for dtype in ('float16', 'bfloat16', 'float32', 'float64'):
            for ratio in ('token', 'sequence_product', 'sequence_masked_geometric_mean', 'sequence_token'):
                for reduction in ('sum', 'active_mean', 'per_sequence_active_mean', 'fixed_length_mean', 'explicit'):
                    with self.subTest(dtype=dtype, ratio=ratio, reduction=reduction):
                        result = self.control(ObjectiveSpec(dtype=dtype, ratio_level=ratio,
                            reduction=reduction, denominator=7., completion_length=8,
                            masking='select_inputs'))
                        self.assertEqual(result.verdict, 'EXPECTED_OBJECTIVE_EFFECT', result.to_dict())
                        self.assertTrue(result.evidence['oracle']['excluded_gradients_zero'])

    def test_forward_mean_and_product_are_distinct(self):
        trace = ObjectiveTrace([[.1, .1, 9]], [[0, 0, 0]], [[1, 1, 1]], [[1, 1, 0]])
        for mode, value in [('sequence_product', math.exp(.2)),
                            ('sequence_masked_geometric_mean', math.exp(.1))]:
            result = audit_objective(ObjectiveSpec(dtype='float64', ratio_level=mode), trace, oracle=True)
            self.assertAlmostEqual(result.evidence['importance_ratios'][0][0], value)

    def test_stop_gradient_has_local_not_coupled_derivative(self):
        base = ObjectiveSpec(dtype='float64', reduction='sum', masking='select_inputs')
        coupled = self.control(replace(base, ratio_level='sequence_masked_geometric_mean'))
        local = self.control(replace(base, ratio_level='sequence_token'))
        self.assertEqual(coupled.evidence['oracle']['loss'], local.evidence['oracle']['loss'])
        self.assertNotEqual(coupled.evidence['expected_coefficients'], local.evidence['expected_coefficients'])

    def test_select_inputs_ignores_excluded_nonfinite_before_subtraction(self):
        for sentinel in (1000., float('inf'), -float('inf'), float('nan')):
            trace = ObjectiveTrace([[.1, sentinel]], [[0, sentinel]], [[1, 1]], [[1, 0]],
                observed_coefficients=[[-math.exp(.1), 0]])
            for ratio in ('token', 'sequence_masked_geometric_mean', 'sequence_token'):
                result = audit_objective(ObjectiveSpec(dtype='float64', ratio_level=ratio,
                    masking='select_inputs', reduction='sum'), trace, oracle=True)
                self.assertEqual(result.verdict, 'EXPECTED_OBJECTIVE_EFFECT', result.to_dict())
                self.assertTrue(result.evidence['oracle']['excluded_gradients_zero'])

    def test_empty_sequence_policy_is_explicit(self):
        base = ObjectiveSpec(dtype='float64', ratio_level='sequence_masked_geometric_mean',
            reduction='per_sequence_active_mean', masking='select_inputs')
        undefined = self.control(base, empty=True)
        # Executing the undefined 0/0 mean is an observed replay anomaly.
        self.assertEqual(undefined.verdict, 'NUMERICAL_ANOMALY_OBSERVED')
        self.assertIn('undefined_normalization_denominator', undefined.evidence['reasons'])
        zero = self.control(replace(base, empty_mask_policy='zero'), empty=True)
        self.assertEqual(zero.verdict, 'EXPECTED_OBJECTIVE_EFFECT')
        trace = ObjectiveTrace([[float('nan')]], [[float('-inf')]], [[1]], [[0]],
            observed_coefficients=[[0]], observed_loss=0.)
        result = audit_objective(replace(base, reduction='active_mean', empty_mask_policy='zero'), trace, oracle=True)
        self.assertEqual(result.verdict, 'EXPECTED_OBJECTIVE_EFFECT')
        self.assertEqual(result.evidence['oracle']['loss'], 0.)

    def test_new_ratio_clip_boundary_is_not_confirmed(self):
        trace = ObjectiveTrace([[math.log(1.2)] * 2], [[0, 0]], [[1, 1]], [[1, 1]],
                               observed_coefficients=[[0, 0]])
        for ratio in ('sequence_masked_geometric_mean', 'sequence_token'):
            result = audit_objective(ObjectiveSpec(dtype='float64', ratio_level=ratio), trace, oracle=True)
            self.assertEqual(result.verdict, 'INSUFFICIENT_EVIDENCE')
            self.assertIn('clip_boundary_subgradient_ambiguous', result.evidence['reasons'])

    def test_differentiable_correction_stays_unsupported(self):
        result = audit_objective(ObjectiveSpec(ratio_level='sequence_token', detach_importance=False),
            ObjectiveTrace([[0]], [[0]], [[1]], [[1]], observed_coefficients=[[-1]]), oracle=True)
        self.assertEqual(result.verdict, 'INSUFFICIENT_EVIDENCE')

    def test_per_sequence_half_reduction_does_not_cast_combined_divisor_to_infinity(self):
        x = torch.zeros((256, 256), dtype=torch.float16, requires_grad=True)
        mask = torch.ones_like(x)
        loss = (-x.exp().sum(-1) / mask.sum(-1)).mean()
        gradient, = torch.autograd.grad(loss, x)
        result = audit_objective(ObjectiveSpec(dtype='float16', reduction='per_sequence_active_mean'),
            ObjectiveTrace(x, torch.zeros_like(x), mask, mask,
                           observed_loss=loss, observed_coefficients=gradient), oracle=True)
        self.assertEqual(result.verdict, 'EXPECTED_OBJECTIVE_EFFECT', result.evidence['reasons'])
        self.assertEqual(result.evidence['oracle']['loss'], -1.)


if __name__ == '__main__':
    unittest.main()
