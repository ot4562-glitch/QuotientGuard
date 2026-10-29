"""Execute real backward/unscale/clip/SGD on tiny controlled models."""
import json
import unittest
import importlib.util

from quotientguard import native_validation as api


@unittest.skipUnless(importlib.util.find_spec('torch'), 'install quotientguard[objective]')
class TrainingValidationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(api, 'validate_training_step'), 'missing training scaling contract')
        import torch
        self.torch = torch

    def test_accumulation_unscale_clip_and_sgd_update(self):
        torch = self.torch
        p = torch.nn.Parameter(torch.tensor([1., -1.]))
        optimizer = torch.optim.SGD([p], lr=.1)
        scaler = torch.amp.GradScaler('cpu', init_scale=16)
        report = api.validate_training_step(lambda batch: (p*batch).sum(), {'p': p},
            [torch.tensor([2., 0.]), torch.tensor([0., -4.])], optimizer=optimizer,
            accumulation_weights=[.5, .5], scaler=scaler, max_grad_norm=1.)
        self.assertEqual(report['status'], 'PASS', report)
        self.assertEqual(report['unscaled_gradients']['p'], [1., -2.])
        self.assertAlmostEqual(p[0].item(), 1-.1/(5**.5+1e-6), places=6)
        self.assertAlmostEqual(p[1].item(), -1+.2/(5**.5+1e-6), places=6)
        self.assertTrue(report['optimizer_step_executed'])
        self.assertTrue(report['grad_scaler_enabled'])
        self.assertFalse(report['trainer_executed'])
        json.dumps(report, allow_nan=False)

    def test_zero_learning_rate_is_normal_no_update(self):
        torch = self.torch
        p = torch.nn.Parameter(torch.tensor([1.]))
        report = api.validate_training_step(lambda _: p.sum(), {'p': p}, [None],
                                           optimizer=torch.optim.SGD([p], lr=0))
        self.assertEqual(report['status'], 'PASS', report)
        self.assertFalse(report['parameters_changed'])
        self.assertIn('zero_learning_rate', report['normal_no_update_reasons'])

    def test_nonfinite_scaled_gradient_skips_real_optimizer_step(self):
        torch = self.torch
        p = torch.nn.Parameter(torch.tensor([1.]))
        report = api.validate_training_step(lambda _: p.sum()*float('inf'), {'p': p}, [None],
            optimizer=torch.optim.SGD([p], lr=.1), scaler=torch.amp.GradScaler('cpu', init_scale=8))
        self.assertEqual(report['status'], 'NUMERICAL_ANOMALY_OBSERVED')
        self.assertFalse(report['parameters_changed'])
        self.assertTrue(report['optimizer_step_skipped'])
        self.assertEqual(p.item(), 1.)

    def test_cpu_autocast_location_and_unused_parameter(self):
        torch = self.torch
        model = torch.nn.Linear(2, 1, bias=False)
        spare = torch.nn.Parameter(torch.ones(1))
        params = {'weight': model.weight, 'spare': spare}
        report = api.validate_training_step(lambda batch: model(batch).float().square().mean(), params,
            [torch.ones(2, 2)], optimizer=torch.optim.SGD(list(params.values()), lr=.01),
            autocast_dtype='bfloat16', device_type='cpu')
        self.assertEqual(report['status'], 'PASS', report)
        self.assertTrue(report['autocast_enabled'])
        self.assertIsNone(report['unscaled_gradients']['spare'])
        self.assertEqual(report['parameter_states']['spare'], 'UNUSED')

    def test_unsupported_optimizer_does_not_execute(self):
        torch = self.torch
        p = torch.nn.Parameter(torch.tensor([1.]))
        calls = []
        report = api.validate_training_step(lambda _: calls.append(1), {'p': p}, [None],
                                           optimizer=torch.optim.Adam([p]))
        self.assertEqual(report['status'], 'UNSUPPORTED')
        self.assertEqual(calls, [])
        self.assertEqual(p.item(), 1.)

    def test_large_finite_gradient_without_clipping_does_not_overflow_norm(self):
        torch = self.torch
        p = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        result = api.validate_training_step(lambda _: p.sum()*1e200, {'p': p}, [None],
                                           optimizer=torch.optim.SGD([p], lr=.1))
        self.assertEqual(result['status'], 'PASS', result)
        self.assertAlmostEqual(p.item()/-1e199, 1.)

    def test_grad_scaler_with_all_optimizer_parameters_unused_is_normal(self):
        torch = self.torch
        p = torch.nn.Parameter(torch.ones(1))
        q = torch.ones(1, requires_grad=True)
        result = api.validate_training_step(lambda _: q.sum(), {'p': p}, [None],
            optimizer=torch.optim.SGD([p], lr=.1), scaler=torch.amp.GradScaler('cpu'))
        self.assertEqual(result['status'], 'PASS', result)
        self.assertTrue(result['optimizer_step_skipped'])
        self.assertFalse(result['parameters_changed'])

    def test_finite_gradient_can_overflow_parameter_update(self):
        torch = self.torch
        p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        result = api.validate_training_step(lambda _: p.sum()*1e308, {'p': p}, [None],
                                           optimizer=torch.optim.SGD([p], lr=10))
        self.assertEqual(result['status'], 'NUMERICAL_ANOMALY_OBSERVED')
        self.assertFalse(result['parameters_finite'])

    def test_finite_gradient_clip_norm_overflow_has_numerical_stage(self):
        torch=self.torch
        p=torch.nn.Parameter(torch.zeros(2))
        result=api.validate_training_step(lambda _:p.sum()*1e20,{'p':p},[None],
            optimizer=torch.optim.SGD([p],lr=.1),max_grad_norm=1.)
        self.assertEqual(result['status'],'NUMERICAL_ANOMALY_OBSERVED')
        self.assertEqual(result['numerical_failure_stage'],'GRADIENT_CLIP_NORM')
