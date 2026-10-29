"""Hand-derived full-loss controls; no foreign framework imports."""
import json
import math
import unittest
import importlib.util
from dataclasses import replace

from quotientguard import ObjectiveSpec, ObjectiveTrace
from quotientguard import objective as api


@unittest.skipUnless(importlib.util.find_spec('torch'), 'install quotientguard[objective]')
class FullObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(api, 'FullObjectiveSpec'), 'missing explicit full-objective contract')
        self.base = ObjectiveSpec(dtype='float64', reduction='sum', masking='select_inputs')

    def spec(self, **kwargs):
        return api.FullObjectiveSpec(policy=self.base, **kwargs)

    def trace(self, x=0., a=2., **kwargs):
        return api.FullObjectiveTrace(ObjectiveTrace([[x]], [[0.]], [[a]], [[1.]]), **kwargs)

    def audit(self, spec, trace):
        result = api.audit_full_objective(spec, trace)
        json.dumps(result.to_dict(), allow_nan=False)
        return result.to_dict()

    def test_full_kl_entropy_values_and_all_input_derivatives(self):
        # r=1, policy=-2; ref-x=log(2), k3=1-log(2).
        result = self.audit(self.spec(kl=api.KLSpec(coefficient=.3), entropy_coefficient=.2),
            self.trace(reference_log_probs=[[math.log(2)]], entropy=[[.7]]))
        self.assertEqual(result['status'], 'SUPPORTED', result)
        self.assertAlmostEqual(result['oracle']['loss'], -2 + .3*(1-math.log(2)) - .14)
        gradients = result['analytic']['gradients']
        self.assertAlmostEqual(gradients['current_log_probs']['values'][0][0], -2.3)
        self.assertEqual(gradients['old_log_probs']['state'], 'DETACHED')
        self.assertEqual(gradients['advantages']['state'], 'DETACHED')
        self.assertEqual(gradients['reference_log_probs']['state'], 'DETACHED')
        self.assertAlmostEqual(gradients['entropy']['values'][0][0], -.2)

    def test_nemo_score_weight_changes_gradient_with_same_forward(self):
        trace = self.trace(a=0., reference_log_probs=[[math.log(2)]])
        plain = self.audit(self.spec(kl=api.KLSpec(coefficient=1)), trace)
        score = self.audit(self.spec(kl=api.KLSpec(coefficient=1, score_weight=True)), trace)
        self.assertEqual(plain['oracle']['loss'], score['oracle']['loss'])
        self.assertAlmostEqual(plain['analytic']['gradients']['current_log_probs']['values'][0][0], -1)
        self.assertAlmostEqual(score['analytic']['gradients']['current_log_probs']['values'][0][0], -math.log(2))

    def test_cispo_cap_is_absolute_and_backward_does_not_saturate(self):
        result = self.audit(self.spec(surrogate='cispo', coefficient_lower=0, coefficient_upper=.2),
                            self.trace(x=math.log(2)))
        self.assertEqual(result['status'], 'SUPPORTED', result)
        self.assertAlmostEqual(result['oracle']['loss'], -.4*math.log(2))
        self.assertAlmostEqual(result['analytic']['gradients']['current_log_probs']['values'][0][0], -.4)

    def test_sapo_smooth_and_detached_score_are_distinct(self):
        trace = self.trace(x=0.)
        smooth = self.audit(self.spec(surrogate='sapo_smooth', tau_positive=2), trace)
        detached = self.audit(self.spec(surrogate='sapo_detached', tau_positive=2), trace)
        self.assertAlmostEqual(smooth['oracle']['loss'], -2.)
        self.assertAlmostEqual(detached['oracle']['loss'], 0.)
        self.assertAlmostEqual(smooth['analytic']['gradients']['current_log_probs']['values'][0][0], -2.)
        self.assertAlmostEqual(detached['analytic']['gradients']['current_log_probs']['values'][0][0], -2.)

    def test_differentiable_inputs_and_sequence_weights(self):
        policy = replace(self.base, importance_level='sequence', detach_old_policy=False,
                         detach_advantage=False, detach_importance=False)
        trace = api.FullObjectiveTrace(ObjectiveTrace([[0., 0.]], [[0., 0.]], [[2., 3.]], [[1, 1]],
                                                    importance_weights=[[.5]]))
        result = self.audit(api.FullObjectiveSpec(policy=policy, correction='weights'), trace)
        g = result['analytic']['gradients']
        self.assertEqual(result['status'], 'SUPPORTED', result)
        self.assertEqual(g['current_log_probs']['values'], [[-1., -1.5]])
        self.assertEqual(g['old_log_probs']['values'], [[1., 1.5]])
        self.assertEqual(g['advantages']['values'], [[-.5, -.5]])
        self.assertEqual(g['importance_weights']['values'], [[-5.]])

    def test_entropy_filter_does_not_mask_kl_or_sequence_ratio_derivative(self):
        trace = api.FullObjectiveTrace(ObjectiveTrace([[0., .1]], [[0., 0.]], [[1., 1.]], [[1, 1]]),
             reference_log_probs=[[0., .1]], policy_filter=[[1, 0]])
        spec = api.FullObjectiveSpec(policy=replace(self.base, ratio_level='sequence_masked_geometric_mean'),
                                     kl=api.KLSpec(estimator='k1', coefficient=.2, direction='current_minus_reference'))
        result = self.audit(spec, trace)
        self.assertEqual(result['status'], 'SUPPORTED', result)
        for value in result['oracle']['gradients']['current_log_probs']['values'][0]:
            self.assertAlmostEqual(value, -.5*math.exp(.05)+.2)

    def test_unused_detached_and_connected_zero_are_distinct(self):
        result = self.audit(self.spec(), self.trace(a=0, entropy=[[1.]]))
        g = result['oracle']['gradients']
        self.assertEqual(g['current_log_probs']['state'], 'CONNECTED')
        self.assertEqual(g['current_log_probs']['values'], [[0.]])
        self.assertEqual(g['entropy']['state'], 'UNUSED')
        self.assertIsNone(g['entropy']['values'])
        self.assertEqual(g['advantages']['state'], 'DETACHED')

    def test_select_inputs_vs_post_term_mask_finite_forward_nonfinite_backward(self):
        trace = api.FullObjectiveTrace(ObjectiveTrace([[0., 0.]], [[0., float('nan')]], [[1., 1.]], [[1, 0]]))
        safe = self.audit(self.spec(surrogate='cispo'), trace)
        literal = self.audit(api.FullObjectiveSpec(policy=replace(self.base, masking='literal'),
                            surrogate='cispo', output_masking='select'), trace)
        self.assertEqual(safe['status'], 'SUPPORTED', safe)
        self.assertEqual(literal['status'], 'NUMERICAL_ANOMALY_OBSERVED', literal)
        self.assertTrue(literal['oracle']['loss_finite'])
        self.assertFalse(literal['oracle']['gradients_finite'])
        self.assertIsNotNone(literal['oracle']['first_nonfinite_stage'])
        self.assertFalse(literal['semantic_confirmation_eligible'])

    def test_detached_gate_overflow_is_contained(self):
        for surrogate in ('cispo', 'sapo_detached'):
            result = self.audit(self.spec(surrogate=surrogate), self.trace(x=1000))
            self.assertEqual(result['status'], 'SUPPORTED', result)
            self.assertTrue(result['oracle']['loss_finite'])
            self.assertTrue(result['oracle']['gradients_finite'])
            self.assertEqual(result['oracle']['first_nonfinite_stage'], 'policy.exp_ratio')

    def test_unsupported_correction_is_not_executed_as_ppo(self):
        result = self.audit(self.spec(correction='weights', correction_placement='inside_clip'),
                            self.trace())
        self.assertEqual(result['status'], 'UNSUPPORTED')
        self.assertIsNone(result['oracle'])

    def test_boundary_is_insufficient_not_wrong_gradient(self):
        result = self.audit(self.spec(), self.trace(x=math.log(1.2)))
        self.assertEqual(result['status'], 'INSUFFICIENT_EVIDENCE')

    def test_kl_reference_gradient_direction_and_clamps(self):
        for direction, expected in [('reference_minus_current', -1.), ('current_minus_reference', -.5)]:
            result = self.audit(self.spec(kl=api.KLSpec(coefficient=1, direction=direction, detach_reference=False)),
                                self.trace(a=0, reference_log_probs=[[math.log(2)]]))
            self.assertEqual(result['status'], 'SUPPORTED', result)
            g = result['analytic']['gradients']
            self.assertAlmostEqual(g['current_log_probs']['values'][0][0], expected)
            self.assertAlmostEqual(g['reference_log_probs']['values'][0][0], -expected)
        result = self.audit(self.spec(kl=api.KLSpec(coefficient=1, delta_bounds=(-20, 20), value_bounds=(-10, 10))),
                            self.trace(a=0, reference_log_probs=[[1000]]))
        self.assertEqual(result['oracle']['loss'], 10.)
        self.assertEqual(result['oracle']['gradients']['current_log_probs']['values'], [[0.]])

    def test_all_dtypes_ratios_reductions_and_surrogates(self):
        for dtype in ('float16', 'bfloat16', 'float32', 'float64'):
            for ratio in ('token', 'sequence_product', 'sequence_masked_geometric_mean', 'sequence_token'):
                for reduction in ('sum', 'active_mean', 'explicit', 'per_sequence_active_mean', 'fixed_length_mean'):
                    for surrogate in ('ppo_clipped', 'cispo', 'sapo_smooth', 'sapo_detached'):
                        with self.subTest(dtype=dtype, ratio=ratio, reduction=reduction, surrogate=surrogate):
                            p = replace(self.base, dtype=dtype, ratio_level=ratio, reduction=reduction,
                                denominator=7, completion_length=4, empty_mask_policy='zero',
                                detach_old_policy=False, detach_advantage=False, detach_importance=False)
                            trace = api.FullObjectiveTrace(ObjectiveTrace([[.09, -.03, 1000], [.07, 0., -.06]],
                                [[0.]*3]*2, [[1., -2., 7.], [-1., .3, 2.]], [[1, 1, 0], [1, 1, 1]],
                                importance_weights=[[.7, 1.3, 1.], [1.2, .5, 1.]]),
                                reference_log_probs=[[.02]*3]*2, entropy=[[.7]*3]*2)
                            result = self.audit(api.FullObjectiveSpec(policy=p, surrogate=surrogate,
                                correction='weights', kl=api.KLSpec(coefficient=.04), entropy_coefficient=.01), trace)
                            self.assertEqual(result['status'], 'SUPPORTED', result.get('mismatches'))

    def test_finite_difference_of_live_inputs_away_from_boundaries(self):
        # A smooth, connected contract permits ordinary finite differences.
        p = replace(self.base, detach_old_policy=False, detach_advantage=False, detach_importance=False)
        spec = api.FullObjectiveSpec(policy=p, correction='weights',
                                    kl=api.KLSpec(coefficient=.3, detach_reference=False), entropy_coefficient=.02)
        values = {'current_log_probs': .07, 'old_log_probs': -.02, 'advantages': -1.4,
                  'importance_weights': .8, 'reference_log_probs': -.06, 'entropy': .9}
        def run(vals):
            tr = api.FullObjectiveTrace(ObjectiveTrace([[vals['current_log_probs']]], [[vals['old_log_probs']]],
                [[vals['advantages']]], [[1]], importance_weights=[[vals['importance_weights']]]),
                reference_log_probs=[[vals['reference_log_probs']]], entropy=[[vals['entropy']]])
            return self.audit(spec, tr)
        result = run(values)
        for name, value in values.items():
            plus, minus = dict(values), dict(values)
            plus[name], minus[name] = value+1e-6, value-1e-6
            difference = (run(plus)['oracle']['loss']-run(minus)['oracle']['loss'])/2e-6
            self.assertAlmostEqual(difference, result['analytic']['gradients'][name]['values'][0][0], places=8)

    def test_empty_rows_and_term_masks_and_explicit_window_scaling(self):
        trace = api.FullObjectiveTrace(ObjectiveTrace([[0., 0.], [0., 0.]], [[0., 0.]]*2,
            [[1., 1.]]*2, [[1, 0], [0, 0]]), entropy=[[2., 2.]]*2,
            entropy_mask=[[0, 1], [0, 0]])
        spec = self.spec(entropy_coefficient=.1, policy_reduction=api.ReductionSpec(
            reduction='explicit', denominator=8, denominator_scope='accumulation_window', process_count=2, scale=3),
            entropy_reduction=api.ReductionSpec(reduction='per_sequence_active_mean', empty_mask_policy='zero'))
        result = self.audit(spec, trace)
        self.assertEqual(result['status'], 'SUPPORTED', result)
        self.assertAlmostEqual(result['oracle']['loss'], -.85)
        self.assertEqual(result['oracle']['gradients']['current_log_probs']['values'], [[-.75, 0.], [0., 0.]])

    def test_clip_rejection_zero_advantage_extremes_and_backward_only(self):
        for x, a, expected in [(math.log(2), 1, 0.), (math.log(.5), -1, 0.), (.1, 0, 0.)]:
            result = self.audit(self.spec(), self.trace(x=x, a=a))
            self.assertEqual(result['status'], 'SUPPORTED', result)
            self.assertEqual(result['oracle']['gradients']['current_log_probs']['values'], [[expected]])
        result = self.audit(self.spec(surrogate='sapo_smooth'), self.trace(x=1000))
        self.assertEqual(result['status'], 'NUMERICAL_ANOMALY_OBSERVED')
        self.assertTrue(result['oracle']['loss_finite'])
        self.assertFalse(result['oracle']['gradients_finite'])
        for value in (float('inf'), -float('inf'), float('nan')):
            trace = api.FullObjectiveTrace(ObjectiveTrace([[0, value]], [[0, value]], [[1, 1]], [[1, 0]]))
            result = self.audit(self.spec(), trace)
            self.assertEqual(result['status'], 'SUPPORTED', result)

    def test_observed_nonfinite_is_not_a_finite_mismatch(self):
        result = self.audit(self.spec(), self.trace(observed_loss=float('nan')))
        self.assertEqual(result['status'], 'NUMERICAL_ANOMALY_OBSERVED')

    def test_native_callable_compares_contract_without_promoting_evidence(self):
        import torch
        from quotientguard.native_validation import validate_callable
        def loss(inputs):
            return -inputs['current_log_probs'].exp().sum() + .2*inputs['entropy'].sum()
        inputs = {'current_log_probs': torch.zeros((1, 1), dtype=torch.float64),
                  'old_log_probs': torch.zeros((1, 1), dtype=torch.float64),
                  'advantages': torch.ones((1, 1), dtype=torch.float64),
                  'mask': torch.ones((1, 1)), 'entropy': torch.ones((1, 1), dtype=torch.float64)}
        report = validate_callable(loss, inputs, framework='control', path_id='wrong_entropy_sign',
                                   objective_contract=self.spec(entropy_coefficient=.2))
        audit = report['runs'][0]['full_objective']
        self.assertEqual(audit['status'], 'CONTRACT_MISMATCH_OBSERVED')
        self.assertIn('observed.gradient.entropy', audit['mismatches'])
        self.assertFalse(report['semantic_confirmation_eligible'])
        self.assertFalse(report['trainer_executed'])

    def test_nemo_score_path_is_detached_when_input_clamp_applies(self):
        result = self.audit(self.spec(kl=api.KLSpec(coefficient=1, score_weight=True,
            delta_bounds=(-2, 2), score_detach_on_delta_clamp=True)),
            self.trace(a=0, reference_log_probs=[[3.]]))
        self.assertEqual(result['status'], 'SUPPORTED', result)
        self.assertEqual(result['oracle']['gradients']['current_log_probs']['values'], [[0.]])

    def test_post_subtraction_mask_preserves_live_cispo_path(self):
        trace = api.FullObjectiveTrace(ObjectiveTrace([[0., float('nan')]], [[0., 0.]], [[1, 1]], [[1, 0]]))
        result = self.audit(api.FullObjectiveSpec(policy=replace(self.base, masking='literal'),
                           ratio_masking='select_delta', surrogate='cispo', output_masking='select'), trace)
        # The ratio is sanitized, but the live x is NaN: forward selection erases
        # that value, and CISPO's detached finite coefficient leaves zero dL/dx.
        self.assertTrue(result['oracle']['loss_finite'])
        self.assertTrue(result['oracle']['gradients_finite'])
        self.assertEqual(result['status'], 'SUPPORTED', result)

    def test_live_zero_advantage_nondifferentiable_branch_is_insufficient(self):
        p = replace(self.base, detach_advantage=False)
        for surrogate in ('ppo_clipped', 'sapo_smooth', 'sapo_detached'):
            result = self.audit(api.FullObjectiveSpec(policy=p, surrogate=surrogate), self.trace(x=math.log(2), a=0))
            self.assertEqual(result['status'], 'INSUFFICIENT_EVIDENCE', result)

    def test_native_precision_mismatch_is_unsupported(self):
        import torch
        from quotientguard.native_validation import validate_callable
        inputs = {name: torch.tensor([[value]], dtype=torch.float64) for name, value in
                  [('current_log_probs', 0.), ('old_log_probs', 0.), ('advantages', 1.), ('mask', 1.)]}
        report = validate_callable(lambda i: -(i['current_log_probs']-i['old_log_probs'].detach()).exp().sum(),
            inputs, framework='control', path_id='precision_mismatch',
            objective_contract=api.FullObjectiveSpec(policy=replace(self.base, dtype='float16')))
        self.assertEqual(report['runs'][0]['full_objective']['status'], 'UNSUPPORTED')

    def test_cli_full_objective_json_and_unsupported_exit(self):
        import contextlib
        import io
        from pathlib import Path
        import tempfile
        from quotientguard.cli import main
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'input.json'
            payload = {'spec': {'policy': {'dtype': 'float64', 'reduction': 'sum'},
                                'kl': {'coefficient': .1, 'reduction': {'reduction': 'sum'}}},
                       'trace': {'policy': {'current_log_probs': [[0]], 'old_log_probs': [[0]],
                                           'advantages': [[1]], 'mask': [[1]]}, 'reference_log_probs': [[0]]}}
            path.write_text(json.dumps(payload))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(['full-objective-audit', str(path), '--json'])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())['status'], 'SUPPORTED')
            payload['spec']['correction_placement'] = 'inside_clip'
            path.write_text(json.dumps(payload))
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(['full-objective-audit', str(path), '--json'])
            self.assertEqual(code, 2)

    def test_differentiable_kl_importance_uses_rollout_coordinate(self):
        spec = self.spec(kl=api.KLSpec(coefficient=1, importance_weight='current_rollout', detach_rollout=False))
        result = self.audit(spec, self.trace(a=0, reference_log_probs=[[math.log(2)]], rollout_log_probs=[[-math.log(3)]]))
        self.assertEqual(result['status'], 'SUPPORTED', result)
        g = result['analytic']['gradients']
        self.assertAlmostEqual(result['oracle']['loss'], 3*(1-math.log(2)))
        self.assertAlmostEqual(g['current_log_probs']['values'][0][0], -3*math.log(2))
        self.assertAlmostEqual(g['rollout_log_probs']['values'][0][0], -3*(1-math.log(2)))

    def test_kl_unfiltered_current_coordinate_is_separate_from_policy(self):
        result = self.audit(self.spec(kl=api.KLSpec(coefficient=1)),
            self.trace(a=1, kl_current_log_probs=[[-math.log(2)]], reference_log_probs=[[0.]]))
        self.assertEqual(result['status'], 'SUPPORTED', result)
        g = result['oracle']['gradients']
        self.assertAlmostEqual(g['current_log_probs']['values'][0][0], -1.)
        self.assertAlmostEqual(g['kl_current_log_probs']['values'][0][0], -1.)

    def test_kl_importance_clamp_and_detach_boundaries(self):
        spec = self.spec(kl=api.KLSpec(coefficient=1, importance_weight='current_rollout', detach_rollout=False,
                                     delta_bounds=(-2,2), score_detach_on_delta_clamp=True))
        result = self.audit(spec, self.trace(a=0, reference_log_probs=[[3]], rollout_log_probs=[[-math.log(3)]]))
        self.assertEqual(result['status'], 'SUPPORTED', result)
        self.assertEqual(result['oracle']['gradients']['current_log_probs']['values'], [[0.]])
        self.assertEqual(result['oracle']['gradients']['rollout_log_probs']['values'], [[0.]])

    def test_nonzero_observed_excluded_gradient_cannot_use_tolerance(self):
        trace = api.FullObjectiveTrace(ObjectiveTrace([[0.,0.]], [[0.,0.]], [[1.,1.]], [[1,0]]),
            observed_gradients={'current_log_probs':[[-1.,1e-12]]})
        result = self.audit(self.spec(),trace)
        self.assertEqual(result['status'],'CONTRACT_MISMATCH_OBSERVED')

    def test_nested_policy_observations_are_not_ignored(self):
        trace = api.FullObjectiveTrace(ObjectiveTrace([[0]],[[0]],[[1]],[[1]],
            observed_loss=100.,observed_coefficients=[[100.]]))
        result = self.audit(self.spec(),trace)
        self.assertEqual(result['status'],'CONTRACT_MISMATCH_OBSERVED')
        self.assertIn('observed.policy.loss',result['mismatches'])
        self.assertIn('observed.policy.gradient.current_log_probs',result['mismatches'])


if __name__ == '__main__':
    unittest.main()
