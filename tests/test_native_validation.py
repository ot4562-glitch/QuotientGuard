"""Controlled execution evidence cannot establish real training reachability."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


def selected_loss(inputs):
    x = inputs['current_log_probs'].masked_fill(inputs['mask'] == 0, 0)
    return -(x.exp() * inputs['mask']).sum()


def unsafe_loss(inputs):
    return -(inputs['current_log_probs'].exp() * inputs['mask']).sum()


@unittest.skipIf(torch is None, 'install quotientguard[objective]')
class NativeValidationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('quotientguard.native_validation'),
                             'controlled callable validation API is missing')
        from quotientguard import native_validation
        self.api = native_validation

    def inputs(self):
        return {'current_log_probs': torch.zeros((1, 2), dtype=torch.float64),
                'old_log_probs': torch.zeros((1, 2), dtype=torch.float64),
                'mask': torch.tensor([[0., 1.]], dtype=torch.float64)}

    def run_loss(self, loss=selected_loss, **kwargs):
        return self.api.validate_callable(loss, self.inputs(),
            framework='synthetic-test', path_id='mask-control', **kwargs)

    def test_mask_mutations_capture_backward_and_preserve_inputs(self):
        inputs = self.inputs()
        report = self.api.validate_callable(selected_loss, inputs, framework='synthetic-test',
            path_id='mask-control', mutations={'overflow': {'current_log_probs': 1000.},
                                            'nan': {'current_log_probs': float('nan')}})
        self.assertEqual(report['evidence_grade'], 'INDEPENDENT_TORCH_CONTROLLED')
        for run in report['runs']:
            self.assertEqual(run['loss'], -1.)
            self.assertEqual(run['current_gradient'], [[0., -1.]])
            self.assertTrue(run['excluded_gradients_exactly_zero'])
            self.assertTrue(run['active_inputs_unchanged'])
            self.assertTrue(run['mask_invariance_passed'])
        self.assertFalse(inputs['current_log_probs'].requires_grad)
        self.assertTrue(torch.equal(inputs['current_log_probs'], torch.zeros_like(inputs['mask'])))
        self.assertFalse(report['semantic_confirmation_eligible'])

    def test_nonfinite_masked_backward_is_retained_in_strict_json(self):
        report = self.run_loss(unsafe_loss,
            mutations={'overflow': {'current_log_probs': 1000.}})
        run = report['runs'][1]
        self.assertEqual(run['loss'], 'NaN')
        self.assertEqual(run['current_gradient'][0][0], 'NaN')
        self.assertFalse(run['excluded_gradients_exactly_zero'])
        self.assertFalse(run['mask_invariance_passed'])
        self.assertEqual(run['numerical_status'], 'NONFINITE_OBSERVED')
        json.dumps(report, allow_nan=False)

    def test_excluded_zero_is_exact_even_for_tiny_finite_gradient(self):
        def small_leak(inputs):
            return selected_loss(inputs) + inputs['current_log_probs'][0, 0] * 1e-30
        report = self.run_loss(small_leak)
        self.assertFalse(report['runs'][0]['excluded_gradients_exactly_zero'])

    def test_native_grade_requires_pinned_executed_source(self):
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report = self.run_loss(execution_kind='NATIVE_IMPORTED', source_sha256=digest,
                               source_commit='controlled-test-commit')
        self.assertEqual(report['evidence_grade'], 'NATIVE_CONTROLLED_REPRODUCED')
        self.assertEqual(report['execution_scope'], 'LOSS_CALLABLE')
        self.assertFalse(report['trainer_executed'])
        self.assertTrue(report['source']['hash_verified'])
        self.assertTrue(report['runs'][0]['callable_executed'])
        self.assertFalse(report['semantic_confirmation_eligible'])
        report = self.run_loss(execution_kind='NATIVE_IMPORTED', source_sha256='0' * 64,
                               source_commit='controlled-test-commit')
        self.assertEqual(report['evidence_grade'], 'UNVERIFIED_CALLABLE_CONTROLLED')

    def test_adapter_cannot_claim_execution_when_callable_was_not_called(self):
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report = self.run_loss(execution_kind='NATIVE_IMPORTED', source_sha256=digest,
            source_commit='test-commit', invoke=lambda fn, inputs: -inputs['current_log_probs'].sum())
        self.assertFalse(report['runs'][0]['callable_executed'])
        self.assertEqual(report['evidence_grade'], 'UNVERIFIED_CALLABLE_CONTROLLED')

    def test_discarded_callable_result_cannot_certify_adapter_gradient(self):
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        def discard(fn, inputs):
            fn(inputs)
            return -inputs['current_log_probs'].sum()
        report = self.run_loss(execution_kind='NATIVE_IMPORTED', source_sha256=digest,
            source_commit='test-commit', invoke=discard)
        self.assertTrue(report['runs'][0]['callable_executed'])
        self.assertEqual(report['evidence_grade'], 'UNVERIFIED_CALLABLE_CONTROLLED')
        self.assertFalse(report['runs'][0]['loss_derived_from_callable_return'])

    def test_adapter_reduction_preserves_callable_graph_origin(self):
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report = self.run_loss(execution_kind='NATIVE_IMPORTED', source_sha256=digest,
            source_commit='test-commit', invoke=lambda fn, inputs: fn(inputs).sum() / 2)
        self.assertTrue(report['runs'][0]['loss_derived_from_callable_return'])
        self.assertEqual(report['evidence_grade'], 'NATIVE_CONTROLLED_REPRODUCED')

    def test_finite_forward_nonfinite_backward_has_distinct_failure_stage(self):
        def finite_forward(inputs):
            x = inputs['current_log_probs']
            # Mirrors post-exp rejection: forward selects zero but exp backward
            # still multiplies its nonfinite output by the zero upstream gradient.
            return torch.where(inputs['mask'].bool(), x.exp(), 0.).sum()
        run = self.run_loss(finite_forward,
            mutations={'overflow': {'current_log_probs': 1000.}})['runs'][1]
        self.assertTrue(run['finite_loss'])
        self.assertFalse(run['finite_current_gradient'])
        self.assertEqual(run['numerical_failure_stage'], 'BACKWARD_ONLY')

    def test_wrapper_chain_records_body_without_promoting_unverified_identity(self):
        import functools
        @functools.wraps(selected_loss)
        def wrapper(inputs):
            return selected_loss(inputs)
        report = self.run_loss(wrapper)
        self.assertEqual(len(report['source']['wrapper_chain']), 2)
        self.assertTrue(all(item['executed'] for item in report['runs'][0]['callable_chain']))
        self.assertTrue(report['runs'][0]['loss_derived_from_callable_return'])

    def test_dynamic_lambda_cannot_gain_native_grade_from_surrounding_source_file(self):
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report = self.run_loss(lambda inputs: selected_loss(inputs),
            execution_kind='NATIVE_IMPORTED', source_sha256=digest, source_commit='test-commit')
        self.assertEqual(report['evidence_grade'], 'UNVERIFIED_CALLABLE_CONTROLLED')

    def test_ast_and_source_equivalent_runs_keep_lower_evidence_grade(self):
        for kind, grade in [('AST_EXTRACTED', 'AST_EXTRACTED_CONTROLLED'),
                            ('SOURCE_EQUIVALENT', 'SOURCE_EQUIVALENT_CONTROLLED')]:
            with self.subTest(kind=kind):
                self.assertEqual(self.run_loss(execution_kind=kind)['evidence_grade'], grade)

    def test_raw_and_sanitized_observations_are_both_retained(self):
        def invoke(fn, inputs):
            loss = fn(inputs)
            return {'loss': loss, 'raw_tensors': {'weight': torch.tensor([float('inf')])},
                    'sanitized_tensors': {'weight': torch.tensor([0.])}}
        run = self.run_loss(invoke=invoke)['runs'][0]
        self.assertEqual(run['raw_tensors']['weight'], ['Infinity'])
        self.assertEqual(run['sanitized_tensors']['weight'], [0.])

    def test_detached_loss_does_not_invent_zero_gradients(self):
        def detached(inputs):
            return torch.tensor(0.)
        run = self.run_loss(detached)['runs'][0]
        self.assertIsNone(run['current_gradient'])
        self.assertIsNone(run['excluded_gradients_exactly_zero'])
        self.assertFalse(run['backward_executed'])

    def test_auxiliary_backward_nonfinite_is_visible_with_finite_policy_gradient(self):
        inputs = self.inputs()
        inputs['auxiliary'] = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        inputs['auxiliary'].grad = torch.ones(2, dtype=torch.float64)
        def full_loss(i):
            return selected_loss(i) + i['auxiliary'].sqrt().sum()
        report = self.api.validate_callable(full_loss, inputs, framework='test', path_id='full-loss',
                                            gradient_inputs=('auxiliary',))
        run = report['runs'][0]
        self.assertTrue(run['finite_loss'])
        self.assertTrue(run['finite_current_gradient'])
        self.assertFalse(run['finite_monitored_gradients'])
        self.assertEqual(run['numerical_failure_stage'], 'BACKWARD_ONLY')
        self.assertEqual(run['numerical_status'], 'NONFINITE_OBSERVED')
        self.assertEqual(run['input_gradients']['auxiliary']['gradient'], ['Infinity', 'Infinity'])
        self.assertFalse(run['mask_invariance_passed'])
        self.assertTrue(run['input_gradients']['auxiliary']['graph_connected'])
        self.assertTrue(torch.equal(inputs['auxiliary'].grad, torch.ones(2, dtype=torch.float64)))

    def test_declared_differentiable_correction_is_audited_without_mutating_inputs(self):
        inputs = self.inputs()
        inputs['correction'] = torch.full_like(inputs['current_log_probs'], 2.)
        def corrected_loss(i):
            return -(i['current_log_probs'].exp() * i['correction'] * i['mask']).sum()
        report = self.api.validate_callable(corrected_loss, inputs, framework='test', path_id='correction',
            gradient_inputs=('correction',), mutations={'excluded': {'correction': float('nan')}})
        normal, mutation = report['runs']
        self.assertEqual(normal['input_gradients']['correction']['gradient'], [[0., -1.]])
        self.assertTrue(normal['finite_monitored_gradients'])
        self.assertFalse(inputs['correction'].requires_grad)
        self.assertIsNone(inputs['correction'].grad)
        self.assertEqual(mutation['numerical_failure_stage'], 'FORWARD_AND_BACKWARD')

    def test_disconnected_monitored_input_is_missing_not_an_invented_zero(self):
        inputs = self.inputs()
        inputs['unused'] = torch.ones(1, dtype=torch.float64)
        run = self.api.validate_callable(selected_loss, inputs, framework='test', path_id='unused',
                                         gradient_inputs=('unused',))['runs'][0]
        self.assertIsNone(run['input_gradients']['unused']['gradient'])
        self.assertFalse(run['input_gradients']['unused']['graph_connected'])
        self.assertIsNone(run['finite_monitored_gradients'])
        self.assertEqual(run['numerical_status'], 'FINITE_OBSERVED')

    def test_gradient_input_names_require_distinct_floating_auxiliary_tensors(self):
        for names in ('missing', ('missing',), ('mask',), ('current_log_probs',), ('old_log_probs', 'old_log_probs')):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.run_loss(gradient_inputs=names)

    def test_invalid_masks_and_mutation_targets_are_rejected(self):
        inputs = self.inputs()
        inputs['mask'][0, 0] = .5
        with self.assertRaises(ValueError):
            self.api.validate_callable(selected_loss, inputs, framework='test', path_id='test')
        with self.assertRaises(ValueError):
            self.run_loss(mutations={'bad': {'mask': 0}})

    def test_missing_path_and_report_hash_are_explicit(self):
        from quotientguard.provenance import sha256_json
        report = self.api.not_executed_report(framework='missing-framework',
            path_id='Trainer.loss', reason='optional framework dependencies unavailable')
        self.assertEqual(report['evidence_grade'], 'NOT_EXECUTED')
        self.assertEqual(report['paths'][0]['status'], 'NOT_EXECUTED')
        self.assertIn('dependencies', report['paths'][0]['reason'])
        self.assertEqual(report['report_sha256'], sha256_json(
            {key: value for key, value in report.items() if key != 'report_sha256'}))

    def test_report_writer_rejects_tampering_and_round_trips_strict_json(self):
        report = self.run_loss(unsafe_loss, mutations={'overflow': {'current_log_probs': 1000.}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            self.api.write_native_report(report, path)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), report)
            report['runs'][0]['loss'] = 999.
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                self.api.write_native_report(report, path)

    def test_changed_source_file_cannot_pin_stale_loaded_callable(self):
        import importlib.util
        import sys
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native_stale_control.py'
            path.write_text("def loss(inputs):\n    return inputs['current_log_probs'].sum()\n", encoding='utf-8')
            spec = importlib.util.spec_from_file_location('native_stale_control', path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            self.addCleanup(sys.modules.pop, spec.name, None)
            spec.loader.exec_module(module)
            path.write_text("def loss(inputs):\n    return -inputs['current_log_probs'].sum()\n", encoding='utf-8')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            report = self.run_loss(module.loss, execution_kind='NATIVE_IMPORTED', source_sha256=digest,
                                   source_commit='a' * 40)
            self.assertEqual(report['evidence_grade'], 'UNVERIFIED_CALLABLE_CONTROLLED')


if __name__ == '__main__':
    unittest.main()
