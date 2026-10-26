import math
import unittest

from quotientguard import ObjectiveSpec, ObjectiveTrace, audit_objective


class ObjectiveBoundaryTests(unittest.TestCase):
    def test_sequence_importance_defaults_to_unit_weights(self):
        trace = ObjectiveTrace([[0, 0]], [[0, 0]], [[1, 1]], [[1, 1]],
                               observed_coefficients=[[-.5, -.5]])
        self.assertEqual(audit_objective(ObjectiveSpec(importance_level='sequence'), trace).verdict,
                         'EXPECTED_OBJECTIVE_EFFECT')

    def test_nonfinite_input_cannot_be_explained_away_without_oracle(self):
        trace = ObjectiveTrace([[math.inf, 0]], [[0, 0]], [[1, 1]], [[0, 1]],
                               observed_coefficients=[[0, -1]])
        result = audit_objective(ObjectiveSpec(reduction='sum'), trace)
        self.assertEqual(result.verdict, 'INSUFFICIENT_EVIDENCE')
        self.assertTrue(result.to_dict()['numerical_events'])

    def test_extreme_finite_inputs_need_execution(self):
        trace = ObjectiveTrace([[1000.]], [[0]], [[1]], [[1]], observed_coefficients=[[0]])
        self.assertEqual(audit_objective(ObjectiveSpec(), trace).verdict, 'INSUFFICIENT_EVIDENCE')

    def test_target_dtype_underflow_and_input_cast_require_execution(self):
        for dtype, value in (('float16', -30.), ('bfloat16', -200.), ('float32', -200.)):
            trace = ObjectiveTrace([[value]], [[0]], [[1]], [[1]], observed_coefficients=[[0]])
            with self.subTest(dtype=dtype):
                self.assertEqual(audit_objective(ObjectiveSpec(dtype=dtype), trace).verdict,
                                 'INSUFFICIENT_EVIDENCE')
        trace = ObjectiveTrace([[-1e10]], [[-1e10]], [[1]], [[1]], observed_coefficients=[[-1]])
        self.assertEqual(audit_objective(ObjectiveSpec(dtype='float16'), trace).verdict,
                         'INSUFFICIENT_EVIDENCE')

    def test_dtype_reduction_overflow_requires_execution(self):
        trace = ObjectiveTrace([[0, 0]], [[0, 0]], [[-60000, -60000]], [[1, 1]],
                               observed_coefficients=[[30000, 30000]])
        self.assertEqual(audit_objective(ObjectiveSpec(dtype='float16'), trace).verdict,
                         'INSUFFICIENT_EVIDENCE')

    def test_clip_boundary_is_insufficient_without_observed_subgradient_rule(self):
        trace = ObjectiveTrace([[math.log(1.2)]], [[0]], [[1]], [[1]],
                               observed_coefficients=[[-1.2]])
        result = audit_objective(ObjectiveSpec(dtype='float64'), trace)
        self.assertEqual(result.verdict, 'INSUFFICIENT_EVIDENCE')
        self.assertIn('clip_boundary_subgradient_ambiguous', result.to_dict()['reasons'])

    def test_excluded_tiny_nonzero_gradient_is_not_expected(self):
        trace = ObjectiveTrace([[0, 0]], [[0, 0]], [[1, 1]], [[0, 1]],
                               observed_coefficients=[[1e-15, -1]])
        self.assertEqual(audit_objective(ObjectiveSpec(reduction='sum'), trace).verdict,
                         'INSUFFICIENT_EVIDENCE')

    def test_grpo_legacy_boundary_remains_fail_closed(self):
        from quotientguard.adapters.trl_sequence import _active_surrogate
        for advantage, ratio in ((1., 1.2), (-1., .8)):
            with self.subTest(advantage=advantage):
                with self.assertRaisesRegex(ValueError, 'boundary'):
                    _active_surrogate(advantage=advantage, ratio=ratio,
                                      epsilon_low=.2, epsilon_high=.2)


if __name__ == '__main__':
    unittest.main()
