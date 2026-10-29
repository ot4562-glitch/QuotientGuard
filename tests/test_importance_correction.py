"""Source-grounded gated old/rollout correction, independent scalar controls."""
import json
import math
import unittest
import importlib.util
from dataclasses import replace

from quotientguard import objective as api


@unittest.skipUnless(importlib.util.find_spec('torch'), 'install quotientguard[objective]')
class ImportanceCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(api,'ImportanceCorrectionSpec'),'missing logprob-to-coefficient contract')
        self.base=api.ObjectiveSpec(dtype='float64',reduction='sum',masking='literal')

    def audit(self, correction, old, rollout, *, mask=None, x=None, policy=None):
        mask=mask or [[1]*len(row) for row in old]
        x=old if x is None else x
        trace=api.FullObjectiveTrace(api.ObjectiveTrace(x,old,[[1]*len(row) for row in old],mask),
                                     rollout_log_probs=rollout)
        spec=api.FullObjectiveSpec(policy=policy or self.base,correction='gated_old_rollout',
                                   importance_correction=correction)
        result=api.audit_full_objective(spec,trace).to_dict()
        json.dumps(result,allow_nan=False)
        return result

    def test_token_clip_uses_absolute_bounds_outside_ppo(self):
        result=self.audit(api.ImportanceCorrectionSpec(mode='clip',lower=.5,upper=2.),
                          [[0.,0.,0.]],[[-math.log(.25),-math.log(1.5),-math.log(3)]])
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertEqual(result['oracle']['correction']['coefficients'],[[.5,1.5,2.]])
        self.assertEqual(result['oracle']['gradients']['current_log_probs']['values'],[[-.5,-1.5,-2.]])
        self.assertEqual(result['oracle']['gradients']['rollout_log_probs']['state'],'DETACHED')

    def test_token_rejection_keeps_token_weight_in_band(self):
        result=self.audit(api.ImportanceCorrectionSpec(mode='mask',lower=.5,upper=2.),
                          [[0.,0.,0.]],[[-math.log(.25),-math.log(1.5),-math.log(3)]])
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertEqual(result['oracle']['correction']['coefficients'],[[0.,1.5,0.]])
        self.assertEqual(result['oracle']['loss'],-1.5)

    def test_sequence_gate_keeps_distinct_token_weights(self):
        result=self.audit(api.ImportanceCorrectionSpec(level='sequence',lower=.5,upper=2.),
                          [[0.,0.],[0.,0.]],[[-math.log(.25),-math.log(4.)],[-math.log(4.),-math.log(4.)]])
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertEqual(result['oracle']['correction']['coefficients'],[[.25,4.],[0.,0.]])
        self.assertEqual(result['oracle']['gradients']['current_log_probs']['values'],[[-.25,-4.],[0.,0.]])

    def test_divergence_gate_uses_rollout_to_old_binary_kl(self):
        old=[[math.log(.5),math.log(.9)]]
        rollout=[[math.log(.5),math.log(.1)]]
        for gating in ('binary_kl','tv'):
            result=self.audit(api.ImportanceCorrectionSpec(gating=gating,lower=0,upper=.2),old,rollout)
            self.assertEqual(result['status'],'SUPPORTED',result)
            self.assertEqual(result['oracle']['correction']['keep'],[[True,False]])
            self.assertEqual(result['oracle']['correction']['coefficients'],[[1.,0.]])

    def test_binary_kl_direction_is_distinguished_by_asymmetric_gate(self):
        result=self.audit(api.ImportanceCorrectionSpec(gating='binary_kl',lower=0,upper=.65),
                          [[math.log(.6)]],[[math.log(.1)]])
        # KL(Bernoulli(.1)||Bernoulli(.6))=.550661; reverse=.750684.
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertAlmostEqual(result['oracle']['correction']['statistic'][0][0],.5506612476718904)
        self.assertTrue(result['oracle']['correction']['keep'][0][0])

    def test_correction_remains_detached_when_old_policy_is_live(self):
        result=self.audit(api.ImportanceCorrectionSpec(lower=.5,upper=2.),[[0.]],[[-math.log(1.5)]],
                          policy=replace(self.base,detach_old_policy=False))
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertEqual(result['oracle']['gradients']['old_log_probs']['values'],[[1.5]])
        self.assertEqual(result['oracle']['gradients']['rollout_log_probs']['state'],'DETACHED')

    def test_ratio_overflow_rejection_is_contained(self):
        result=self.audit(api.ImportanceCorrectionSpec(lower=.5,upper=2.),[[0.]],[[-1000.]])
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertEqual(result['oracle']['loss'],0.)
        self.assertEqual(result['oracle']['gradients']['current_log_probs']['values'],[[0.]])
        self.assertEqual(result['oracle']['first_nonfinite_stage'],'correction.exp_ratio')

    def test_half_binary_kl_nonfinite_that_changes_gate_is_numerical(self):
        result=self.audit(api.ImportanceCorrectionSpec(gating='binary_kl',lower=0,upper=.5),[[0.]],[[0.]],
                          policy=replace(self.base,dtype='float16'))
        self.assertEqual(result['status'],'NUMERICAL_ANOMALY_OBSERVED')
        self.assertTrue(result['oracle']['loss_finite'])
        self.assertTrue(result['oracle']['gradients_finite'])
        self.assertEqual(result['oracle']['first_nonfinite_stage'],'correction.token_statistic')

    def test_all_masked_sequence_gate_has_no_invented_denominator(self):
        result=self.audit(api.ImportanceCorrectionSpec(level='sequence'),[[0.,0.]],[[0.,0.]],mask=[[0,0]],
                          policy=replace(self.base,empty_mask_policy='zero'))
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertEqual(result['oracle']['correction']['coefficients'],[[0.,0.]])
        self.assertEqual(result['oracle']['first_nonfinite_stage'],'correction.sequence_statistic')

    def test_threshold_boundary_is_insufficient(self):
        result=self.audit(api.ImportanceCorrectionSpec(lower=.5,upper=2.),[[0.]],[[-math.log(2.)]])
        self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE')

    def test_excluded_token_threshold_boundary_does_not_gate_active_loss(self):
        result=self.audit(api.ImportanceCorrectionSpec(lower=1.,upper=2.),[[0.,0.]],[[-math.log(1.5),0.]],mask=[[1,0]])
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertEqual(result['oracle']['loss'],-1.5)

    def test_zero_upper_divergence_bound_remains_a_boundary(self):
        for gating in ('binary_kl','tv'):
            result=self.audit(api.ImportanceCorrectionSpec(gating=gating,lower=0,upper=0),[[math.log(.5)]],[[math.log(.5)]])
            self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE')

    def test_unsupported_source_combinations_are_not_replayed(self):
        for spec in (api.ImportanceCorrectionSpec(mode='clip',level='sequence'),
                     api.ImportanceCorrectionSpec(mode='clip',gating='tv',lower=0),
                     api.ImportanceCorrectionSpec(gating='binary_kl',lower=.5)):
            result=self.audit(spec,[[0.]],[[0.]])
            self.assertEqual(result['status'],'UNSUPPORTED',result)
            self.assertIsNone(result['oracle'])

    def test_sampling_temperature_changes_correct_density_weight(self):
        # Categorical p_old=(.8,.2), behavior temperature2 gives (2/3,1/3).
        # Correct processed rollout density produces weights(1.2,.6). Raw
        # old logprobs would instead yield(1,1), a different declared density.
        result=self.audit(api.ImportanceCorrectionSpec(lower=.1,upper=3.),
                          [[math.log(.8),math.log(.2)]],[[math.log(2/3),math.log(1/3)]])
        self.assertEqual(result['status'],'SUPPORTED',result)
        values=result['oracle']['correction']['coefficients'][0]
        self.assertAlmostEqual(values[0],1.2)
        self.assertAlmostEqual(values[1],.6)

    def test_dtype_gate_matrix_with_excluded_tokens(self):
        controls=[api.ImportanceCorrectionSpec(mode='clip',lower=.5,upper=1.2),
                  api.ImportanceCorrectionSpec(lower=.4,upper=3.5),
                  api.ImportanceCorrectionSpec(level='sequence',lower=.4,upper=3.5)]
        controls.extend(api.ImportanceCorrectionSpec(level=level,gating=gating,lower=0,upper=.5)
                        for gating in ('binary_kl','tv') for level in ('token','sequence'))
        for dtype in ('float16','bfloat16','float32','float64'):
            for correction in controls:
                with self.subTest(dtype=dtype,correction=correction):
                    result=self.audit(correction,[[math.log(.5)]*3]*2,
                        [[math.log(.2),math.log(.25),math.log(.3)]]*2,mask=[[1,1,0],[1,1,1]],
                        policy=replace(self.base,dtype=dtype))
                    self.assertEqual(result['status'],'SUPPORTED',result['reasons'])
