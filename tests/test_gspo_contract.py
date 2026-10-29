"""Hand-derived controls for rollout-reference ratios and nonempty reductions."""
import json
import math
import unittest
import importlib.util
from dataclasses import replace

from quotientguard import (FullObjectiveSpec,FullObjectiveTrace,ObjectiveSpec,ObjectiveTrace,ReductionSpec,
                           KLSpec,audit_full_objective)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'install quotientguard[objective]')
class GSPOContractTests(unittest.TestCase):
    def setUp(self):
        self.policy=ObjectiveSpec(dtype='float64',ratio_level='sequence_masked_geometric_mean',reduction='sum')

    def spec(self,**kwargs):
        return FullObjectiveSpec(policy=self.policy,correction='rollout_ratio',correction_placement='inside_clip',**kwargs)

    def trace(self,ratio=1.1,advantage=1.):
        return FullObjectiveTrace(ObjectiveTrace([[0.,0.,0.]], [[-.4,-.4,-.4]],
            [[advantage]*3],[[1,1,0]]),rollout_log_probs=[[-math.log(ratio)]*3])

    def audit(self,spec,trace):
        result=audit_full_objective(spec,trace).to_dict()
        json.dumps(result,allow_nan=False)
        return result

    def test_rollout_reference_changes_the_ratio_inside_clipping(self):
        result=self.audit(self.spec(),self.trace())
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertAlmostEqual(result['oracle']['loss'],-2.2)
        self.assertEqual(result['oracle']['gradients']['old_log_probs']['state'],'UNUSED')
        for value in result['oracle']['gradients']['current_log_probs']['values'][0][:2]:
            self.assertAlmostEqual(value,-1.1)
        self.assertEqual(result['oracle']['gradients']['rollout_log_probs']['state'],'DETACHED')
        off=self.audit(FullObjectiveSpec(policy=self.policy),self.trace())
        self.assertAlmostEqual(off['oracle']['loss'],-2.4)
        self.assertEqual(off['oracle']['gradients']['current_log_probs']['values'],[[0.,0.,0.]])

    def test_live_rollout_partial_is_sequence_coupled_and_old_is_unused(self):
        result=self.audit(self.spec(detach_rollout_policy=False),self.trace())
        self.assertEqual(result['status'],'SUPPORTED',result)
        values=result['oracle']['gradients']['rollout_log_probs']['values'][0]
        self.assertAlmostEqual(values[0],1.1)
        self.assertAlmostEqual(values[1],1.1)
        self.assertEqual(values[2],0.)
        self.assertIsNone(result['oracle']['gradients']['old_log_probs']['values'])

    def test_source_multiplies_ratio_by_mask_before_surrogate(self):
        result=self.audit(self.spec(ratio_output_masking='multiply'),self.trace())
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertAlmostEqual(result['oracle']['stages']['policy.masked_ratio'][0][0],1.1)
        self.assertEqual(result['oracle']['stages']['policy.masked_ratio'][0][2],0.)

    def test_inside_ratio_clipping_saturates_positive_and_negative_gradients(self):
        for ratio,advantage,loss in [(1.5,1.,-2.4),(.5,-1.,1.6)]:
            result=self.audit(self.spec(),self.trace(ratio,advantage))
            self.assertEqual(result['status'],'SUPPORTED',result)
            self.assertAlmostEqual(result['oracle']['loss'],loss)
            self.assertEqual(result['oracle']['gradients']['current_log_probs']['values'],[[0.,0.,0.]])

    def test_unused_old_nonfinite_does_not_enter_rollout_reference_loss(self):
        trace=self.trace()
        trace.policy.old_log_probs=[[float('nan'),-float('inf'),float('nan')]]
        result=self.audit(self.spec(),trace)
        self.assertEqual(result['status'],'SUPPORTED',result)

    def test_policy_live_rollout_and_kl_detach_are_merged_without_losing_live_path(self):
        trace=self.trace()
        trace.reference_log_probs=[[math.log(2)]*3]
        result=self.audit(self.spec(detach_rollout_policy=False,
            kl=KLSpec(coefficient=.1,importance_weight='current_rollout',detach_rollout=True)),trace)
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertEqual(result['oracle']['gradients']['rollout_log_probs']['state'],'CONNECTED')
        self.assertEqual(result['oracle']['term_gradients']['kl']['rollout_log_probs']['state'],'DETACHED')
        self.assertAlmostEqual(result['oracle']['gradients']['rollout_log_probs']['values'][0][0],1.1)

    def test_nonempty_sequence_mean_excludes_empty_rows_from_outer_count(self):
        trace=FullObjectiveTrace(ObjectiveTrace([[0,0],[0,0]],[[0,0]]*2,[[1,1]]*2,[[1,1],[0,0]]))
        reduction=ReductionSpec(reduction='nonempty_sequence_mean',denominator_epsilon=1e-8)
        result=self.audit(FullObjectiveSpec(policy=replace(self.policy,masking='select_inputs'),policy_reduction=reduction),trace)
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertAlmostEqual(result['oracle']['loss'],-2/(2+1e-8))
        self.assertAlmostEqual(result['oracle']['gradients']['current_log_probs']['values'][0][0],-1/(2+1e-8))
        legacy=self.audit(FullObjectiveSpec(policy=replace(self.policy,reduction='per_sequence_active_mean',empty_mask_policy='zero')),trace)
        self.assertAlmostEqual(legacy['oracle']['loss'],-.5)

    def test_global_nonempty_sequence_count_and_process_scaling_are_explicit(self):
        trace=FullObjectiveTrace(ObjectiveTrace([[0,0],[0,0]],[[0,0]]*2,[[1,1]]*2,[[1,1],[0,0]]))
        reduction=ReductionSpec(reduction='nonempty_sequence_mean',denominator=4.,denominator_scope='global_batch',
                                process_count=2,denominator_epsilon=1e-8)
        result=self.audit(FullObjectiveSpec(policy=self.policy,policy_reduction=reduction),trace)
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertAlmostEqual(result['oracle']['loss'],-1/(2+1e-8))

    def test_all_empty_outer_mean_is_undefined_unless_zero_is_declared(self):
        trace=FullObjectiveTrace(ObjectiveTrace([[0,0]],[[0,0]],[[1,1]],[[0,0]]))
        reduction=ReductionSpec(reduction='nonempty_sequence_mean',denominator_epsilon=1e-8)
        result=self.audit(FullObjectiveSpec(policy=self.policy,policy_reduction=reduction),trace)
        self.assertEqual(result['status'],'NUMERICAL_ANOMALY_OBSERVED')
        zero=self.audit(FullObjectiveSpec(policy=self.policy,policy_reduction=replace(reduction,empty_mask_policy='zero')),trace)
        self.assertEqual(zero['status'],'SUPPORTED',zero)
        self.assertEqual(zero['oracle']['loss'],0.)

    def test_nonempty_reduction_preserves_source_float32_sequence_selector(self):
        trace=FullObjectiveTrace(ObjectiveTrace([[0,0]],[[0,0]],[[1,1]],[[1,1]]))
        for dtype,expected in [('float16','torch.float32'),('bfloat16','torch.float32'),('float32','torch.float32'),('float64','torch.float64')]:
            spec=FullObjectiveSpec(policy=replace(self.policy,dtype=dtype),policy_reduction=
                ReductionSpec(reduction='nonempty_sequence_mean',denominator_epsilon=1e-8))
            result=self.audit(spec,trace)
            self.assertEqual(result['status'],'SUPPORTED',result)
            self.assertEqual(result['oracle']['loss_dtype'],expected)

    def test_unsupported_inside_ratio_combinations_remain_explicit(self):
        for spec in (FullObjectiveSpec(correction='rollout_ratio',correction_placement='inside_clip'),
                     FullObjectiveSpec(policy=self.policy,correction='rollout_ratio'),
                     FullObjectiveSpec(policy=self.policy,correction='weights',correction_placement='inside_clip')):
            result=self.audit(spec,self.trace())
            self.assertEqual(result['status'],'UNSUPPORTED')

    def test_rollout_reference_live_gradient_matches_finite_difference(self):
        spec=self.spec(detach_rollout_policy=False)
        trace=self.trace()
        result=self.audit(spec,trace)
        self.assertEqual(result['status'],'SUPPORTED',result)
        trace.rollout_log_probs[0][0]+=1e-6
        plus=self.audit(spec,trace)['oracle']['loss']
        trace.rollout_log_probs[0][0]-=2e-6
        minus=self.audit(spec,trace)['oracle']['loss']
        self.assertAlmostEqual((plus-minus)/2e-6,1.1,places=8)

    def test_shared_detached_kl_does_not_exempt_excluded_rollout_gradient(self):
        trace=self.trace()
        trace.reference_log_probs=[[0.,0.,0.]]
        trace.kl_mask=[[0,0,1]]
        trace.observed_gradients={'rollout_log_probs':[[1.1,1.1,1e-12]]}
        result=self.audit(self.spec(detach_rollout_policy=False,
            kl=KLSpec(coefficient=.1,importance_weight='current_rollout',detach_rollout=True)),trace)
        self.assertEqual(result['status'],'CONTRACT_MISMATCH_OBSERVED')

    def test_half_empty_inner_denominator_underflow_is_not_hidden_by_outer_selector(self):
        trace=FullObjectiveTrace(ObjectiveTrace([[0,0],[0,0]],[[0,0]]*2,[[1,1]]*2,[[1,1],[0,0]]))
        reduction=ReductionSpec(reduction='nonempty_sequence_mean',denominator_epsilon=1e-8)
        result=self.audit(FullObjectiveSpec(policy=replace(self.policy,dtype='float16'),policy_reduction=reduction),trace)
        self.assertEqual(result['status'],'NUMERICAL_ANOMALY_OBSERVED')
        self.assertEqual(result['oracle']['first_nonfinite_stage'],'policy.sequence_means')

    def test_all_empty_microbatch_with_positive_global_count_is_zero(self):
        trace=FullObjectiveTrace(ObjectiveTrace([[0,0]],[[0,0]],[[1,1]],[[0,0]]))
        reduction=ReductionSpec(reduction='nonempty_sequence_mean',denominator_epsilon=1e-8,
                                denominator=4.,denominator_scope='global_batch',process_count=2)
        result=self.audit(FullObjectiveSpec(policy=self.policy,policy_reduction=reduction),trace)
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertEqual(result['oracle']['loss'],0.)

    def test_zero_empty_policy_does_not_clamp_a_declared_explicit_denominator(self):
        trace=FullObjectiveTrace(ObjectiveTrace([[0.]],[[0.]],[[1.]],[[1]]))
        reduction=ReductionSpec(reduction='nonempty_sequence_mean',denominator=.5,denominator_scope='global_batch',
                                denominator_epsilon=1e-8,empty_mask_policy='zero')
        result=self.audit(FullObjectiveSpec(policy=self.policy,policy_reduction=reduction),trace)
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertAlmostEqual(result['oracle']['loss'],-1/(.5*(1+1e-8)))
