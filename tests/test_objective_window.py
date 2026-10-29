"""Counter-derived normalizers and effective microbatch objective controls."""
import json
import unittest
import importlib.util
from dataclasses import replace

import quotientguard as api


@unittest.skipUnless(importlib.util.find_spec('torch'), 'install quotientguard[objective]')
class ObjectiveWindowTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(api,'audit_objective_window'),'missing normalization-window contract')
        self.policy=api.ObjectiveSpec(dtype='float64',masking='select_inputs')
        self.reference=api.FullObjectiveSpec(policy=self.policy)

    def trace(self,advantages,mask=None,**kwargs):
        zeros=[[0.]*len(row) for row in advantages]
        mask=mask or [[1]*len(row) for row in advantages]
        return api.FullObjectiveTrace(api.ObjectiveTrace(zeros,zeros,advantages,mask),**kwargs)

    def batch(self,index,trace,*,spec=None,rank=0,scale=1.,**kwargs):
        return api.ObjectiveWindowBatch(spec or self.reference,trace,rank=rank,microbatch_index=index,
                                        backward_scale=scale,**kwargs)

    def audit(self,batches,*,reference=None,counts=None,process_count=1,**kwargs):
        spec=api.ObjectiveWindowSpec(reference or self.reference,counts=counts or {},process_count=process_count)
        result=api.audit_objective_window(spec,batches,**kwargs).to_dict()
        json.dumps(result,allow_nan=False)
        return result

    def test_unequal_token_means_require_token_weighted_microbatches(self):
        batches=[self.batch(0,self.trace([[1.]]),scale=.5),self.batch(1,self.trace([[3.,3.,3.]]),scale=.5)]
        wrong=self.audit(batches)
        self.assertEqual(wrong['status'],'CONTRACT_MISMATCH_OBSERVED',wrong)
        self.assertAlmostEqual(wrong['effective_terms']['total'],-2.)
        self.assertAlmostEqual(wrong['reference_terms']['total'],-2.5)
        batches[0].backward_scale=.25
        batches[1].backward_scale=.75
        correct=self.audit(batches)
        self.assertEqual(correct['status'],'SUPPORTED',correct)
        self.assertEqual(correct['reference_gradients']['current_log_probs']['values'],[[-.25,0.,0.],[-.75,-.75,-.75]])

    def test_sum_then_single_window_division_is_not_a_mean_of_means(self):
        summed=api.FullObjectiveSpec(policy=replace(self.policy,reduction='sum'))
        batches=[self.batch(0,self.trace([[1.]]),spec=summed,scale=.25),
                 self.batch(1,self.trace([[3.,3.,3.]]),spec=summed,scale=.25)]
        result=self.audit(batches)
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertAlmostEqual(result['effective_terms']['total'],-2.5)

    def dapo_window(self):
        reference=api.FullObjectiveSpec(policy=self.policy,policy_reduction=api.ReductionSpec(
            reduction='explicit',denominator=8.,denominator_scope='accumulation_window'))
        micro=replace(reference,policy_reduction=api.ReductionSpec(reduction='explicit',denominator=4.,
            denominator_scope='accumulation_window',scale=2.))
        masks=[[[1,1]],[[1,0]],[[1,0]],[[0,0]]]
        batches=[]
        for rank in range(2):
            for index in range(2):
                mask=masks[rank*2+index]
                observed=[[-.25*m for m in mask[0]]]
                batches.append(self.batch(index,self.trace([[1.,1.]],mask),spec=micro,rank=rank,scale=.5,
                    count_masks={'policy':[[1,1]]},observed_backward_gradients={'current_log_probs':observed}))
        counts={'policy':api.DenominatorCountSpec(unit='active_tokens',mask_stage='pre_filter',
                                                declared_total=8.,divide_by_process_count=True)}
        return reference,batches,counts

    def test_prefilter_window_count_trainer_compensation_and_rank_average(self):
        reference,batches,counts=self.dapo_window()
        result=self.audit(batches,reference=reference,counts=counts,process_count=2)
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertEqual(result['count_totals']['policy']['value'],8.)
        self.assertAlmostEqual(result['effective_terms']['total'],-.5)
        self.assertTrue(result['counts_reconstructed_from_supplied_masks'])
        self.assertFalse(result['trainer_executed'])
        self.assertFalse(result['collectives_executed'])

    def test_postfilter_count_cannot_replace_prefilter_producer(self):
        reference,batches,counts=self.dapo_window()
        counts['policy']=replace(counts['policy'],declared_total=4.)
        result=self.audit(batches,reference=reference,counts=counts,process_count=2)
        self.assertEqual(result['status'],'COUNT_MISMATCH_OBSERVED',result)

    def test_missing_prefilter_mask_remains_insufficient(self):
        reference,batches,counts=self.dapo_window()
        batches[0].count_masks={}
        result=self.audit(batches,reference=reference,counts=counts,process_count=2)
        self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE')

    def test_loss_mask_must_not_activate_tokens_absent_from_prefilter_mask(self):
        reference,batches,counts=self.dapo_window()
        batches[0].count_masks={'policy':[[0,1]]}
        result=self.audit(batches,reference=reference,counts=counts,process_count=2)
        self.assertEqual(result['status'],'COUNT_MISMATCH_OBSERVED')

    def test_double_process_compensation_is_a_gradient_mismatch(self):
        reference,batches,counts=self.dapo_window()
        for batch in batches:
            batch.spec=replace(batch.spec,policy_reduction=replace(batch.spec.policy_reduction,process_count=2))
            batch.observed_backward_gradients={}
        result=self.audit(batches,reference=reference,counts=counts,process_count=2)
        self.assertEqual(result['status'],'CONTRACT_MISMATCH_OBSERVED',result)
        self.assertTrue(result['counts_reconstructed_from_supplied_masks'])

    def test_explicit_count_mismatch_does_not_pass_with_zero_advantages(self):
        spec=api.FullObjectiveSpec(policy=self.policy,policy_reduction=api.ReductionSpec(
            reduction='explicit',denominator=99.,denominator_scope='accumulation_window'))
        result=self.audit([self.batch(0,self.trace([[0.]]),spec=spec)],
            counts={'policy':api.DenominatorCountSpec(unit='active_tokens')})
        self.assertEqual(result['status'],'COUNT_MISMATCH_OBSERVED')

    def test_explicit_denominator_needs_a_count_unit(self):
        spec=api.FullObjectiveSpec(policy=self.policy,policy_reduction=api.ReductionSpec(reduction='explicit',denominator=1.))
        result=self.audit([self.batch(0,self.trace([[1.]]),spec=spec)])
        self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE')

    def test_sequence_mean_and_nonempty_mean_have_different_window_weights(self):
        traces=[self.trace([[1.,1.]]),self.trace([[3.],[1.]],[[1],[0]])]
        for reduction,weights in [('per_sequence_active_mean',(1/3,2/3)),('nonempty_sequence_mean',(.5,.5))]:
            spec=api.FullObjectiveSpec(policy=self.policy,policy_reduction=api.ReductionSpec(
                reduction=reduction,empty_mask_policy='zero',denominator_epsilon=1e-8))
            result=self.audit([self.batch(i,tr,spec=spec,scale=weights[i]) for i,tr in enumerate(traces)],reference=spec)
            self.assertEqual(result['status'],'SUPPORTED',result)

    def test_fixed_completion_slots_are_counted_from_rows_and_declared_length(self):
        spec=api.FullObjectiveSpec(policy=self.policy,policy_reduction=api.ReductionSpec(
            reduction='fixed_length_mean',completion_length=4))
        batches=[self.batch(0,self.trace([[1.,1.]]),spec=spec,scale=1/3),
                 self.batch(1,self.trace([[1.],[1.]]),spec=spec,scale=2/3)]
        result=self.audit(batches,reference=spec)
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertAlmostEqual(result['reference_terms']['total'],-1/3)

    def test_missing_rank_is_not_assumed_to_have_zero_contribution(self):
        result=self.audit([self.batch(0,self.trace([[1.]]))],process_count=2)
        self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE')

    def test_duplicate_rank_and_microbatch_index_is_rejected(self):
        with self.assertRaises(ValueError):
            self.audit([self.batch(0,self.trace([[1.]])),self.batch(0,self.trace([[1.]]))])

    def test_observed_backward_gradient_uses_trainer_scale_before_rank_average(self):
        reference,batches,counts=self.dapo_window()
        batches[0].observed_backward_gradients={'current_log_probs':[[-.5,-.5]]}
        result=self.audit(batches,reference=reference,counts=counts,process_count=2)
        self.assertEqual(result['status'],'CONTRACT_MISMATCH_OBSERVED')
        self.assertTrue(any('backward' in reason for reason in result['mismatches']))

    def test_optional_input_coordinates_are_not_invented_for_missing_batches(self):
        result=self.audit([self.batch(0,self.trace([[1.]],entropy=[[.5]])),self.batch(1,self.trace([[1.]]))])
        self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE')

    def test_supplied_reference_snapshot_must_match_the_window_inputs(self):
        result=self.audit([self.batch(0,self.trace([[1.]]))],reference_trace=self.trace([[2.]]))
        self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE')

    def test_all_empty_zero_mean_is_a_normal_zero_window(self):
        spec=api.FullObjectiveSpec(policy=replace(self.policy,empty_mask_policy='zero'))
        result=self.audit([self.batch(0,self.trace([[1.]],[[0]]),spec=spec)],reference=spec)
        self.assertEqual(result['status'],'SUPPORTED',result)
        self.assertEqual(result['effective_terms']['total'],0.)

    def test_nonfinite_observation_keeps_priority_over_unsupported_count_math(self):
        spec=api.FullObjectiveSpec(policy=self.policy,policy_reduction=api.ReductionSpec(reduction='explicit',denominator=0.))
        result=self.audit([self.batch(0,self.trace([[1.]],[[0]],observed_loss=float('nan')),spec=spec)])
        self.assertEqual(result['status'],'NUMERICAL_ANOMALY_OBSERVED')

    def test_term_specific_normalizers_preserve_live_auxiliary_gradients(self):
        reference=api.FullObjectiveSpec(policy=replace(self.policy,detach_advantage=False,detach_old_policy=False),
            policy_reduction=api.ReductionSpec(reduction='explicit',denominator=4.,denominator_scope='accumulation_window'),
            kl=api.KLSpec(coefficient=.2,detach_reference=False,reduction=api.ReductionSpec(
                reduction='explicit',denominator=2.,denominator_scope='accumulation_window')),
            entropy_coefficient=.3)
        micro=replace(reference,policy_reduction=replace(reference.policy_reduction,scale=2),
                      kl=replace(reference.kl,reduction=replace(reference.kl.reduction,scale=2)))
        batches=[self.batch(i,self.trace([[1.,2.]],reference_log_probs=[[.1,.2]],entropy=[[.7,.8]],
            kl_mask=[[1,0]],entropy_mask=[[1,0]]),spec=micro,scale=.5) for i in range(2)]
        result=self.audit(batches,reference=reference,counts={name:api.DenominatorCountSpec('active_tokens')
                                                           for name in ('policy','kl')})
        self.assertEqual(result['status'],'SUPPORTED',result)
        for name in ('reference_log_probs','advantages','old_log_probs','entropy'):
            self.assertEqual(result['effective_gradients'][name]['state'],'CONNECTED')

    def test_batch_scope_uses_all_ranks_of_the_same_microbatch(self):
        reference=replace(self.reference,policy_reduction=api.ReductionSpec(reduction='explicit',denominator=4.))
        batches=[self.batch(i,self.trace([[1.]]),rank=r,scale=.5,spec=replace(reference,
            policy_reduction=api.ReductionSpec(reduction='explicit',denominator=1.,denominator_scope='batch')))
            for r in range(2) for i in range(2)]
        result=self.audit(batches,reference=reference,process_count=2,
                          counts={'policy':api.DenominatorCountSpec('active_tokens',divide_by_process_count=True)})
        self.assertEqual(result['status'],'SUPPORTED',result)
        batches.pop()
        self.assertEqual(self.audit(batches,reference=reference,process_count=2,
            counts={'policy':api.DenominatorCountSpec('active_tokens',divide_by_process_count=True)})['status'],
            'INSUFFICIENT_EVIDENCE')

    def test_fixed_slots_count_cannot_understate_active_completion_length(self):
        spec=replace(self.reference,policy_reduction=api.ReductionSpec(reduction='explicit',denominator=1.))
        result=self.audit([self.batch(0,self.trace([[1.,1.]]),spec=spec)],reference=spec,
                          counts={'policy':api.DenominatorCountSpec('fixed_length_slots',completion_length=1)})
        self.assertEqual(result['status'],'COUNT_MISMATCH_OBSERVED',result)

    def test_window_scaling_all_cpu_dtypes_and_no_input_grad_mutation(self):
        import torch
        for dtype in ('float16','bfloat16','float32','float64'):
            with self.subTest(dtype=dtype):
                current=torch.tensor([[0.]],dtype=getattr(torch,dtype),requires_grad=True)
                current.grad=torch.ones_like(current)
                spec=replace(self.reference,policy=replace(self.policy,dtype=dtype))
                trace=api.FullObjectiveTrace(api.ObjectiveTrace(current,[[0.]],[[1.]],[[1]]))
                result=self.audit([self.batch(i,trace,spec=spec,scale=.5) for i in range(2)],reference=spec)
                self.assertEqual(result['status'],'SUPPORTED',result)
                self.assertEqual(current.grad.item(),1.)

    def test_local_half_backward_overflow_is_not_hidden_by_rank_averaging(self):
        spec=replace(self.reference,policy=replace(self.policy,dtype='float16'))
        batches=[self.batch(0,self.trace([[30000.]]),rank=r,scale=3.,spec=spec) for r in range(2)]
        result=self.audit(batches,reference=spec,process_count=2)
        self.assertEqual(result['status'],'NUMERICAL_ANOMALY_OBSERVED',result)
        self.assertIn('local',','.join(result['reasons']))

    def test_scaled_replay_reports_overflow_stage(self):
        spec=replace(self.reference,policy=replace(self.policy,dtype='float16'))
        result=self.audit([self.batch(0,self.trace([[40000.]]),scale=2.,spec=spec)],reference=spec)
        self.assertEqual(result['status'],'NUMERICAL_ANOMALY_OBSERVED')
        self.assertIsNotNone(result['microbatches'][0]['local_backward']['first_nonfinite_stage'])

    def test_replay_scale_must_be_a_finite_nonnegative_real(self):
        from quotientguard.full_objective_torch import full_objective_oracle
        for scale in (-1,float('nan'),float('inf'),True):
            with self.subTest(scale=scale),self.assertRaises(ValueError):
                full_objective_oracle(self.reference,self.trace([[1.]]),loss_scale=scale)

    def test_cli_json_and_exit_codes(self):
        from contextlib import redirect_stdout
        from dataclasses import asdict
        from io import StringIO
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from quotientguard.cli import main
        reference,batches,counts=self.dapo_window()
        spec=api.ObjectiveWindowSpec(reference,counts,process_count=2)
        with TemporaryDirectory() as directory:
            path=Path(directory)/'window.json'
            output=Path(directory)/'audit.json'
            def run(expected):
                path.write_text(json.dumps({'spec':asdict(spec),'batches':[asdict(b) for b in batches]}),encoding='utf-8')
                stream=StringIO()
                with redirect_stdout(stream):code=main(['objective-window-audit',str(path),'--json','--out',str(output)])
                self.assertEqual(code,expected)
                self.assertEqual(json.loads(stream.getvalue()),json.loads(output.read_text()))
            run(0)
            batches[0].backward_scale=1.
            run(1)
            batches[0].count_masks={}
            run(2)

    def test_half_process_average_happens_after_local_gradient_rounding(self):
        policy=replace(self.policy,dtype='float16',atol=.06,rtol=0.)
        reference=api.FullObjectiveSpec(policy=policy,policy_reduction=api.ReductionSpec(reduction='sum',scale=1/15))
        micro=api.FullObjectiveSpec(policy=policy,policy_reduction=api.ReductionSpec(reduction='sum'))
        batches=[self.batch(0,self.trace([[1931.]]),spec=micro,rank=r,scale=.2) for r in range(3)]
        result=self.audit(batches,reference=reference,process_count=3)
        self.assertEqual(result['status'],'CONTRACT_MISMATCH_OBSERVED',result)
        self.assertEqual(result['effective_gradients']['current_log_probs']['values'],[[-128.625]]*3)
        self.assertEqual(result['reference_gradients']['current_log_probs']['values'],[[-128.75]]*3)
