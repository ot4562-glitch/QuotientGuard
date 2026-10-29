"""Syntax-risk controls: these labels never establish an upstream defect."""
import ast
import importlib.util
import unittest

from quotientguard.survey import _patterns


class MissingMaskPatternTests(unittest.TestCase):
    def test_direct_mask_multiply_is_a_risk_candidate(self):
        patterns = {item[1] for item in _patterns(ast.parse(
            'def loss(weight, mask):\n    return (weight * mask).sum()\n'))}
        self.assertIn('MASK_MULTIPLY_NONFINITE', patterns)

    def test_masked_subtraction_is_a_risk_candidate(self):
        patterns = {item[1] for item in _patterns(ast.parse(
            'def loss(current, old, mask):\n    return ((current-old)*mask).sum()\n'))}
        self.assertIn('MASKED_SUBTRACTION', patterns)


class PatternDataflowTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('quotientguard.survey_patterns'), 'bounded dataflow analyzer missing')
        from quotientguard.survey_patterns import analyze_source
        self.analyze = analyze_source

    def labels(self, text):
        return {x['pattern'] for x in self.analyze(text)['candidates']}

    def test_assigned_and_aliased_exp_reaches_later_mask(self):
        source = 'import torch as th\ndef loss(x, old, mask):\n    d=x-old\n    ratio=th.exp(d)\n    copy=ratio\n    return (copy*mask).sum()\n'
        self.assertIn('EXP_BEFORE_MASK', self.labels(source))

    def test_selection_before_exp_is_distinct_from_selection_after_exp(self):
        before = 'def loss(x, mask):\n    selected=torch.where(mask, x, 0.)\n    return (selected.exp()*mask).sum()\n'
        after = 'def loss(x, mask):\n    return torch.where(mask, x.exp(), 0.).sum()\n'
        self.assertNotIn('EXP_BEFORE_MASK', self.labels(before))
        self.assertIn('WHERE_AFTER_UNSAFE_ARITHMETIC', self.labels(after))

    def test_selected_inputs_avoid_masked_subtraction_hazard(self):
        source = 'def loss(x, old, mask):\n    x=torch.where(mask,x,0.)\n    old=torch.where(mask,old,0.)\n    return ((x-old)*mask).sum()\n'
        self.assertNotIn('MASKED_SUBTRACTION', self.labels(source))

    def test_detached_rejection_and_normal_clip_are_not_declared_bugs(self):
        source = 'def loss(x, mask):\n    weight=x.exp().detach()\n    accepted=torch.where(mask,weight,0.)\n    return accepted.sum()\n'
        result = self.analyze(source)
        self.assertNotIn('WHERE_AFTER_UNSAFE_ARITHMETIC', {x['pattern'] for x in result['candidates']})
        normal = self.analyze('def loss(ratio,adv):\n    return torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()\n')
        self.assertTrue(any(x['kind'] == 'CLIPPING_OR_REJECTION' for x in normal['observations']))
        self.assertTrue(all(x['classification'] == 'STATIC_RISK_CANDIDATE' for x in result['candidates']))
        self.assertFalse(result['semantic_confirmation_eligible'])

    def test_guarded_count_and_padded_mean_are_different_observations(self):
        guarded = self.analyze('def loss(value,mask):\n    return value.sum()/mask.sum().clamp_min(1)\n')
        self.assertNotIn('MASK_SUM_DENOMINATOR', {x['pattern'] for x in guarded['candidates']})
        self.assertTrue(any(x['kind'] == 'ACTIVE_TOKEN_DENOMINATOR' and x['zero_guarded'] for x in guarded['observations']))
        padded = self.analyze('def loss(value,mask):\n    return (value*mask).mean()\n')
        self.assertTrue(any(x['kind'] == 'FULL_TENSOR_MEAN' for x in padded['observations']))

    def test_sanitization_keeps_raw_value_provenance(self):
        result = self.analyze('def loss(delta):\n    raw=delta.exp()\n    cleaned=torch.nan_to_num(raw)\n    return cleaned.sum()\n')
        candidate = next(x for x in result['candidates'] if x['pattern'] == 'NONFINITE_REPLACEMENT')
        self.assertIn('exp', candidate['operations'])

    def test_sequence_aggregation_and_detach_are_observations(self):
        result = self.analyze('def loss(x,old,mask):\n    delta=x-old.detach()\n    seq=(delta*mask).sum(-1)/mask.sum(-1).clamp_min(1)\n    return seq.exp().detach()\n')
        kinds = {x['kind'] for x in result['observations']}
        self.assertTrue({'SEQUENCE_AGGREGATION', 'STOP_GRADIENT'} <= kinds)

    def test_detached_nonfinite_coefficient_reenters_a_live_product(self):
        result = self.analyze('def loss(x,old,mask):\n    coefficient=(x-old).exp().clamp(0.,1.28).detach()\n    value=coefficient*x\n    return torch.where(mask,value,0.).sum()\n')
        candidates = [c for c in result['candidates'] if c['pattern'] == 'DETACHED_COEFFICIENT_PRODUCT']
        self.assertEqual(len(candidates), 1)
        self.assertIn('exp', candidates[0]['operations'])
        self.assertEqual(candidates[0]['classification'], 'STATIC_RISK_CANDIDATE')
        self.assertFalse(result['semantic_confirmation_eligible'])

    def test_detached_coefficient_through_constant_scaling_is_tracked(self):
        source = 'def loss(x,old,mask):\n    coefficient=(x-old).exp().detach()*2.\n    return torch.where(mask,coefficient*x,0.).sum()\n'
        self.assertIn('DETACHED_COEFFICIENT_PRODUCT', self.labels(source))

    def test_detached_products_and_boolean_rejection_are_negative_controls(self):
        for source in (
            'def loss(x,old,mask):\n    c=(x-old).exp().detach()\n    return torch.where(mask,c*x.detach(),0.).sum()\n',
            'def loss(x,old,mask):\n    c=((x-old).exp().detach()<1.3).float()\n    return torch.where(mask,c*x,0.).sum()\n',
            'def loss(x,mask):\n    c=x.exp().detach()\n    c=1.\n    return torch.where(mask,c*x,0.).sum()\n'):
            self.assertNotIn('DETACHED_COEFFICIENT_PRODUCT', self.labels(source))

    def test_boolean_and_finite_mask_products_are_negative_controls(self):
        for source in ('def f(mask):\n    return mask*mask\n', 'def f(mask):\n    return 1.*mask\n'):
            self.assertNotIn('MASK_MULTIPLY_NONFINITE', self.labels(source))

    def test_reassignment_and_unknown_branches_do_not_retain_false_safety(self):
        replaced = 'def f(x,mask):\n    value=x.exp()\n    value=1.\n    return value*mask\n'
        self.assertNotIn('EXP_BEFORE_MASK', self.labels(replaced))
        branched = 'def f(x,mask,flag):\n    if flag:\n        value=torch.where(mask,x,0.)\n    else:\n        value=x\n    return value.exp()*mask\n'
        self.assertIn('EXP_BEFORE_MASK', self.labels(branched))

    def test_nonzero_fallback_and_later_arithmetic_are_not_exp_safety_proofs(self):
        for expression in ('torch.where(mask,x,1000.).exp()',
                           '(torch.where(mask,x,0.)+1000.).exp()',
                           'torch.where(other_mask,x,0.).exp()'):
            self.assertIn('EXP_BEFORE_MASK', self.labels(
                'def loss(x,mask,other_mask):\n    return '+expression+'*mask\n'))

    def test_masked_fill_zero_before_exp_is_recognized(self):
        self.assertNotIn('EXP_BEFORE_MASK', self.labels(
            'def loss(x,mask):\n    return x.masked_fill(mask==0,0.).exp()*mask\n'))

    def test_lower_clamp_invalidates_zero_fallback_before_exp(self):
        self.assertIn('EXP_BEFORE_MASK', self.labels(
            'def loss(x,mask):\n    return torch.where(mask,x,0.).clamp(min=1000.).exp()*mask\n'))

    def test_keyword_tensor_inputs_retain_unsafe_provenance(self):
        self.assertIn('WHERE_AFTER_UNSAFE_ARITHMETIC', self.labels(
            'def loss(x,mask):\n    return torch.where(condition=mask,input=x.exp(),other=0.).sum()\n'))
        result = self.analyze('def loss(x):\n    raw=x.exp()\n    return torch.nan_to_num(input=raw).sum()\n')
        candidate = next(c for c in result['candidates'] if c['pattern'] == 'NONFINITE_REPLACEMENT')
        self.assertIn('exp', candidate['operations'])


if __name__ == '__main__':
    unittest.main()
