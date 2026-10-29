"""Bounded parameter-space checks using explicit sibling loss coordinates."""
import unittest
import importlib.util

from quotientguard import ObjectiveSpec,ObjectiveTrace,FullObjectiveSpec,FullObjectiveTrace
from quotientguard import native_validation as api


@unittest.skipUnless(importlib.util.find_spec('torch'), 'install quotientguard[objective]')
class ParameterValidationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(api,'validate_parameter_graph'),'missing parameter Jacobian contract')
        import torch
        self.torch=torch
        self.spec=FullObjectiveSpec(policy=ObjectiveSpec(dtype='float64',reduction='sum'),entropy_coefficient=.1)

    def test_independent_loss_coefficients_chain_to_actual_model_parameters(self):
        torch=self.torch
        p=torch.nn.Parameter(torch.tensor([[.01]],dtype=torch.float64))
        q=torch.nn.Parameter(torch.tensor([[.7]],dtype=torch.float64))
        x=p*2
        entropy=q*3
        loss=-x.exp().sum()-.1*entropy.sum()
        trace=FullObjectiveTrace(ObjectiveTrace(x,torch.zeros_like(x),torch.ones_like(x),torch.ones_like(x)),entropy=entropy)
        p.grad=torch.ones_like(p)*123
        report=api.validate_parameter_graph(self.spec,trace,loss,{'p':p,'q':q})
        self.assertEqual(report['status'],'PASS',report)
        self.assertAlmostEqual(report['expected_gradients']['p'][0][0],-2*torch.exp(p*2).item())
        self.assertAlmostEqual(report['expected_gradients']['q'][0][0],-.3)
        self.assertEqual(p.grad.item(),123)
        self.assertIsNone(q.grad)

    def test_hidden_parameter_branch_is_detected_even_with_equal_forward(self):
        torch=self.torch
        p=torch.nn.Parameter(torch.tensor([[0.]],dtype=torch.float64))
        q=torch.nn.Parameter(torch.tensor([[1.]],dtype=torch.float64))
        x=p*2
        entropy=torch.ones_like(x,requires_grad=True)
        loss=-x.exp().sum()-.1*entropy.sum()+(q-q.detach()).sum()
        trace=FullObjectiveTrace(ObjectiveTrace(x,torch.zeros_like(x),torch.ones_like(x),torch.ones_like(x)),entropy=entropy)
        report=api.validate_parameter_graph(self.spec,trace,loss,{'p':p,'q':q})
        self.assertEqual(report['status'],'CONTRACT_MISMATCH_OBSERVED',report)
        self.assertIsNone(report['expected_gradients']['q'])
        self.assertEqual(report['observed_gradients']['q'],[[1.]])

    def test_dependent_coordinate_boundary_is_explicitly_unsupported(self):
        torch=self.torch
        p=torch.nn.Parameter(torch.tensor([[0.]],dtype=torch.float64))
        x=p*2
        entropy=x*3
        trace=FullObjectiveTrace(ObjectiveTrace(x,torch.zeros_like(x),torch.ones_like(x),torch.ones_like(x)),entropy=entropy)
        report=api.validate_parameter_graph(self.spec,trace,-x.exp().sum()-.1*entropy.sum(),{'p':p})
        self.assertEqual(report['status'],'UNSUPPORTED')
        self.assertIn('dependent_input_coordinates',report['reasons'])

    def test_frozen_ancestor_is_unsupported_rather_than_detach_defect(self):
        torch=self.torch
        p=torch.nn.Parameter(torch.tensor([[.01]],dtype=torch.float64))
        x=p*2
        trace=FullObjectiveTrace(ObjectiveTrace(x,p,torch.ones_like(x),torch.ones_like(x)))
        spec=FullObjectiveSpec(policy=self.spec.policy)
        report=api.validate_parameter_graph(spec,trace,-(x-p.detach()).exp().sum(),{'p':p})
        self.assertEqual(report['status'],'UNSUPPORTED',report)

    def test_retained_coordinate_is_rejected_before_mutating_its_grad(self):
        torch=self.torch
        p=torch.nn.Parameter(torch.tensor([[.01]],dtype=torch.float64))
        x=p*2
        x.retain_grad()
        x.grad=torch.ones_like(x)*123
        trace=FullObjectiveTrace(ObjectiveTrace(x,torch.zeros_like(x),torch.ones_like(x),torch.ones_like(x)))
        report=api.validate_parameter_graph(FullObjectiveSpec(policy=self.spec.policy),trace,-x.exp().sum(),{'p':p})
        self.assertEqual(report['status'],'UNSUPPORTED',report)
        self.assertEqual(x.grad.item(),123)
