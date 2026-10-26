import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'install quotientguard[objective]')
class ObjectiveBenchmarkTests(unittest.TestCase):
    def test_independent_controls_and_holdout(self):
        import quotientguard
        self.assertTrue(hasattr(quotientguard, 'run_objective_benchmark'),
                        'independent objective benchmark is missing')
        result = quotientguard.run_objective_benchmark()
        self.assertEqual(result['outcome'], 'PASS')
        for partition in ('controls', 'holdout'):
            metrics = result[partition]['metrics']
            self.assertEqual(metrics['false_positives'], 0)
            self.assertEqual(metrics['false_negatives'], 0)
            self.assertGreater(metrics['normal_cases'], 0)
            self.assertGreater(metrics['numerical_cases'], 0)
            self.assertGreater(metrics['insufficient_cases'], 0)
        self.assertEqual(len(result['historical_review']['repositories']), 10)
        self.assertTrue(all(r['execution'] == 'NOT_EXECUTED'
                            for r in result['historical_review']['repositories']))


if __name__ == '__main__':
    unittest.main()
