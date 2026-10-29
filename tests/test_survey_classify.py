"""Conservative static classification; fixtures are never executed."""
import ast
import hashlib
import unittest

from quotientguard.survey_classify import classify_python, classify_repository, compare_objective_components


LOSS = '''import torch
def policy_loss(log_probs, old_log_probs, advantages, mask):
    ratio = torch.exp(log_probs - old_log_probs)
    losses = -ratio * advantages
    return (losses * mask).sum() / mask.sum()
'''


class ClassificationTests(unittest.TestCase):
    def test_leading_utf8_bom_is_parsed_without_changing_source_identity(self):
        source = '\ufeff' + LOSS
        digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
        report = classify_python(source, 'actor.py', expected_sha256=digest)
        self.assertEqual(report['status'], 'INSPECTED')
        self.assertEqual(report['source_sha256'], digest)
        component = report['components'][0]
        self.assertEqual(component['kind'], 'LOCAL_POLICY_OBJECTIVE_IMPLEMENTATION')
        self.assertEqual(component['source_sha256'], digest)
        self.assertEqual(component['start_line'], 2)
        plain = classify_python(LOSS, 'actor.py')['components'][0]
        self.assertEqual(component['contract']['expression_ast_sha256'], plain['contract']['expression_ast_sha256'])
        self.assertEqual(component['contract']['expression'], plain['contract']['expression'])
        self.assertTrue(all(s['source_sha256'] == digest for s in component['contract']['source_statements']))
        # A scope starting on the BOM line must also have correct byte offsets.
        first_line_scope = LOSS.split('\n', 1)[1]
        first = classify_python('\ufeff' + first_line_scope, 'actor.py')['components'][0]
        self.assertEqual(first['start_line'], 1)
        self.assertEqual(first['excerpt'], first_line_scope.rstrip('\n'))
        # Hash checks attest the original bytes, including the BOM.
        without_bom = hashlib.sha256(LOSS.encode('utf-8')).hexdigest()
        self.assertEqual(classify_python(source, 'actor.py', expected_sha256=without_bom)['status'], 'HASH_MISMATCH')

    def test_embedded_or_repeated_bom_remains_invalid_python(self):
        for source in ('\ufeff\ufeff' + LOSS, LOSS + '\ufeff'):
            with self.subTest(source=repr(source[:15])):
                self.assertEqual(classify_python(source, 'actor.py')['status'], 'SYNTAX_ERROR')

    def test_local_loss_is_source_bound_but_independence_unknown(self):
        report = classify_python(LOSS, "actor.py")
        components = report["components"]
        self.assertEqual(components[0]["kind"], "LOCAL_POLICY_OBJECTIVE_IMPLEMENTATION")
        self.assertEqual(components[0]["source_sha256"], hashlib.sha256(LOSS.encode()).hexdigest())
        self.assertEqual(components[0]["start_line"], 2)
        self.assertEqual(components[0]["end_line"], 5)
        self.assertEqual(components[0]["contract"]["status"], "STATIC_EXPRESSION_GRAPH_ONLY")
        repo = classify_repository([report], omitted_paths=["other.py"])
        self.assertEqual(repo["whole_repository_classification"], "UNKNOWN")
        self.assertFalse(repo["independent_implementation_confirmed"])

    def test_external_trainer_wrapper_is_not_local_loss(self):
        text = 'from trl import GRPOTrainer\ndef main():\n    trainer = GRPOTrainer()\n    trainer.train()\n'
        report = classify_python(text, "train.py")
        self.assertEqual([c["kind"] for c in report["components"]], ["EXTERNAL_TRAINER_WRAPPER"])
        self.assertFalse(any(c.get("contract") for c in report["components"]))

    def test_keywords_comments_and_unrelated_exp_are_not_implementation(self):
        text = '# PPO GRPO RLVR loss advantages\nimport torch\ndef evaluate(log_probs, old_log_probs, advantages):\n    ratio = torch.exp(log_probs - old_log_probs)\n    print(advantages)\n    return ratio.mean()\n'
        report = classify_python(text, "ppo_trainer.py")
        self.assertFalse(any(c["kind"] == "LOCAL_POLICY_OBJECTIVE_IMPLEMENTATION" for c in report["components"]))

    def test_loss_must_flow_to_return_or_backward(self):
        text = LOSS.replace('return (losses * mask).sum() / mask.sum()', 'return 0')
        self.assertEqual(classify_python(text, "loss.py")["components"], [])

    def test_nested_function_does_not_taint_parent(self):
        text = 'def unrelated():\n' + '\n'.join('    ' + l for l in LOSS.splitlines()) + '\n    return 0\n'
        names = [c["symbol"] for c in classify_python(text, "x.py")["components"]]
        self.assertEqual(names, ["unrelated.policy_loss"])

    def test_dynamic_control_flow_retains_fragment_without_contract(self):
        text = LOSS.replace('    losses =', '    if config.enabled:\n        losses =').replace('    return', '    return')
        report = classify_python(text, "actor.py")
        self.assertEqual(report["components"][0]["contract"]["status"], "UNKNOWN")

    def test_hash_mismatch_and_syntax_failure_do_not_classify(self):
        self.assertEqual(classify_python(LOSS, "x.py", expected_sha256="0" * 64)["status"], "HASH_MISMATCH")
        self.assertEqual(classify_python('def broken(:', "x.py")["status"], "SYNTAX_ERROR")

    def test_fork_metadata_and_distinct_hash_do_not_prove_change(self):
        left = classify_python(LOSS, "actor.py")
        right = classify_python('# comment\n' + LOSS, "actor.py")
        comparison = compare_objective_components(left, right)
        self.assertEqual(comparison[0]["status"], "SAME_STATIC_OBJECTIVE_EXPRESSION")
        repo = classify_repository([right], relationship={"is_fork": True, "parent": "a/b"})
        self.assertFalse(repo["independent_implementation_confirmed"])
        self.assertEqual(repo["relationship_status"], "FORK_METADATA_ONLY")

    def test_matching_path_symbol_graph_difference_supports_modified_component_only(self):
        a = classify_python(LOSS, "actor.py")
        b = classify_python(LOSS.replace('mask.sum()', '(mask.sum() + 1)'), "actor.py")
        self.assertEqual(compare_objective_components(a, b)[0]["status"], "STATIC_OBJECTIVE_EXPRESSION_CHANGED")
        self.assertEqual(compare_objective_components(a, classify_python(LOSS, "other.py")), [])

    def test_eval_dataset_evidence_never_excludes_omitted_training(self):
        report = classify_python('from datasets import load_dataset\ndef evaluate():\n    return load_dataset("test")\n', "eval.py")
        repo = classify_repository([report], omitted_paths=["train.py"])
        self.assertEqual(repo["whole_repository_classification"], "UNKNOWN")
        self.assertEqual(repo["selected_component_classification"], "EVALUATION_OR_DATA_COMPONENT_CANDIDATE")

    def test_test_source_is_not_production_implementation(self):
        report = classify_python(LOSS, "tests/test_loss.py")
        self.assertEqual(report["components"][0]["kind"], "TEST_POLICY_OBJECTIVE_EXAMPLE")

    def test_wrapper_requires_import_resolution_and_training_call(self):
        for text in ('class Trainer: pass\nx = Trainer()\nx.train()\n', 'from trl import GRPOTrainer\nx = GRPOTrainer()\n'):
            self.assertEqual(classify_python(text, "train.py")["components"], [])

    def test_rebinding_external_trainer_does_not_confirm_wrapper(self):
        text = 'from trl import GRPOTrainer\nx = GRPOTrainer()\nx = something_else()\nx.train()\n'
        self.assertEqual(classify_python(text, "train.py")["components"], [])

    def test_unreachable_second_return_cannot_restore_contract(self):
        text = LOSS.replace('    return (losses', '    return 0\n    return (losses')
        component = classify_python(text, "actor.py")["components"][0]
        self.assertEqual(component["contract"]["status"], "UNKNOWN")

    def test_contract_only_annotates_return_dependencies(self):
        text = LOSS.replace('    ratio =', '    unused_mask = mask.sum()\n    ratio =')
        contract = classify_python(text, "actor.py")["components"][0]["contract"]
        self.assertNotIn('unused_mask', '\n'.join(s["excerpt"] for s in contract["source_statements"]))

    def test_advantage_weighting_helper_is_not_a_loss_implementation(self):
        text = 'import torch\ndef apply_weighting(log_probs, old_log_probs, advantages):\n    weight = torch.exp(log_probs - old_log_probs)\n    return advantages * weight\n'
        self.assertEqual(classify_python(text, "core_algos.py")["components"], [])

    def test_shadowed_import_is_not_confirmed_external_wrapper(self):
        for text in (
            'from trl import GRPOTrainer\nGRPOTrainer = LocalTrainer\nx = GRPOTrainer()\nx.train()\n',
            'from trl import GRPOTrainer\ndef main(GRPOTrainer):\n    x = GRPOTrainer()\n    x.train()\n',
        ):
            self.assertEqual(classify_python(text, "train.py")["components"], [])

    def test_valid_deep_ast_returns_analysis_limit_without_aborting_batch(self):
        text = 'x=' + '+'.join(['x'] * 1100)
        ast.parse(text)
        reports = [classify_python(text, "long.py"), classify_python(LOSS, "actor.py")]
        self.assertEqual(reports[0]["status"], "ANALYSIS_LIMIT")
        self.assertEqual(reports[0]["components"], [])
        self.assertEqual(reports[0]["source_sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(reports[1]["status"], "INSPECTED")
        self.assertTrue(reports[1]["components"])

    def test_enclosing_argument_shadows_external_trainer_import(self):
        text = 'from trl import GRPOTrainer\ndef main(GRPOTrainer):\n    def inner():\n        x = GRPOTrainer()\n        x.train()\n    return inner\n'
        self.assertFalse(any(c["kind"] == "EXTERNAL_TRAINER_WRAPPER" for c in classify_python(text, "train.py")["components"]))

    def test_enclosing_rebindings_and_dynamic_bindings_fail_closed(self):
        for binding in ('GRPOTrainer = supplied', 'for GRPOTrainer in supplied:\n        pass', 'with supplied as GRPOTrainer:\n        pass'):
            text = 'from trl import GRPOTrainer\ndef main():\n    ' + binding + '\n    def inner():\n        x = GRPOTrainer()\n        x.train()\n    return inner\n'
            self.assertFalse(any(c["kind"] == "EXTERNAL_TRAINER_WRAPPER" for c in classify_python(text, "train.py")["components"]))

    def test_unshadowed_closure_and_enclosing_local_import_still_recognized(self):
        for text in (
            'from trl import GRPOTrainer\ndef main():\n    def inner():\n        x = GRPOTrainer()\n        x.train()\n    return inner\n',
            'def main():\n    from trl import GRPOTrainer\n    def inner():\n        x = GRPOTrainer()\n        x.train()\n    return inner\n',
        ):
            components = classify_python(text, "train.py")["components"]
            self.assertEqual([(c["symbol"], c["external_constructor"]) for c in components], [("main.inner", "trl.GRPOTrainer")])


if __name__ == "__main__":
    unittest.main()
