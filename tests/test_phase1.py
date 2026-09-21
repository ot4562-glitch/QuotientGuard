import json
from pathlib import Path
import unittest

from quotientguard import (
    DecodedTextContract,
    IntegerAnswerContract,
    check_path,
    compare_certificates,
    run_paper_a1_demo,
)
from quotientguard.core import audit_receipt


ROOT = Path(__file__).resolve().parents[1]


class Phase1Tests(unittest.TestCase):
    def load(self, name):
        return json.loads((ROOT / "examples" / name).read_text(encoding="utf-8"))

    def test_integer_verifier_contract_groups_distinct_surface_forms(self):
        contract = IntegerAnswerContract()
        self.assertEqual(contract.classify("4"), contract.classify("+4"))
        self.assertEqual(contract.classify(" 04 "), contract.classify("4"))

    def test_decoded_contract_keeps_distinct_strings_separate(self):
        contract = DecodedTextContract()
        self.assertNotEqual(contract.classify("4"), contract.classify("+4"))

    def test_check_auto_detects_exact_receipt(self):
        cert = check_path(ROOT / "examples" / "compatible.json")
        self.assertEqual(
            cert["classes"][0]["compatibility_status"],
            "CERTIFIED_BY_EXACT_CLASS_CONSTANCY",
        )

    def test_check_auto_detects_jsonl_sample_gap(self):
        cert = check_path(ROOT / "examples" / "observations.jsonl")
        self.assertEqual(
            cert["classes"][0]["compatibility_status"],
            "MEASURED_PROJECTED_SAMPLE_GAP",
        )

    def test_demo_reproduces_a1_coefficient_mismatch(self):
        receipt, cert = run_paper_a1_demo()
        self.assertEqual(receipt["policy_snapshot_id"], "paper-a1-step2")
        self.assertEqual(
            cert["classes"][0]["compatibility_status"],
            "COEFFICIENT_MISMATCH_OBSERVED",
        )
        self.assertEqual(cert["summary"]["policy_status"], "OBSERVED")

    def test_compare_finds_new_gap_without_inventing_failure_policy(self):
        baseline = audit_receipt(self.load("compatible.json"))
        candidate_payload = self.load("incompatible.json")
        candidate_payload["equivalence_contract"] = self.load("compatible.json")[
            "equivalence_contract"
        ]
        candidate = audit_receipt(candidate_payload)

        result = compare_certificates(baseline, candidate)
        self.assertEqual(result["summary"]["new_measured_gap_count"], 1)
        self.assertEqual(result["summary"]["policy_status"], "OBSERVED")
        self.assertTrue(result["estimator_changed"])

    def test_compare_user_policy_can_fail_on_new_gap(self):
        baseline = audit_receipt(self.load("compatible.json"))
        candidate_payload = self.load("incompatible.json")
        candidate_payload["equivalence_contract"] = self.load("compatible.json")[
            "equivalence_contract"
        ]
        candidate = audit_receipt(candidate_payload)

        result = compare_certificates(
            baseline,
            candidate,
            fail_on_new_gap=True,
        )
        self.assertEqual(result["summary"]["policy_status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
