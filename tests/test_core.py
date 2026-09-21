import copy
import math
import unittest

from quotientguard.core import audit_receipt


def compatible_receipt(snapshot="fixture-compatible"):
    return {
        "schema_version": "quotientguard.receipt.v2",
        "policy_snapshot_id": snapshot,
        "run_id": "test-compatible",
        "policy": {"max_gap_ratio": 0.1},
        "classes": [
            {
                "class_id": "z",
                "coefficient_evidence": "exact",
                "representatives": [
                    {
                        "representative_id": "a",
                        "conditional_probability": 0.5,
                        "effective_coefficient": 1.5,
                        "score": [1.0, 0.0],
                    },
                    {
                        "representative_id": "b",
                        "conditional_probability": 0.5,
                        "effective_coefficient": 1.5,
                        "score": [0.0, 1.0],
                    },
                ],
            }
        ],
    }


class CoreTests(unittest.TestCase):
    def test_exact_class_constant_full_gap_is_zero(self):
        cert = audit_receipt(compatible_receipt())
        row = cert["classes"][0]
        self.assertEqual(
            row["compatibility_status"],
            "CERTIFIED_BY_EXACT_CLASS_CONSTANCY",
        )
        self.assertAlmostEqual(row["gap_norm"], 0.0, places=14)
        self.assertLessEqual(row["decomposition_residual_max_abs"], 1e-14)
        self.assertEqual(cert["summary"]["policy_status"], "PASS")
        self.assertEqual(cert["policy_snapshot_id"], "fixture-compatible")

    def test_nonconstant_exact_coefficient_measures_expected_gap(self):
        receipt = compatible_receipt()
        reps = receipt["classes"][0]["representatives"]
        reps[0]["effective_coefficient"] = 1.0
        reps[1]["effective_coefficient"] = 2.0
        cert = audit_receipt(receipt)
        row = cert["classes"][0]
        self.assertEqual(row["compatibility_status"], "MEASURED_GAP")
        self.assertAlmostEqual(row["gap"][0], -0.25, places=14)
        self.assertAlmostEqual(row["gap"][1], 0.25, places=14)
        self.assertAlmostEqual(row["gap_norm"], math.sqrt(0.125), places=14)
        self.assertAlmostEqual(row["gap_ratio"], math.sqrt(0.1), places=14)
        self.assertEqual(cert["summary"]["policy_status"], "FAIL")

    def test_exact_coefficient_only_constant_is_sufficient_certificate(self):
        receipt = compatible_receipt()
        receipt.pop("policy")
        for rep in receipt["classes"][0]["representatives"]:
            rep.pop("score")
        cert = audit_receipt(receipt)
        row = cert["classes"][0]
        self.assertEqual(
            row["compatibility_status"],
            "CERTIFIED_BY_EXACT_CLASS_CONSTANCY",
        )
        self.assertIsNone(row["gap_norm"])

    def test_sampled_coefficient_only_constant_is_not_exact_certificate(self):
        receipt = compatible_receipt()
        receipt.pop("policy")
        receipt["classes"][0]["coefficient_evidence"] = "sampled"
        for rep in receipt["classes"][0]["representatives"]:
            rep.pop("score")
        cert = audit_receipt(receipt)
        self.assertEqual(
            cert["classes"][0]["compatibility_status"],
            "ESTIMATED_CLASS_CONSTANCY",
        )
        self.assertEqual(cert["summary"]["sufficiently_certified_count"], 0)

    def test_sampled_aggregated_scores_are_rejected(self):
        receipt = compatible_receipt()
        receipt["classes"][0]["coefficient_evidence"] = "sampled"
        with self.assertRaisesRegex(ValueError, "audit_sampled_observations"):
            audit_receipt(receipt)

    def test_exact_coefficient_only_nonconstant_is_not_overclaimed(self):
        receipt = compatible_receipt()
        for index, rep in enumerate(receipt["classes"][0]["representatives"]):
            rep.pop("score")
            rep["effective_coefficient"] = 1.0 + index
        receipt.pop("policy")
        cert = audit_receipt(receipt)
        row = cert["classes"][0]
        self.assertEqual(row["compatibility_status"], "NOT_CERTIFIED")
        self.assertEqual(cert["summary"]["policy_status"], "OBSERVED")

    def test_invalid_probability_sum_rejected(self):
        receipt = compatible_receipt()
        receipt["classes"][0]["representatives"][0]["conditional_probability"] = 0.4
        with self.assertRaises(ValueError):
            audit_receipt(receipt)

    def test_mixed_score_presence_rejected(self):
        receipt = compatible_receipt()
        receipt["classes"][0]["representatives"][0].pop("score")
        with self.assertRaises(ValueError):
            audit_receipt(receipt)

    def test_gap_only_policy_is_not_passed_without_gap_evidence(self):
        receipt = compatible_receipt()
        for rep in receipt["classes"][0]["representatives"]:
            rep.pop("score")
        cert = audit_receipt(receipt)
        self.assertEqual(cert["classes"][0]["policy"]["status"], "OBSERVED")
        self.assertEqual(cert["summary"]["policy_status"], "OBSERVED")

    def test_receipt_requires_policy_snapshot(self):
        receipt = compatible_receipt()
        receipt.pop("policy_snapshot_id")
        with self.assertRaises(ValueError):
            audit_receipt(receipt)

    def test_score_space_none_rejects_score_vectors(self):
        receipt = compatible_receipt()
        receipt["score_space"] = {"kind": "none"}
        with self.assertRaises(ValueError):
            audit_receipt(receipt)

    def test_declared_full_score_space_rejects_unscored_receipt(self):
        receipt = compatible_receipt()
        for rep in receipt["classes"][0]["representatives"]:
            rep.pop("score")
        receipt["score_space"] = {"kind": "full"}
        with self.assertRaises(ValueError):
            audit_receipt(receipt)

    def test_certificate_is_deterministic(self):
        receipt = compatible_receipt()
        a = audit_receipt(copy.deepcopy(receipt))
        b = audit_receipt(copy.deepcopy(receipt))
        self.assertEqual(a["certificate_sha256"], b["certificate_sha256"])
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
