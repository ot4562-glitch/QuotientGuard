import copy
import unittest

from quotientguard.core import audit_receipt
from quotientguard.sentinel import summarize_certificates


def receipt(snapshot, coefficient_b=2.0, with_scores=True):
    reps = [
        {
            "representative_id": "a",
            "conditional_probability": 0.5,
            "effective_coefficient": 1.0,
            "score": [1.0, 0.0],
        },
        {
            "representative_id": "b",
            "conditional_probability": 0.5,
            "effective_coefficient": coefficient_b,
            "score": [0.0, 1.0],
        },
    ]
    if not with_scores:
        for rep in reps:
            rep.pop("score")
    return {
        "schema_version": "quotientguard.receipt.v2",
        "policy_snapshot_id": snapshot,
        "run_id": snapshot,
        "classes": [
            {
                "class_id": "z",
                "coefficient_evidence": "exact",
                "representatives": reps,
            }
        ],
    }


class SentinelTests(unittest.TestCase):
    def test_repeated_measured_gap_can_alert(self):
        certs = [
            audit_receipt(copy.deepcopy(receipt(f"step-{i}")))
            for i in range(3)
        ]
        result = summarize_certificates(certs, max_measured_gap_events=1)
        row = result["classes"][0]
        self.assertEqual(row["measured_gap_events"], 3)
        self.assertEqual(row["max_consecutive_gap_events"], 3)
        self.assertEqual(row["outcome"], "ALERT")
        self.assertEqual(result["summary"]["outcome"], "ALERT")

    def test_no_policy_means_observed(self):
        certs = [audit_receipt(receipt("step-0", coefficient_b=1.0))]
        result = summarize_certificates(certs)
        self.assertEqual(result["summary"]["outcome"], "OBSERVED")

    def test_uncertified_is_separate_from_measured_gap(self):
        certs = [
            audit_receipt(receipt(f"step-{i}", with_scores=False))
            for i in range(2)
        ]
        result = summarize_certificates(certs, max_uncertified_events=1)
        row = result["classes"][0]
        self.assertEqual(row["measured_gap_events"], 0)
        self.assertEqual(row["uncertified_events"], 2)
        self.assertEqual(row["outcome"], "ALERT")

    def test_certified_history_passes_declared_policy(self):
        certs = [
            audit_receipt(receipt(f"step-{i}", coefficient_b=1.0))
            for i in range(2)
        ]
        result = summarize_certificates(
            certs,
            max_measured_gap_events=0,
            max_uncertified_events=0,
            max_consecutive_gap_events=0,
        )
        self.assertEqual(result["summary"]["outcome"], "PASS")

    def test_duplicate_snapshot_is_rejected(self):
        cert = audit_receipt(receipt("step-0"))
        with self.assertRaises(ValueError):
            summarize_certificates([cert, copy.deepcopy(cert)])


if __name__ == "__main__":
    unittest.main()
