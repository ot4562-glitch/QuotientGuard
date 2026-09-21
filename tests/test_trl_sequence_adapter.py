import unittest

from quotientguard.adapters.trl_sequence import trl_sequence_batch_to_receipt
from quotientguard.core import audit_receipt


class TrlSequenceAdapterTests(unittest.TestCase):
    def base_payload(self, loss_type="grpo"):
        return {
            "schema_version": "trl-sequence-v2",
            "policy_snapshot_id": "theta-0",
            "batch_id": "batch-0",
            "run_id": "test",
            "source_provenance": {
                "framework": "trl",
                "source_commit": "a98fa6a4428f9aae58dfb26d729d7437f662f27a",
                "source_hash": "e9f5ad165c620c8da1eceade84b00e405c76691d9af363681e8f7cbbfae8a709",
            },
            "config": {
                "loss_type": loss_type,
                "importance_sampling_level": "sequence",
                "beta": 0.0,
                "epsilon_low": 0.2,
                "epsilon_high": 0.2,
                "batch_size": 8,
                "max_completion_length": 4,
                "active_count": 16,
            },
            "samples": [
                {
                    "class_id": "z",
                    "trajectory_id": "a",
                    "advantage": 1.0,
                    "sequence_ratio": 1.0,
                    "completion_length": 1,
                },
                {
                    "class_id": "z",
                    "trajectory_id": "b",
                    "advantage": 1.0,
                    "sequence_ratio": 1.0,
                    "completion_length": 2,
                },
            ],
        }

    def test_grpo_length_enters_effective_coefficient(self):
        receipt = trl_sequence_batch_to_receipt(self.base_payload("grpo"))
        reps = receipt["classes"][0]["representatives"]
        by_id = {rep["trajectory_id"]: rep for rep in reps}
        self.assertAlmostEqual(by_id["a"]["effective_coefficient"], 1.0 / 8.0)
        self.assertAlmostEqual(by_id["b"]["effective_coefficient"], 1.0 / 16.0)

    def test_dr_grpo_fixed_denominator_is_sampled_class_constancy(self):
        receipt = trl_sequence_batch_to_receipt(self.base_payload("dr_grpo"))
        reps = receipt["classes"][0]["representatives"]
        self.assertAlmostEqual(
            reps[0]["effective_coefficient"],
            reps[1]["effective_coefficient"],
        )
        cert = audit_receipt(receipt)
        self.assertEqual(
            cert["classes"][0]["compatibility_status"],
            "ESTIMATED_CLASS_CONSTANCY",
        )

    def test_bnpo_batch_denominator_is_common(self):
        receipt = trl_sequence_batch_to_receipt(self.base_payload("bnpo"))
        reps = receipt["classes"][0]["representatives"]
        self.assertAlmostEqual(reps[0]["effective_coefficient"], 1.0 / 16.0)
        self.assertAlmostEqual(reps[1]["effective_coefficient"], 1.0 / 16.0)

    def test_equal_length_clipping_separates_coefficients(self):
        payload = self.base_payload("grpo")
        payload["samples"] = [
            {
                "class_id": "accepted-answer-4",
                "trajectory_id": "a",
                "advantage": 1.0,
                "sequence_ratio": 1.3488,
                "completion_length": 1,
            },
            {
                "class_id": "accepted-answer-4",
                "trajectory_id": "b",
                "advantage": 1.0,
                "sequence_ratio": 0.8181,
                "completion_length": 1,
            },
        ]
        receipt = trl_sequence_batch_to_receipt(payload)
        reps = {
            r["trajectory_id"]: r
            for r in receipt["classes"][0]["representatives"]
        }
        self.assertEqual(reps["a"]["effective_coefficient"], 0.0)
        self.assertAlmostEqual(reps["b"]["effective_coefficient"], 0.8181 / 8.0)
        cert = audit_receipt(receipt)
        self.assertEqual(
            cert["classes"][0]["compatibility_status"],
            "COEFFICIENT_MISMATCH_OBSERVED",
        )

    def test_beta_nonzero_rejected(self):
        payload = self.base_payload()
        payload["config"]["beta"] = 0.01
        with self.assertRaises(ValueError):
            trl_sequence_batch_to_receipt(payload)

    def test_snapshot_is_required(self):
        payload = self.base_payload()
        payload.pop("policy_snapshot_id")
        with self.assertRaises(ValueError):
            trl_sequence_batch_to_receipt(payload)

    def test_source_provenance_is_required(self):
        payload = self.base_payload()
        payload.pop("source_provenance")
        with self.assertRaisesRegex(ValueError, "source_provenance"):
            trl_sequence_batch_to_receipt(payload)

    def test_unknown_trl_commit_fails_closed(self):
        payload = self.base_payload()
        payload["source_provenance"]["source_commit"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported TRL source commit"):
            trl_sequence_batch_to_receipt(payload)

    def test_missing_trl_source_hash_fails_closed(self):
        payload = self.base_payload()
        payload["source_provenance"].pop("source_hash")
        with self.assertRaisesRegex(ValueError, "source hash"):
            trl_sequence_batch_to_receipt(payload)

    def test_wrong_trl_source_hash_fails_closed(self):
        payload = self.base_payload()
        payload["source_provenance"]["source_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source hash"):
            trl_sequence_batch_to_receipt(payload)

    def test_trajectory_id_is_required(self):
        payload = self.base_payload()
        payload["samples"][0].pop("trajectory_id")
        with self.assertRaises(ValueError):
            trl_sequence_batch_to_receipt(payload)


if __name__ == "__main__":
    unittest.main()
