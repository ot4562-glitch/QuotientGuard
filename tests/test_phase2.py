import unittest

from quotientguard.adapters.trl_runtime import TRLRuntimeCapture
from quotientguard.adapters.trl_sequence import (
    REFERENCE_TRL_COMMIT,
    effective_coefficient_trace,
)
from quotientguard.core import audit_receipt
from quotientguard.demo import paper_a1_payload
from quotientguard.explain import explain_trl_payload
from quotientguard.testing import assert_no_measured_gap, assert_policy_passed


class Phase2Tests(unittest.TestCase):
    def source(self):
        return {
            "framework": "trl",
            "source_commit": REFERENCE_TRL_COMMIT,
            "source_hash": "e9f5ad165c620c8da1eceade84b00e405c76691d9af363681e8f7cbbfae8a709",
        }

    def config(self):
        return {
            "loss_type": "grpo",
            "importance_sampling_level": "sequence",
            "beta": 0.0,
            "epsilon_low": 0.2,
            "epsilon_high": 0.2,
            "batch_size": 8,
            "common_divisor": 1.0,
        }

    def test_explain_uses_same_coefficient_trace_as_adapter(self):
        payload = paper_a1_payload()
        result = explain_trl_payload(payload)
        row = result["classes"][0]
        self.assertIn("clipping_branch_differs", row["mechanisms_observed"])
        self.assertIn("sequence_ratio_differs", row["mechanisms_observed"])
        self.assertNotIn("completion_length_differs", row["mechanisms_observed"])
        self.assertEqual(row["traces"][0]["branch"], "clipped_high")
        self.assertEqual(row["traces"][0]["value"], 0.0)
        self.assertEqual(row["traces"][1]["branch"], "active_ratio")

    def test_runtime_capture_validates_production_coefficient(self):
        capture = TRLRuntimeCapture(
            policy_snapshot_id="theta-runtime",
            source_provenance=self.source(),
            config=self.config(),
            equivalence_contract={"name": "accepted-answer"},
        )
        samples = [
            {
                "advantage": 1.0,
                "sequence_ratio": 1.3488,
                "completion_length": 1,
            },
            {
                "advantage": 1.0,
                "sequence_ratio": 0.8181,
                "completion_length": 1,
            },
        ]
        for index, sample in enumerate(samples):
            trace = effective_coefficient_trace(sample, self.config())
            capture.record_sequence_sample(
                batch_id="batch-0",
                sample_index=index,
                class_id="accepted-4",
                trajectory_id=f"t-{index}",
                sample=sample,
                realized_coefficient=trace.value,
                state_consuming_point="loss_contribution",
            )

        cert = capture.certificate()
        self.assertEqual(capture.observation_count, 2)
        self.assertEqual(
            capture.observations()[0]["metadata"]["state_consuming_point"],
            "loss_contribution",
        )
        self.assertEqual(
            cert["observation_metadata_summary"]["state_consuming_points"],
            ["loss_contribution"],
        )
        self.assertEqual(
            cert["classes"][0]["compatibility_status"],
            "COEFFICIENT_MISMATCH_OBSERVED",
        )

    def test_runtime_capture_fails_on_reconstruction_mismatch(self):
        capture = TRLRuntimeCapture(
            policy_snapshot_id="theta-runtime",
            source_provenance=self.source(),
            config=self.config(),
        )
        sample = {
            "advantage": 1.0,
            "sequence_ratio": 1.0,
            "completion_length": 1,
        }
        with self.assertRaisesRegex(ValueError, "does not match"):
            capture.record_sequence_sample(
                batch_id="batch-0",
                sample_index=0,
                class_id="z",
                trajectory_id="a",
                sample=sample,
                realized_coefficient=999.0,
                state_consuming_point="loss_contribution",
            )

    def test_runtime_capture_rejects_named_but_not_consuming_location(self):
        capture = TRLRuntimeCapture(
            policy_snapshot_id="theta-runtime",
            source_provenance=self.source(),
            config=self.config(),
        )
        sample = {
            "advantage": 1.0,
            "sequence_ratio": 1.0,
            "completion_length": 1,
        }
        trace = effective_coefficient_trace(sample, self.config())
        with self.assertRaises(ValueError):
            capture.record_sequence_sample(
                batch_id="batch-0",
                sample_index=0,
                class_id="z",
                trajectory_id="a",
                sample=sample,
                realized_coefficient=trace.value,
                state_consuming_point="trainer_function_name",
            )

    def test_dependency_free_assertion_helpers(self):
        receipt = {
            "schema_version": "quotientguard.receipt.v2",
            "policy_snapshot_id": "theta",
            "policy": {"max_gap_ratio": 0.1},
            "classes": [
                {
                    "class_id": "z",
                    "coefficient_evidence": "exact",
                    "representatives": [
                        {
                            "representative_id": "a",
                            "conditional_probability": 0.5,
                            "effective_coefficient": 1.0,
                            "score": [1.0, 0.0],
                        },
                        {
                            "representative_id": "b",
                            "conditional_probability": 0.5,
                            "effective_coefficient": 1.0,
                            "score": [0.0, 1.0],
                        },
                    ],
                }
            ],
        }
        cert = audit_receipt(receipt)
        assert_policy_passed(cert)
        assert_no_measured_gap(cert)


if __name__ == "__main__":
    unittest.main()
