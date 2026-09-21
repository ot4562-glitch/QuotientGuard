import unittest

from quotientguard.benchmark import benchmark_sampled_audit
from quotientguard.core import audit_receipt
from quotientguard.sentinel import summarize_certificates
from quotientguard.sketch import HashProjection, sketch_observations


def exact_receipt(
    snapshot,
    *,
    coefficient_b=1.0,
    estimator_name="estimator-a",
    captured_at=None,
):
    receipt = {
        "schema_version": "quotientguard.receipt.v2",
        "policy_snapshot_id": snapshot,
        "estimator": {"name": estimator_name},
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
                        "effective_coefficient": coefficient_b,
                        "score": [0.0, 1.0],
                    },
                ],
            }
        ],
    }
    if captured_at is not None:
        receipt["snapshot_metadata"] = {"captured_at": captured_at}
    return receipt


class Phase3Tests(unittest.TestCase):
    def test_hash_projection_is_deterministic(self):
        projection = HashProjection(dimension=8, seed="test-seed")
        vector = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(projection.project(vector), projection.project(vector))
        self.assertEqual(len(projection.project(vector)), 8)

    def test_nonzero_projection_implies_nonzero_source_for_basis_witness(self):
        projection = HashProjection(dimension=4, seed="test-seed")
        projected = projection.project([0.0, 7.0, 0.0])
        self.assertTrue(any(value != 0.0 for value in projected))
        metadata = projection.metadata(3)
        self.assertIn("zero projection does not certify", metadata["semantics"])

    def test_sketch_observations_preserves_fixed_snapshot_fields(self):
        observations = [
            {
                "policy_snapshot_id": "theta",
                "batch_id": "b",
                "sample_index": 0,
                "class_id": "z",
                "trajectory_id": "a",
                "realized_coefficient": 1.0,
                "score": [1.0, 0.0, 0.0],
            },
            {
                "policy_snapshot_id": "theta",
                "batch_id": "b",
                "sample_index": 1,
                "class_id": "z",
                "trajectory_id": "b",
                "realized_coefficient": 2.0,
                "score": [0.0, 1.0, 0.0],
            },
        ]
        sketched, metadata = sketch_observations(
            observations,
            dimension=2,
            seed="fixture",
        )
        self.assertEqual(sketched[0]["policy_snapshot_id"], "theta")
        self.assertEqual(sketched[0]["score_kind"], "sketch")
        self.assertEqual(len(sketched[0]["score"]), 2)
        self.assertEqual(metadata["source_dimension"], 3)
        self.assertEqual(metadata["sketch_dimension"], 2)

    def test_sentinel_tracks_snapshot_metadata(self):
        certs = [
            audit_receipt(
                exact_receipt(
                    "step-1",
                    captured_at="2026-09-21T10:00:00+09:00",
                )
            ),
            audit_receipt(
                exact_receipt(
                    "step-2",
                    captured_at="2026-09-21T10:01:00+09:00",
                )
            ),
        ]
        result = summarize_certificates(certs)
        self.assertEqual(result["summary"]["first_policy_snapshot_id"], "step-1")
        self.assertEqual(result["summary"]["last_policy_snapshot_id"], "step-2")
        self.assertEqual(
            result["snapshot_metadata"][0]["captured_at"],
            "2026-09-21T10:00:00+09:00",
        )

    def test_sentinel_max_gap_ratio_policy(self):
        cert = audit_receipt(
            exact_receipt("step-gap", coefficient_b=2.0)
        )
        result = summarize_certificates([cert], max_gap_ratio=0.1)
        self.assertEqual(result["summary"]["outcome"], "ALERT")
        self.assertEqual(result["classes"][0]["violations"][0]["rule"], "max_gap_ratio")

    def test_sentinel_can_require_stable_estimator_fingerprint(self):
        certs = [
            audit_receipt(
                exact_receipt("step-1", estimator_name="estimator-a")
            ),
            audit_receipt(
                exact_receipt("step-2", estimator_name="estimator-b")
            ),
        ]
        result = summarize_certificates(certs, require_same_estimator=True)
        self.assertEqual(result["summary"]["outcome"], "ALERT")
        self.assertEqual(result["summary"]["global_violation_count"], 1)
        self.assertEqual(
            result["policy"]["global_violations"][0]["rule"],
            "require_same_estimator",
        )

    def test_local_benchmark_reports_scope_without_overclaim(self):
        result = benchmark_sampled_audit(
            sample_count=16,
            score_dimension=4,
            repeats=2,
        )
        self.assertEqual(
            result["schema_version"],
            "quotientguard.benchmark.v1",
        )
        self.assertGreaterEqual(result["median_ms"], 0.0)
        self.assertIn("not an end-to-end", result["scope_note"])


if __name__ == "__main__":
    unittest.main()
