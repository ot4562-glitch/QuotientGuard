import unittest

from quotientguard.core import audit_receipt
from quotientguard.observations import observations_to_receipt
from quotientguard.sampled import audit_sampled_observations


def row(
    sample_index,
    trajectory_id,
    coefficient,
    *,
    score=None,
    snapshot="theta-0",
    batch="batch-0",
):
    value = {
        "policy_snapshot_id": snapshot,
        "batch_id": batch,
        "sample_index": sample_index,
        "class_id": "z",
        "trajectory_id": trajectory_id,
        "realized_coefficient": coefficient,
    }
    if score is not None:
        value["score"] = score
        value["score_kind"] = "sketch"
    return value


class ObservationTests(unittest.TestCase):
    def test_aggregate_observations_is_explicitly_sampled(self):
        observations = [
            row(0, "a", 1.0, score=[1.0, 0.0]),
            row(1, "a", 1.2, score=[1.0, 0.0]),
            row(2, "b", 1.8, score=[0.0, 1.0]),
            row(3, "b", 2.0, score=[0.0, 1.0]),
        ]
        receipt = observations_to_receipt(observations, score_kind="sketch")
        self.assertEqual(receipt["schema_version"], "quotientguard.receipt.v2")
        self.assertEqual(receipt["policy_snapshot_id"], "theta-0")
        self.assertEqual(receipt["classes"][0]["coefficient_evidence"], "sampled")

        self.assertEqual(receipt["score_space"]["kind"], "none")
        self.assertTrue(receipt["aggregation"]["source_had_scores"])
        cert = audit_receipt(receipt)
        self.assertEqual(
            cert["classes"][0]["compatibility_status"],
            "COEFFICIENT_MISMATCH_OBSERVED",
        )
        self.assertEqual(cert["summary"]["sufficiently_certified_count"], 0)

    def test_direct_sampled_projected_gap(self):
        observations = [
            row(0, "a", 1.0, score=[1.0, 0.0]),
            row(1, "a", 1.2, score=[1.0, 0.0], batch="batch-1"),
            row(2, "b", 1.8, score=[0.0, 1.0]),
            row(3, "b", 2.0, score=[0.0, 1.0], batch="batch-1"),
        ]
        cert = audit_sampled_observations(observations, score_kind="sketch")
        result = cert["classes"][0]
        self.assertEqual(
            result["compatibility_status"],
            "MEASURED_PROJECTED_SAMPLE_GAP",
        )
        self.assertAlmostEqual(result["gap"][0], -0.2, places=14)
        self.assertAlmostEqual(result["gap"][1], 0.2, places=14)
        self.assertLessEqual(result["decomposition_residual_max_abs"], 1e-14)
        self.assertEqual(result["uncertainty"]["cluster_unit"], "batch_id")
        self.assertEqual(
            result["uncertainty"]["method"],
            "leave_one_batch_out_jackknife",
        )
        self.assertEqual(result["uncertainty"]["cluster_count"], 2)
        for value in result["uncertainty"]["standard_error_vector"]:
            self.assertAlmostEqual(value, 0.0, places=14)

    def test_zero_projected_sample_gap_does_not_certify_full_space(self):
        observations = [
            row(0, "a", 1.0, score=[1.0]),
            row(1, "b", 2.0, score=[1.0]),
        ]
        cert = audit_sampled_observations(observations, score_kind="sketch")
        result = cert["classes"][0]
        self.assertEqual(
            result["compatibility_status"],
            "SAMPLED_PROJECTED_ZERO_NOT_CERTIFIED",
        )
        self.assertEqual(
            result["uncertainty"]["method"],
            "unavailable_insufficient_batches",
        )

    def test_cluster_jackknife_fails_closed_when_delete_cluster_loses_support(self):
        observations = [
            row(0, "a", 1.0, score=[1.0, 0.0], batch="batch-a"),
            row(0, "b", 2.0, score=[0.0, 1.0], batch="batch-b"),
        ]
        cert = audit_sampled_observations(observations, score_kind="sketch")
        self.assertEqual(
            cert["classes"][0]["uncertainty"]["method"],
            "unavailable_insufficient_cluster_support",
        )

    def test_fast_cluster_jackknife_matches_direct_batch_deletion(self):
        observations = [
            row(0, "a", 1.0, score=[1.0, 0.0], batch="b0"),
            row(1, "b", 1.7, score=[0.0, 1.0], batch="b0"),
            row(0, "a", 1.3, score=[1.0, 0.0], batch="b1"),
            row(1, "b", 2.2, score=[0.0, 1.0], batch="b1"),
            row(0, "a", 0.8, score=[1.0, 0.0], batch="b2"),
            row(1, "b", 2.0, score=[0.0, 1.0], batch="b2"),
        ]
        full = audit_sampled_observations(observations, score_kind="sketch")
        observed_se = full["classes"][0]["uncertainty"]["standard_error_vector"]

        delete_gaps = []
        for omitted in ("b0", "b1", "b2"):
            remaining = [
                observation
                for observation in observations
                if observation["batch_id"] != omitted
            ]
            cert = audit_sampled_observations(remaining, score_kind="sketch")
            delete_gaps.append(cert["classes"][0]["gap"])

        cluster_count = len(delete_gaps)
        expected = []
        for coordinate in range(2):
            center = (
                sum(gap[coordinate] for gap in delete_gaps) / cluster_count
            )
            variance = (
                (cluster_count - 1)
                / cluster_count
                * sum(
                    (gap[coordinate] - center) ** 2
                    for gap in delete_gaps
                )
            )
            expected.append(variance ** 0.5)

        for observed, reference in zip(observed_se, expected):
            self.assertAlmostEqual(observed, reference, places=14)

    def test_sampled_coefficient_only_equality_is_not_exact_certificate(self):
        observations = [
            row(0, "a", 1.0),
            row(1, "a", 1.0, batch="batch-1"),
            row(2, "b", 1.0),
            row(3, "b", 1.0, batch="batch-1"),
        ]
        cert = audit_sampled_observations(observations)
        self.assertEqual(
            cert["classes"][0]["compatibility_status"],
            "ESTIMATED_CLASS_CONSTANCY",
        )
        self.assertEqual(cert["summary"]["sufficiently_certified_count"], 0)

    def test_single_realization_per_trajectory_is_insufficient_if_equal(self):
        observations = [
            row(0, "a", 1.0),
            row(1, "b", 1.0),
        ]
        cert = audit_sampled_observations(observations)
        self.assertEqual(
            cert["classes"][0]["compatibility_status"],
            "INSUFFICIENT_REPETITION",
        )

    def test_realized_coefficient_difference_is_observed_mismatch(self):
        observations = [
            row(0, "a", 0.0),
            row(1, "b", 0.1),
        ]
        cert = audit_sampled_observations(observations)
        self.assertEqual(
            cert["classes"][0]["compatibility_status"],
            "COEFFICIENT_MISMATCH_OBSERVED",
        )

    def test_mixed_policy_snapshots_rejected(self):
        observations = [
            row(0, "a", 1.0, snapshot="theta-0"),
            row(1, "b", 1.0, snapshot="theta-1"),
        ]
        with self.assertRaises(ValueError):
            audit_sampled_observations(observations)

    def test_duplicate_batch_sample_key_rejected(self):
        observations = [
            row(0, "a", 1.0),
            row(0, "b", 1.0),
        ]
        with self.assertRaises(ValueError):
            audit_sampled_observations(observations)

    def test_single_trajectory_class_rejected(self):
        observations = [
            row(0, "a", 1.0),
            row(1, "a", 1.0, batch="batch-1"),
        ]
        with self.assertRaises(ValueError):
            observations_to_receipt(observations)


if __name__ == "__main__":
    unittest.main()
