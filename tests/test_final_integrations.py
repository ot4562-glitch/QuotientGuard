import hashlib
import tempfile
import unittest
from pathlib import Path

from quotientguard import (
    CallbackRuntimeCapture,
    OpenRLHFRuntimeCapture,
    VerlRuntimeCapture,
    run_builtin_certification_suite,
    run_certification_suite,
    verify_source_file,
)
from quotientguard.source_manifests import SOURCE_MANIFESTS


ROOT = Path(__file__).resolve().parents[1]


class FinalIntegrationTests(unittest.TestCase):
    def _exercise_capture(self, capture):
        callback = capture.callback()
        callback(
            batch_id="b0",
            sample_index=0,
            class_id="accepted",
            trajectory_id="a",
            realized_coefficient=1.0,
            state_consuming_point="loss_contribution",
        )
        callback(
            batch_id="b0",
            sample_index=1,
            class_id="accepted",
            trajectory_id="b",
            realized_coefficient=2.0,
            state_consuming_point="loss_contribution",
        )
        cert = capture.certificate(run_id="integration-test")
        self.assertEqual(
            cert["classes"][0]["compatibility_status"],
            "COEFFICIENT_MISMATCH_OBSERVED",
        )
        self.assertEqual(
            cert["observation_metadata_summary"]["state_consuming_points"],
            ["loss_contribution"],
        )
        self.assertFalse(cert["estimator"]["reconstruction_validation"])
        return cert

    def test_verl_explicit_runtime_capture(self):
        cert = self._exercise_capture(
            VerlRuntimeCapture(
                policy_snapshot_id="theta-verl",
                framework_version="0.7.1",
            )
        )
        self.assertEqual(cert["estimator"]["framework"], "verl")

    def test_openrlhf_explicit_runtime_capture(self):
        cert = self._exercise_capture(
            OpenRLHFRuntimeCapture(
                policy_snapshot_id="theta-openrlhf",
                framework_version="0.10",
            )
        )
        self.assertEqual(cert["estimator"]["framework"], "openrlhf")

    def test_custom_callback_capture(self):
        cert = self._exercise_capture(
            CallbackRuntimeCapture(
                framework="my-trainer",
                policy_snapshot_id="theta-custom",
                source_commit="deadbeef",
            )
        )
        self.assertEqual(cert["estimator"]["adapter"], "custom_callback_runtime")

    def test_explicit_runtime_requires_framework_provenance(self):
        with self.assertRaisesRegex(ValueError, "provenance"):
            VerlRuntimeCapture(policy_snapshot_id="theta")

    def test_source_verifier_hashes_real_bytes(self):
        payload = b"quotientguard-source-verifier-fixture\n"
        digest = hashlib.sha256(payload).hexdigest()
        manifest_id = "unit-test-source"
        SOURCE_MANIFESTS[manifest_id] = {
            "framework": "unit",
            "component": "fixture.py",
            "source_commit": "fixture",
            "sha256": digest,
            "validation_evidence": {},
        }
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "fixture.py"
                path.write_bytes(payload)
                result = verify_source_file(path, manifest_id=manifest_id)
                self.assertTrue(result["verified"])
                path.write_bytes(payload + b"changed")
                with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                    verify_source_file(path, manifest_id=manifest_id)
        finally:
            SOURCE_MANIFESTS.pop(manifest_id, None)

    def test_public_certification_suite(self):
        result = run_certification_suite(ROOT / "fixtures" / "certification.json")
        self.assertEqual(result["summary"]["outcome"], "PASS")
        self.assertEqual(result["summary"]["case_count"], 5)
        self.assertEqual(result["summary"]["pass_count"], 5)

    def test_builtin_certification_suite(self):
        result = run_builtin_certification_suite()
        self.assertEqual(result["summary"]["outcome"], "PASS")
        self.assertEqual(result["summary"]["case_count"], 5)
        self.assertEqual(result["summary"]["pass_count"], 5)


if __name__ == "__main__":
    unittest.main()
