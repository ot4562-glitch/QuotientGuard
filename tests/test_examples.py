import json
from pathlib import Path
import unittest

from quotientguard.core import audit_receipt


ROOT = Path(__file__).resolve().parents[1]


class ExampleTests(unittest.TestCase):
    def load(self, name):
        return json.loads((ROOT / "examples" / name).read_text(encoding="utf-8"))

    def test_compatible_example(self):
        cert = audit_receipt(self.load("compatible.json"))
        self.assertEqual(cert["summary"]["policy_status"], "PASS")
        self.assertEqual(
            cert["classes"][0]["compatibility_status"],
            "CERTIFIED_BY_EXACT_CLASS_CONSTANCY",
        )

    def test_incompatible_example(self):
        cert = audit_receipt(self.load("incompatible.json"))
        self.assertEqual(cert["summary"]["policy_status"], "FAIL")
        self.assertEqual(cert["summary"]["measured_gap_count"], 1)

    def test_coefficient_screen(self):
        cert = audit_receipt(self.load("coefficient_only.json"))
        statuses = [row["compatibility_status"] for row in cert["classes"]]
        self.assertEqual(
            statuses,
            ["CERTIFIED_BY_EXACT_CLASS_CONSTANCY", "NOT_CERTIFIED"],
        )


if __name__ == "__main__":
    unittest.main()
