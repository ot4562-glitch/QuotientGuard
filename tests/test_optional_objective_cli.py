"""Optional full/window commands explain missing dependencies without traceback."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OptionalObjectiveCLITests(unittest.TestCase):
    def test_no_site_packages_returns_actionable_dependency_error(self):
        for command, filename in [('full-objective-audit', 'full-objective.json'),
                                  ('objective-window-audit', 'objective-window.json')]:
            with self.subTest(command=command):
                result = subprocess.run([sys.executable, '-S', '-m', 'quotientguard', command,
                    str(ROOT/'examples'/filename)], cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('quotientguard[objective]', result.stderr)
                self.assertNotIn('Traceback', result.stderr)
