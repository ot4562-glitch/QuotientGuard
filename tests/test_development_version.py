import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_docs_for_test', ROOT / 'tools/check_release_docs.py')
release_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_docs)


class DevelopmentVersionTests(unittest.TestCase):
    def fixture(self, directory, package='1.2.0.dev0', stable='1.1.0', unreleased=True):
        root = Path(directory)
        (root / 'quotientguard').mkdir()
        (root / 'pyproject.toml').write_text('[project]\nversion = "' + package + '"\n')
        (root / 'quotientguard/version.py').write_text('__version__ = "' + package + '"\n')
        (root / 'README.md').write_text('Current stable release: **v' + stable + '**\n'
            'https://github.com/ot4562-glitch/QuotientGuard/releases/tag/v' + stable + '\n[v' + stable + ']\n')
        (root / 'CHANGELOG.md').write_text(('# Changelog\n\n## Unreleased\n\nDevelopment contracts.\n\n'
            if unreleased else '# Changelog\n\n') + '## ' + stable + ' - 2026-10-26\n\nStable notes.\n')
        return root

    def test_dev_version_preserves_current_stable_and_uses_unreleased_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(release_docs, 'ROOT', self.fixture(directory)):
                version, notes = release_docs.check_release_docs()
                self.assertEqual(version, '1.2.0.dev0')
                self.assertEqual(notes, 'Development contracts.')
                with self.assertRaisesRegex(ValueError, 'development'):
                    release_docs.check_release_docs(tag='v1.2.0.dev0')

    def test_dev_without_unreleased_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(release_docs, 'ROOT', self.fixture(directory, unreleased=False)):
                with self.assertRaisesRegex(ValueError, 'Unreleased'):
                    release_docs.check_release_docs()

    def test_stable_still_requires_exact_agreement_and_release_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(release_docs, 'ROOT', self.fixture(directory, package='1.1.0')):
                version, notes = release_docs.check_release_docs(tag='v1.1.0')
                self.assertEqual(version, '1.1.0')
                self.assertEqual(notes, 'Stable notes.')
                with self.assertRaises(ValueError):
                    release_docs.check_release_docs(tag='v1.2.0')

    def test_dev_cannot_promote_readme_to_unreleased_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(release_docs, 'ROOT', self.fixture(directory, stable='1.2.0')):
                with self.assertRaises(ValueError):
                    release_docs.check_release_docs()

    def test_local_final_package_requires_explicit_candidate_and_dated_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory, package='1.2.0', unreleased=False)
            with patch.object(release_docs, 'ROOT', root):
                with self.assertRaises(ValueError):
                    release_docs.check_release_docs()
                readme = root / 'README.md'
                readme.write_text(readme.read_text() + 'Current release candidate: **v1.2.0** (unpublished)\n')
                with self.assertRaises(ValueError):
                    release_docs.check_release_docs()
                changelog = root / 'CHANGELOG.md'
                changelog.write_text(changelog.read_text().replace('# Changelog\n',
                    '# Changelog\n\n## 1.2.0 - 2026-10-29\n\nFull objective and source survey.\n'))
                self.assertEqual(release_docs.check_release_docs(),
                                 ('1.2.0', 'Full objective and source survey.'))
                with self.assertRaisesRegex(ValueError, 'unpublished'):
                    release_docs.check_release_docs(tag='v1.2.0')

    def test_candidate_must_match_package_and_be_newer_than_stable(self):
        for package, candidate in [('1.2.0', '1.3.0'), ('1.1.0', '1.1.0')]:
            with self.subTest(package=package, candidate=candidate), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory, package=package, unreleased=False)
                readme = root / 'README.md'
                readme.write_text(readme.read_text() +
                                  f'Current release candidate: **v{candidate}** (unpublished)\n')
                with patch.object(release_docs, 'ROOT', root):
                    with self.assertRaises(ValueError):
                        release_docs.check_release_docs()


if __name__ == '__main__':
    unittest.main()
