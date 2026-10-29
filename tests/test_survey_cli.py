import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from quotientguard.cli import main, build_parser


class SurveyCLITests(unittest.TestCase):
    def test_commands_are_exposed(self):
        commands = build_parser()._subparsers._group_actions[0].choices
        for name in ('survey-discover', 'survey-inventory', 'survey-screen', 'survey-export', 'survey-compare'):
            self.assertIn(name, commands)

    def test_resource_budget_flags_and_memory_limited_continuation(self):
        from quotientguard.survey import SourceFile, SourceInventory
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / 'inventory.json'
            inventory.write_text(json.dumps({'repositories': [dict(repo_id='1', full_name='unit/repo', commit_sha='a'*40)]}))
            paths = ('a.py', 'b.py', 'c.py')
            downloads = []
            def fetch(repo, acquisition_plan=None):
                yield SourceInventory(paths, ())
                for path in paths:
                    if path not in acquisition_plan.exclude_paths:
                        downloads.append(path)
                        yield SourceFile(path, b'x=1')
            args = ['survey-screen', str(inventory), '--store', str(root/'store'), '--run-id', 'run',
                    '--incremental', '--max-inventory-bytes', '4096', '--max-report-bytes', '65536']
            with patch('quotientguard.survey_github.GitHubClient.fetch_files',
                       new=lambda self, repo, acquisition_plan=None: fetch(repo, acquisition_plan)):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(args + ['--max-result-bytes', '10', '--out', str(root/'small.json')]), 0)
                    self.assertEqual(main(args + ['--max-result-bytes', '4096', '--resume-partial', '--out', str(root/'full.json')]), 0)
            self.assertEqual(json.loads((root/'small.json').read_text())['counts']['partial'], 1)
            full = json.loads((root/'full.json').read_text())
            self.assertEqual(full['counts']['completed'], 1)
            self.assertEqual(downloads, list(paths))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(['survey-export', '--store', str(root/'store'), '--run-id', full['run_id'],
                               '--max-report-bytes', '1024', '--out', str(root/'refused.json')])
            self.assertEqual(status, 2)
            self.assertIn('REPORT_BYTES_LIMIT', output.getvalue())
            self.assertFalse((root/'refused.json').exists())

    def test_screen_resume_export_compare_and_request_bound(self):
        from quotientguard.survey import SourceFile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / 'inventory.json'
            inventory.write_text(json.dumps({'repositories': [dict(repo_id='1', full_name='unit/repo', commit_sha='a'*40)]}))
            args = ['survey-screen', str(inventory), '--store', str(root / 'store'), '--run-id', 'one']
            with patch('quotientguard.survey_github.GitHubClient.fetch_files', return_value=[
                    SourceFile('loss.py', b'loss = weight * policy_loss\n')]) as fetch:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(args), 0)
                    self.assertEqual(main(args), 0)
                self.assertEqual(fetch.call_count, 1)
            out = root / 'export.json'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['survey-export', '--store', str(root / 'store'), '--run-id', 'one', '--out', str(out)]), 0)
                self.assertEqual(main(['survey-compare', 'one', 'one', '--store', str(root / 'store')]), 0)
            data = json.loads(out.read_text())
            self.assertEqual(data['counts']['completed'], 1)
            self.assertNotIn('SEMANTIC_MISMATCH_CONFIRMED', out.read_text())


if __name__ == '__main__':
    unittest.main()
