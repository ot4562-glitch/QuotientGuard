import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


class SurveyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('quotientguard.survey'), 'survey engine is missing')
        from quotientguard import survey
        self.s = survey
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = survey.SurveyStore(Path(self.tmp.name))
        self.addCleanup(lambda: self.store.close())
        self.repo = survey.Repository('12', 'owner/repo', 'a' * 40)

    def test_resume_dedup_hash_and_repro_plan(self):
        calls = []
        source = b'import torch\ndef loss(x, old, mask):\n    ratio = torch.exp(x-old)\n    return (ratio * mask).sum() / mask.sum()\n'
        def fetch(repo):
            calls.append(repo)
            return [self.s.SourceFile('loss.py', source), self.s.SourceFile('loss.py', source)]
        self.store.create_run('first')
        engine = self.s.SurveyEngine(self.store)
        first = engine.run('first', [self.repo, self.repo], fetch)
        again = engine.run('first', [self.repo], fetch)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first, again)
        self.assertEqual(first['counts']['repositories'], 1)
        result = first['repositories'][0]['result']
        self.assertEqual(result['files'][0]['source_sha256'], hashlib.sha256(source).hexdigest())
        candidates = result['candidates']
        self.assertTrue(candidates)
        self.assertTrue(all(c['classification'] == 'STATIC_RISK_CANDIDATE' for c in candidates))
        self.assertEqual(len({c['candidate_id'] for c in candidates}), len(candidates))
        self.assertTrue(all(c['reproduction_plan']['execution_authorized'] is False for c in candidates))
        raw = self.store.export_json('first')
        self.assertEqual(raw, self.store.export_json('first'))
        self.assertEqual(json.loads(raw)['run_id'], 'first')
        self.assertTrue((Path(self.tmp.name) / 'objects' / hashlib.sha256(source).hexdigest()).exists())

    def test_new_commit_and_compare_do_not_reuse_stale_source(self):
        self.store.create_run('old')
        self.store.create_run('new')
        engine = self.s.SurveyEngine(self.store)
        engine.run('old', [self.repo], lambda r: [self.s.SourceFile('x.py', b'x = value / mask.sum()')])
        newer = self.s.Repository('12', 'renamed/repo', 'b' * 40)
        engine.run('new', [newer], lambda r: [self.s.SourceFile('x.py', b'x = 1')])
        comparison = self.s.compare_runs(self.store, 'old', 'new')
        self.assertEqual(comparison['changed_commits'], ['12'])
        self.assertEqual(len(comparison['removed_candidate_ids']), 1)
        self.assertEqual(comparison['added_candidate_ids'], [])

    def test_checkpoint_resumes_new_store_and_records_failure_separately(self):
        self.store.create_run('run')
        repos = [self.repo, self.s.Repository('13', 'owner/other', 'c' * 40)]
        engine = self.s.SurveyEngine(self.store)
        first = engine.run('run', repos, lambda r: [], max_repositories=1)
        self.assertEqual(first['counts']['pending'], 1)
        self.store.close()
        self.store = self.s.SurveyStore(Path(self.tmp.name))
        def failed(repo):
            raise self.s.SourceFetchError('NETWORK_ERROR', 'unavailable')
        report = self.s.SurveyEngine(self.store).run('run', repos, failed)
        self.assertEqual(report['counts']['failed'], 1)
        self.assertEqual(report['counts']['completed'], 1)
        self.assertEqual(report['repositories'][1]['result']['failure']['kind'], 'NETWORK_ERROR')

    def test_syntax_decode_traversal_link_and_size_are_not_clean_results(self):
        self.store.create_run('bad')
        files = [self.s.SourceFile('bad.py', b'def (:'), self.s.SourceFile('encoding.py', b'\xff'),
                 self.s.SourceFile('../escape.py', b'x=1'), self.s.SourceFile('link.py', b'x=1', 'symlink'),
                 self.s.SourceFile('large.py', b'x' * 101)]
        report = self.s.SurveyEngine(self.store, self.s.SurveyLimits(max_file_bytes=100)).run('bad', [self.repo], lambda r: files)
        statuses = {f['status'] for f in report['repositories'][0]['result']['files']}
        self.assertEqual(statuses, {'SYNTAX_ERROR', 'DECODE_ERROR', 'UNSAFE_PATH', 'LINK_SKIPPED', 'SIZE_LIMIT'})
        self.assertEqual(report['counts']['partial'], 1)

    def test_cache_is_bound_to_limits_and_integrity(self):
        self.store.create_run('one')
        self.store.create_run('two')
        source = b'x = 1'
        self.s.SurveyEngine(self.store).run('one', [self.repo], lambda r: [self.s.SourceFile('x.py', source)])
        calls = []
        self.s.SurveyEngine(self.store, self.s.SurveyLimits(max_files=1)).run('two', [self.repo], lambda r: calls.append(r) or [])
        self.assertEqual(len(calls), 1)
        obj = Path(self.tmp.name) / 'objects' / hashlib.sha256(source).hexdigest()
        obj.write_bytes(b'corruption')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.store.export('one')

    def test_transient_failure_retry_is_explicit_and_budget_failure_stops_batch(self):
        self.store.create_run('retry')
        repos = [self.repo, self.s.Repository('13', 'owner/other', 'c' * 40)]
        calls = []
        def limited(repo):
            calls.append(repo)
            raise self.s.SourceFetchError('RATE_LIMIT')
        engine = self.s.SurveyEngine(self.store)
        report = engine.run('retry', repos, limited)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report['counts']['pending'], 1)
        report = engine.run('retry', repos, lambda r: [])
        self.assertEqual(report['counts']['failed'], 1)
        report = engine.run('retry', repos, lambda r: [], retry_failed=True)
        self.assertEqual(report['counts']['failed'], 0)
        self.assertEqual(report['counts']['completed'], 2)

    def test_identical_sources_form_families_without_independence_claim(self):
        self.store.create_run('families')
        repos = [self.repo, self.s.Repository('13', 'owner/other', 'c' * 40)]
        report = self.s.SurveyEngine(self.store).run('families', repos,
            lambda r: [self.s.SourceFile('loss.py', b'x = value / mask.sum()')])
        self.assertEqual(report['counts'].get('unique_source_hashes'), 1)
        self.assertEqual(report['counts'].get('candidate_families'), 1)
        self.assertEqual(report['counts']['candidates'], 2)
        self.assertEqual(len(report['candidate_families'][0]['candidate_ids']), 2)
        self.assertFalse(report['candidate_families'][0]['independent_implementation_confirmed'])

    def test_same_commit_cache_is_reusable_across_runs(self):
        for name in ('before', 'after'):
            self.store.create_run(name)
        engine = self.s.SurveyEngine(self.store)
        engine.run('before', [self.repo], lambda r: [self.s.SourceFile('x.py', b'x=1')])
        def unexpected_fetch(repo):
            self.fail('same pinned commit and limits should use verified cache')
        report = engine.run('after', [self.repo], unexpected_fetch)
        self.assertEqual(report['counts']['completed'], 1)

    def test_changed_configuration_fails_before_any_new_source_fetch(self):
        self.store.create_run('immutable')
        later_id = self.s.Repository('99', 'owner/later', 'd' * 40)
        self.s.SurveyEngine(self.store).run('immutable', [later_id], lambda r: [])
        calls = []
        with self.assertRaisesRegex(ValueError, 'configuration'):
            self.s.SurveyEngine(self.store, self.s.SurveyLimits(max_files=1)).run(
                'immutable', [self.repo], lambda r: calls.append(r) or [])
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
