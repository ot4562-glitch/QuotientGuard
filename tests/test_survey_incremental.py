import tempfile
import unittest
from pathlib import Path
from quotientguard.survey import Repository, SourceFile, SourceInventory, SurveyStore, SurveyEngine, SurveyLimits, SourceFetchError


class IncrementalSurveyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = SurveyStore(Path(self.tmp.name) / 'store')
        self.addCleanup(self.store.close)
        self.repo = Repository('12', 'owner/repo', 'a' * 40)

    def test_partial_is_explicitly_extended_in_new_snapshot_without_refetch(self):
        calls = []
        def fetch(repo, acquisition_plan=None):
            cached = set(acquisition_plan.exclude_paths) if acquisition_plan else set()
            calls.append(cached)
            yield SourceInventory(('a.py', 'b.py'), ())
            for path in ('a.py', 'b.py'):
                if path not in cached:
                    yield SourceFile(path, b'x=1')
        self.store.create_run('old')
        SurveyEngine(self.store, SurveyLimits(max_files=1)).run('old', [self.repo], fetch, incremental=True)
        old = self.store.export_json('old')
        self.store.create_run('new')
        report = SurveyEngine(self.store, SurveyLimits(max_files=2)).run('new', [self.repo], fetch, incremental=True)
        self.assertEqual(calls, [set(), {'a.py'}])
        self.assertEqual(report['counts']['inspected_files'], 2)
        self.assertEqual(report['counts']['completed'], 1)
        self.assertEqual(old, self.store.export_json('old'))

    def test_checkpoint_survives_interruption_before_next_fetch(self):
        self.store.create_run('run')
        def interrupted(repo, acquisition_plan=None):
            yield SourceInventory(('a.py', 'b.py'), ())
            yield SourceFile('a.py', b'x=1')
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            SurveyEngine(self.store).run('run', [self.repo], interrupted, incremental=True)
        plans = []
        def resumed(repo, acquisition_plan=None):
            plans.append(acquisition_plan.exclude_paths)
            yield SourceInventory(('a.py', 'b.py'), ())
            yield SourceFile('b.py', b'x=2')
        report = SurveyEngine(self.store).run('run', [self.repo], resumed, incremental=True)
        self.assertEqual(plans, [('a.py',)])
        self.assertEqual(report['counts']['inspected_files'], 2)

    def test_import_verified_legacy_sources_without_mutating_previous_store(self):
        old_dir = Path(self.tmp.name) / 'previous'
        previous = SurveyStore(old_dir)
        previous.create_run('old')
        SurveyEngine(previous).run('old', [self.repo], lambda r: [SourceFile('a.py', b'x=1')])
        before = previous.export_json('old')
        previous.close()
        hashes = {p.name: p.read_bytes() for p in old_dir.iterdir() if p.is_file()}
        imported = self.store.import_verified_run(old_dir, 'old')
        self.assertEqual(imported['imported_files'], 1)
        self.store.create_run('new')
        def no_fetch(repo, acquisition_plan=None):
            self.fail('complete imported source must be reused without network')
        report = SurveyEngine(self.store).run('new', [self.repo], no_fetch, incremental=True)
        self.assertEqual(report['counts']['inspected_files'], 1)
        self.assertEqual(hashes, {p.name: p.read_bytes() for p in old_dir.iterdir() if p.is_file()})

    def test_failure_retry_reuses_completed_file_checkpoint(self):
        self.store.create_run('run')
        def failed(repo, acquisition_plan=None):
            yield SourceInventory(('a.py', 'b.py'), ())
            yield SourceFile('a.py', b'x=1')
            raise SourceFetchError('NETWORK_ERROR')
        engine = SurveyEngine(self.store)
        engine.run('run', [self.repo], failed, incremental=True)
        previous_failure = self.store.export_json('run')
        plans = []
        def fetch(repo, acquisition_plan=None):
            plans.append(acquisition_plan.exclude_paths)
            yield SourceInventory(('a.py', 'b.py'), ())
            yield SourceFile('b.py', b'x=2')
        report = engine.run('run', [self.repo], fetch, incremental=True, retry_failed=True)
        self.assertEqual(plans, [('a.py',)])
        self.assertEqual(report['counts']['completed'], 1)
        self.assertEqual(previous_failure, self.store.export_json('run'))

    def test_quota_refuses_new_evidence_and_reports_partial(self):
        self.store.create_run('quota')
        # Budget includes transient rollback capacity, not just the main DB.
        self.store.max_disk_bytes = 400000
        report = SurveyEngine(self.store).run('quota', [self.repo],
            lambda r: [SourceFile('large.py', b'#' * 100000)], incremental=True)
        self.assertEqual(report['repositories'][0]['result']['failure']['kind'], 'DISK_QUOTA')
        self.assertLessEqual(self.store.disk_usage_bytes(), self.store.max_disk_bytes)

    def test_analysis_policy_change_reuses_bytes_but_reanalyses_and_preserves_old_export(self):
        from unittest.mock import patch
        self.store.create_run('v1')
        SurveyEngine(self.store).run('v1', [self.repo], lambda r: [SourceFile('a.py', b'x=1')], incremental=True)
        before = self.store.export_json('v1')
        self.store.create_run('v2')
        with patch('quotientguard.survey.ANALYSIS_VERSION', 'test-v3'):
            with patch('quotientguard.survey.inspect_source', wraps=__import__('quotientguard.survey', fromlist=['inspect_source']).inspect_source) as analysis:
                SurveyEngine(self.store).run('v2', [self.repo], lambda r: [], incremental=True)
                self.assertEqual(analysis.call_count, 1)
        self.assertEqual(before, self.store.export_json('v1'))

    def test_parser_change_reanalyses_verified_bytes_and_keeps_historical_results(self):
        import sys
        from unittest.mock import patch
        self.store.create_run('parser-a')
        SurveyEngine(self.store).run('parser-a', [self.repo],
            lambda r: [SourceInventory(('a.py',), ()), SourceFile('a.py', b'x=1')], incremental=True)
        original = self.store.export_json('parser-a')
        original_runtime = self.store.export('parser-a')['repositories'][0]['result']['parser_runtime']
        self.assertEqual(original_runtime['version'], sys.version)
        objects = {p.name: p.read_bytes() for p in self.store.objects.iterdir() if p.is_file()}
        def no_fetch(repo, acquisition_plan=None):
            self.fail('parser changes must reuse verified source bytes')
        self.store.create_run('parser-b')
        with patch.object(sys, 'version', sys.version + ' different-parser'):
            with patch('quotientguard.survey.ast.parse', side_effect=SyntaxError('different grammar')) as parse:
                changed = SurveyEngine(self.store).run('parser-b', [self.repo], no_fetch, incremental=True)
                self.assertEqual(parse.call_count, 1, 'analysis from a different parser was reused')
            self.assertEqual(changed['repositories'][0]['result']['files'][0]['status'], 'SYNTAX_ERROR')
            self.assertNotEqual(changed['repositories'][0]['result']['parser_runtime'], original_runtime)
            # A repeated run under this same parser should reuse its own analysis.
            self.store.create_run('parser-b-reuse')
            with patch('quotientguard.survey.ast.parse', side_effect=AssertionError('unexpected reparse')):
                same = SurveyEngine(self.store).run('parser-b-reuse', [self.repo], no_fetch, incremental=True)
            self.assertEqual(same['repositories'][0]['result']['files'][0]['status'], 'SYNTAX_ERROR')
        self.assertEqual(original, self.store.export_json('parser-a'))
        self.assertEqual(objects, {p.name: p.read_bytes() for p in self.store.objects.iterdir() if p.is_file()})

    def test_legacy_cache_and_existing_run_do_not_cross_parser_versions(self):
        import sys
        from unittest.mock import patch
        self.store.create_run('legacy-parser-a')
        SurveyEngine(self.store).run('legacy-parser-a', [self.repo],
            lambda r: [SourceFile('a.py', b'x=1')])
        original = self.store.export_json('legacy-parser-a')
        self.store.create_run('legacy-parser-b')
        calls = []
        def fetch(repo):
            calls.append(repo)
            return [SourceFile('a.py', b'x=1')]
        with patch.object(sys, 'version', sys.version + ' different-parser'):
            engine = SurveyEngine(self.store)
            changed = engine.run('legacy-parser-b', [self.repo], fetch)
            self.assertEqual(changed['repositories'][0]['result']['parser_runtime']['version'], sys.version)
            self.assertEqual(len(calls), 1, 'legacy repository analysis from a different parser was reused')
            with self.assertRaisesRegex(ValueError, 'configuration is immutable'):
                engine.run('legacy-parser-a', [self.repo], fetch)
        self.assertEqual(original, self.store.export_json('legacy-parser-a'))

    def test_checkpoint_is_durable_before_generator_advances(self):
        self.store.create_run('run')
        def fetch(repo, acquisition_plan=None):
            yield SourceInventory(('a.py', 'b.py'), ())
            yield SourceFile('a.py', b'x=1')
            import sqlite3
            read = sqlite3.connect(self.store.directory / 'survey.sqlite3')
            try:
                self.assertEqual(read.execute("SELECT COUNT(*) FROM file_checkpoints WHERE run_id='run'").fetchone()[0], 1)
            finally:
                read.close()
            yield SourceFile('b.py', b'x=2')
        SurveyEngine(self.store).run('run', [self.repo], fetch, incremental=True)

    def test_database_growth_is_quota_bounded_without_unhandled_sqlite_error(self):
        self.store.create_run('quota-db')
        self.store.max_disk_bytes = 400000
        report = SurveyEngine(self.store).run('quota-db', [self.repo],
            lambda r: [SourceFile(f'{i}.py', b'x=1') for i in range(100)], incremental=True)
        self.assertEqual(report['repositories'][0]['result']['failure']['kind'], 'DISK_QUOTA')
        self.assertLessEqual(self.store.disk_usage_bytes(), self.store.max_disk_bytes)

    def test_explicit_partial_resume_advances_same_handle_with_immutable_snapshots(self):
        self.store.create_run('run')
        calls = []
        def fetch(repo, acquisition_plan=None):
            calls.append(acquisition_plan.exclude_paths)
            yield SourceInventory(('a.py', 'b.py', 'c.py'), ())
            for path in ('a.py', 'b.py', 'c.py'):
                if path not in acquisition_plan.exclude_paths:
                    yield SourceFile(path, b'x=1')
        engine = SurveyEngine(self.store, SurveyLimits(max_files=1))
        first = engine.run('run', [self.repo], fetch, incremental=True)
        before = self.store.export_json('run')
        second = engine.run('run', [self.repo], fetch, incremental=True, resume_partial=True)
        self.assertEqual(second['counts']['inspected_files'], 2)
        self.assertNotEqual(second['run_id'], 'run')
        third = engine.run('run', [self.repo], fetch, incremental=True, resume_partial=True)
        self.assertEqual(third['counts']['inspected_files'], 3)
        self.assertEqual(third['counts']['completed'], 1)
        self.assertEqual(before, self.store.export_json('run'))
        again = engine.run('run', [self.repo], fetch, incremental=True, resume_partial=True)
        self.assertEqual(again, third)
        self.assertEqual(calls, [(), ('a.py',), ('a.py', 'b.py')])

    def test_explicit_paths_prioritise_requested_analysis_without_downloading_cached_bytes(self):
        self.store.create_run('seed')
        SurveyEngine(self.store).run('seed', [self.repo], lambda r: [SourceFile('a.py', b'x=1'), SourceFile('z.py', b'x=2')])
        self.store.create_run('target')
        calls = []
        def fetch(repo, acquisition_plan=None):
            calls.append(acquisition_plan)
            yield SourceInventory(('b.py',), ('a.py', 'z.py'))
            yield SourceFile('b.py', b'x=3')
        # Legacy source acquisition is not indexed automatically; explicit import
        # is covered separately. Seed verified indexed bytes in incremental mode.
        self.store.create_run('indexed')
        SurveyEngine(self.store).run('indexed', [self.repo], lambda r: [SourceFile('a.py', b'x=1'), SourceFile('z.py', b'x=2')], incremental=True)
        result = SurveyEngine(self.store, SurveyLimits(max_files=1)).run('target', [self.repo], fetch,
            incremental=True, remaining_paths=['b.py'])
        self.assertEqual([f['path'] for f in result['repositories'][0]['result']['files']], ['b.py'])
        self.assertEqual(set(calls[0].exclude_paths), {'a.py', 'z.py'})

    def test_configured_run_inventory_cannot_change_terminal_snapshot(self):
        self.store.create_run('immutable')
        engine = SurveyEngine(self.store)
        engine.run('immutable', [self.repo], lambda r: [SourceInventory(('a.py',), ()), SourceFile('a.py', b'x=1')], incremental=True)
        before = self.store.export_json('immutable')
        for other in (Repository('99', 'owner/new', 'b'*40), Repository('12', 'owner/repo', 'b'*40)):
            with self.assertRaisesRegex(ValueError, 'inventory.*immutable'):
                engine.run('immutable', [other], lambda r: self.fail('must reject before acquisition'), incremental=True)
        self.assertEqual(before, self.store.export_json('immutable'))

    def test_explicit_cached_path_needs_no_discovery_network_without_inventory(self):
        self.store.create_run('seed')
        SurveyEngine(self.store).run('seed', [self.repo], lambda r: [SourceFile('a.py', b'x=1')], incremental=True)
        self.store.create_run('target')
        def unexpected(repo, acquisition_plan=None):
            self.fail('explicit verified cached path needs no acquisition request')
        report = SurveyEngine(self.store).run('target', [self.repo], unexpected,
            incremental=True, remaining_paths=['a.py'])
        self.assertEqual(report['counts']['inspected_files'], 1)
