"""Measure actual evidence bytes at transaction boundaries, not just after commit."""
import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from quotientguard.survey import (Repository, SourceFetchError, SourceFile, SourceInventory,
                                 SurveyEngine, SurveyLimits, SurveyStore, canonical_json)


class SurveyDiskResourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # Constructor failures may leave unreachable sqlite handles on older code.
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(gc.collect)
        self.directory = Path(self.tmp.name) / 'store'

    def usage(self):
        return sum(p.stat().st_size for p in self.directory.rglob('*') if p.is_file())

    def test_schema_bootstrap_cannot_overshoot_small_disk_budget(self):
        with self.assertRaises(SourceFetchError) as error:
            SurveyStore(self.directory, max_disk_bytes=16384)
        self.assertEqual(error.exception.kind, 'DISK_QUOTA')
        self.assertLessEqual(self.usage(), 16384)

    def test_create_run_reserves_active_journal_not_only_committed_database(self):
        store = SurveyStore(self.directory)
        self.addCleanup(store.close)
        peaks = []
        store.db.create_function('observe_bytes', 0, lambda: peaks.append(self.usage()) or 0)
        store.db.execute('CREATE TRIGGER measure_run AFTER INSERT ON runs BEGIN SELECT observe_bytes(); END')
        store.db.commit()
        budget = self.usage() + 4096
        store.max_disk_bytes = budget
        try:
            store.create_run('small-write')
        except SourceFetchError as error:
            self.assertEqual(error.kind, 'DISK_QUOTA')
        self.assertLessEqual(max(peaks + [self.usage()]), budget)
        self.assertEqual(store.db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')

    def test_refusing_too_tight_reopen_preserves_previous_evidence(self):
        store = SurveyStore(self.directory)
        store.create_run('original', {'context': 'immutable'})
        store.close()
        before = {p.name: p.read_bytes() for p in self.directory.iterdir() if p.is_file()}
        with self.assertRaises(SourceFetchError) as error:
            SurveyStore(self.directory, max_disk_bytes=self.usage() + 4096)
        self.assertEqual(error.exception.kind, 'DISK_QUOTA')
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.directory.iterdir() if p.is_file()})

    def test_large_last_object_cannot_consume_terminal_failure_snapshot_reservation(self):
        store = SurveyStore(self.directory, max_disk_bytes=800000)
        self.addCleanup(store.close)
        store.create_run('run')
        repo = Repository('1', 'test/resource', 'a'*40)
        def fetch(repo, acquisition_plan=None):
            yield SourceInventory(tuple(f'{i}.py' for i in range(100)), ())
            for i in range(99):
                yield SourceFile(f'{i}.py', b'x=1')
            yield SourceFile('99.py', b'#' + b'x'*200000)
        report = SurveyEngine(store).run('run', [repo], fetch, incremental=True)
        result = report['repositories'][0]['result']
        self.assertEqual(result['failure']['kind'], 'DISK_QUOTA')
        self.assertEqual(result['status'], 'FAILED')
        self.assertEqual(report['counts']['inspected_files'], 99)
        self.assertEqual(store.db.execute("SELECT status FROM inventory WHERE run_id='run'").fetchone()[0], 'FAILED')
        self.assertEqual(len(list(store.objects.iterdir())), 1, 'declined large source must not become an orphan object')
        self.assertLessEqual(store.disk_usage_bytes(), 800000)
        self.assertEqual(report, store.export('run'))

    def test_legacy_source_admission_preserves_both_cache_and_result_capacity(self):
        for quota in (500000, 550000):
            with self.subTest(quota=quota):
                store = SurveyStore(self.directory / str(quota), max_disk_bytes=quota)
                self.addCleanup(store.close)
                store.create_run('legacy')
                repo = Repository('1', 'test/resource', 'a'*40)
                def fetch(repo):
                    yield SourceInventory(tuple(f'{i}.py' for i in range(100)), ())
                    for i in range(99):
                        yield SourceFile(f'{i}.py', b'x=1')
                    yield SourceFile('99.py', b'#' + b'x'*200000)
                report = SurveyEngine(store).run('legacy', [repo], fetch)
                result = report['repositories'][0]['result']
                self.assertEqual(result['failure']['kind'], 'DISK_QUOTA')
                self.assertEqual(result['status'], 'FAILED')
                self.assertGreater(report['counts']['inspected_files'], 0)
                self.assertEqual(len(list(store.objects.iterdir())), 1)
                self.assertLessEqual(store.disk_usage_bytes(), quota)
                self.assertEqual(report, store.export('legacy'))


class SurveyMemoryAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = SurveyStore(Path(self.tmp.name) / 'store')
        self.addCleanup(self.store.close)
        self.repo = Repository('17', 'owner/many-files', 'a' * 40)

    def test_result_buffer_limit_can_expand_without_redownloading_overflow_file(self):
        paths = tuple(f'{i:04}.py' for i in range(100))
        downloads = []
        def fetch(repo, acquisition_plan=None):
            yield SourceInventory(paths, ())
            for path in paths:
                if path not in acquisition_plan.exclude_paths:
                    downloads.append(path)
                    yield SourceFile(path, b'x=1')
        self.store.create_run('small')
        small = SurveyEngine(self.store, SurveyLimits(max_result_bytes=2500)).run(
            'small', [self.repo], fetch, incremental=True)
        result = small['repositories'][0]['result']
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertEqual(result['limit_reason'], 'RESULT_BYTES_LIMIT')
        self.assertLessEqual(len(canonical_json(result['files'])) + len(canonical_json(result['candidates'])), 2500)
        snapshot = self.store.export_json('small')
        self.store.create_run('expanded')
        expanded = SurveyEngine(self.store, SurveyLimits(max_result_bytes=128000)).run(
            'expanded', [self.repo], fetch, incremental=True)
        self.assertEqual(expanded['counts']['completed'], 1)
        self.assertEqual(expanded['counts']['inspected_files'], 100)
        self.assertEqual(len(downloads), 100)
        self.assertEqual(len(set(downloads)), 100)
        self.assertEqual(snapshot, self.store.export_json('small'))

    def test_large_inventory_is_rejected_before_next_source_and_before_persistence(self):
        self.store.create_run('bounded')
        def fetch(repo, acquisition_plan=None):
            yield SourceInventory(tuple('directory/' + 'a' * 100 + f'{i}.py' for i in range(200)), ())
            self.fail('must stop acquisition at inventory admission failure')
        report = SurveyEngine(self.store, SurveyLimits(max_inventory_bytes=1024)).run(
            'bounded', [self.repo], fetch, incremental=True)
        result = report['repositories'][0]['result']
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertEqual(result['limit_reason'], 'INVENTORY_BYTES_LIMIT')
        self.assertIsNone(result['coverage']['known_paths'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM source_inventories').fetchone()[0], 0)

    def test_cached_result_budget_does_not_refetch_already_cached_tail(self):
        paths = ('a.py', 'b.py', 'c.py')
        self.store.create_run('seed')
        SurveyEngine(self.store).run('seed', [self.repo],
            lambda repo, acquisition_plan=None: [SourceInventory(paths, ())] + [SourceFile(p, b'x=1') for p in paths], incremental=True)
        self.store.create_run('tight')
        report = SurveyEngine(self.store, SurveyLimits(max_result_bytes=10)).run(
            'tight', [self.repo], lambda *a, **k: self.fail('bounded cached report must not trigger downloads'), incremental=True)
        self.assertEqual(report['repositories'][0]['result']['limit_reason'], 'RESULT_BYTES_LIMIT')

    def test_same_budget_resume_is_rejected_but_expanded_budget_advances(self):
        paths = ('a.py', 'b.py', 'c.py')
        downloads = []
        def fetch(repo, acquisition_plan=None):
            yield SourceInventory(paths, ())
            for path in paths:
                if path not in acquisition_plan.exclude_paths:
                    downloads.append(path)
                    yield SourceFile(path, b'x=1')
        self.store.create_run('run')
        small = SurveyEngine(self.store, SurveyLimits(max_result_bytes=10))
        small.run('run', [self.repo], fetch, incremental=True)
        snapshot = self.store.export_json('run')
        with self.assertRaises(SourceFetchError) as error:
            small.run('run', [self.repo], fetch, incremental=True, resume_partial=True)
        self.assertEqual(error.exception.kind, 'RESULT_BYTES_LIMIT')
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM runs').fetchone()[0], 1)
        advanced = SurveyEngine(self.store, SurveyLimits(max_result_bytes=4096)).run(
            'run', [self.repo], fetch, incremental=True, resume_partial=True)
        self.assertEqual(advanced['counts']['completed'], 1)
        self.assertEqual(downloads, list(paths))
        self.assertEqual(snapshot, self.store.export_json('run'))

    def test_empty_inventory_policy_still_obeys_byte_budget(self):
        self.store.create_run('policy')
        report = SurveyEngine(self.store, SurveyLimits(max_inventory_bytes=128)).run(
            'policy', [self.repo], lambda repo, acquisition_plan=None: [SourceInventory((), (), selection_policy='界' * 20)], incremental=True)
        self.assertEqual(report['repositories'][0]['result']['limit_reason'], 'INVENTORY_BYTES_LIMIT')

    def test_remote_inventory_labels_do_not_become_local_source_paths(self):
        self.store.create_run('remote-label')
        omitted = 'logs/Apr-12_20:21:26/policy.py'
        report = SurveyEngine(self.store).run('remote-label', [self.repo],
            lambda repo, acquisition_plan=None: [SourceInventory(('a.py',), (omitted,)), SourceFile('a.py', b'x=1')], incremental=True)
        result = report['repositories'][0]['result']
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertIsNone(result['failure'])
        self.assertEqual(result['coverage']['uninspected_paths'], [omitted])
        self.assertEqual(result['files'][0]['path'], 'a.py')
        self.assertEqual(len(list(self.store.objects.iterdir())), 1)

    def test_existing_large_inventory_is_not_deserialized_under_tighter_budget(self):
        self.store.create_run('seed')
        paths = tuple(f'directory/{i:05}.py' for i in range(200))
        SurveyEngine(self.store).run('seed', [self.repo],
            lambda repo, acquisition_plan=None: [SourceInventory(paths, ()), SourceFile(paths[0], b'x=1')], incremental=True)
        self.store.create_run('tight')
        oversized_inventory = self.store.db.execute('SELECT inventory FROM source_inventories').fetchone()[0]
        self.assertGreater(len(oversized_inventory), 1024)
        original_loads = __import__('json').loads
        def bounded_loads(text, *args, **kwargs):
            self.assertNotEqual(text, oversized_inventory, 'oversized cached inventory reached the JSON parser')
            return original_loads(text, *args, **kwargs)
        with patch('quotientguard.survey.json.loads', side_effect=bounded_loads):
            report = SurveyEngine(self.store, SurveyLimits(max_inventory_bytes=1024)).run(
                'tight', [self.repo], lambda *a, **k: self.fail('no acquisition after admission failure'), incremental=True)
        self.assertEqual(report['repositories'][0]['result']['limit_reason'], 'INVENTORY_BYTES_LIMIT')

    def test_report_preflight_rejects_before_deserializing_large_rows(self):
        self.store.close()
        # This independent store uses a deliberately tiny report budget.
        self.store = SurveyStore(Path(self.tmp.name) / 'limited', max_export_bytes=1024)
        self.addCleanup(self.store.close)
        self.store.create_run('large', {'padding': 'x' * 4096})
        with patch('quotientguard.survey.json.loads', side_effect=AssertionError('must preflight before JSON loads')):
            with self.assertRaises(SourceFetchError) as error:
                self.store.export('large')
        self.assertEqual(error.exception.kind, 'REPORT_BYTES_LIMIT')

    def test_prior_import_admits_rows_before_deserializing_them(self):
        self.store.create_run('seed')
        SurveyEngine(self.store).run('seed', [self.repo], lambda repo: [SourceFile('a.py', b'x=1')])
        target = SurveyStore(Path(self.tmp.name) / 'import-target', max_export_bytes=1024)
        self.addCleanup(target.close)
        with patch('quotientguard.survey.json.loads', side_effect=AssertionError('oversized import must not be parsed')):
            with self.assertRaises(SourceFetchError) as error:
                target.import_verified_run(self.store.directory, 'seed')
        self.assertEqual(error.exception.kind, 'IMPORT_RECORD_BYTES_LIMIT')
        self.assertEqual(list(target.objects.iterdir()), [])
