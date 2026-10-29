import importlib.util
import hashlib
import json
import unittest
from urllib.parse import parse_qs, urlparse


class GitHubSurveyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('quotientguard.survey_github'), 'GitHub survey source is missing')
        from quotientguard import survey_github
        self.g = survey_github

    def response(self, payload, status=200, headers=None):
        return self.g.HTTPResponse(status, headers or {}, json.dumps(payload).encode())

    def test_search_splits_dates_without_overlap_and_deduplicates(self):
        queries = []
        def transport(url, headers):
            q = parse_qs(urlparse(url).query)['q'][0]
            queries.append(q)
            if '2026-01-01..2026-01-04' in q:
                return self.response({'total_count': 1500, 'items': [], 'incomplete_results': False})
            return self.response({'total_count': 1, 'items': [{'id': 12, 'full_name': 'a/b'}], 'incomplete_results': False})
        result = self.g.GitHubClient(transport=transport).discover('PPO language:Python', '2026-01-01', '2026-01-04')
        self.assertEqual(len(result['repositories']), 1)
        self.assertEqual(len(queries), 3)
        self.assertTrue(any('2026-01-01..2026-01-02' in q for q in queries))
        self.assertTrue(any('2026-01-03..2026-01-04' in q for q in queries))
        self.assertTrue(result['coverage_complete'])

    def test_pagination_single_day_cap_is_explicit(self):
        calls = []
        def transport(url, headers):
            page = int(parse_qs(urlparse(url).query)['page'][0])
            calls.append(page)
            return self.response({'total_count': 1200, 'items': [{'id': page * 100 + i, 'full_name': f'a/r{i}'} for i in range(100)]})
        report = self.g.GitHubClient(transport=transport).discover('PPO', '2026-01-01', '2026-01-01')
        self.assertEqual(len(report['repositories']), 1000)
        self.assertEqual(calls, list(range(1, 11)))
        self.assertFalse(report['coverage_complete'])
        self.assertEqual(report['partitions'][0]['status'], 'SEARCH_CAP')

    def test_missing_results_are_not_reported_as_complete_coverage(self):
        report = self.g.GitHubClient(transport=lambda u, h: self.response({'total_count': 1, 'items': []})).discover(
            'PPO', '2026-01-01', '2026-01-01')
        self.assertFalse(report['coverage_complete'])
        self.assertEqual(report['partitions'][0]['status'], 'SHORT_PAGE')

    def test_rate_limit_not_retried_and_failure_is_counted_without_token(self):
        calls = []
        def transport(url, headers):
            calls.append(url)
            return self.response({'message': 'denied secret-token'}, 403, {'X-RateLimit-Remaining': '0'})
        result = self.g.GitHubClient(token='secret-token', transport=transport).discover('PPO', '2026-01-01', '2026-01-01')
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['failure_count'], 1)
        self.assertEqual(result['failures'][0]['kind'], 'RATE_LIMIT')
        self.assertNotIn('secret-token', json.dumps(result))

    def test_tree_filter_commit_and_blob_fetch_are_bounded(self):
        calls = []
        def transport(url, headers):
            calls.append(url)
            if '/commits/' in url:
                return self.response({'sha': 'a' * 40})
            if '/git/trees/' in url:
                return self.response({'truncated': False, 'tree': [
                    {'path': 'good.py', 'type': 'blob', 'mode': '100644', 'size': 3,
                     'sha': hashlib.sha1(b'blob 3\0x=1').hexdigest()},
                    {'path': 'link.py', 'type': 'blob', 'mode': '120000', 'size': 3, 'sha': 'c' * 40},
                    {'path': '../bad.py', 'type': 'blob', 'mode': '100644', 'size': 3, 'sha': 'd' * 40}]})
            return self.response({'encoding': 'base64', 'content': 'eD0x'})
        client = self.g.GitHubClient(transport=transport)
        repo = client.resolve_repository({'id': 12, 'full_name': 'a/b', 'default_branch': 'main'})
        files = list(client.fetch_files(repo))
        files = [f for f in files if hasattr(f, 'content')]
        self.assertEqual(repo.commit_sha, 'a' * 40)
        self.assertEqual(next(f for f in files if f.path == 'good.py').content, b'x=1')
        self.assertEqual(next(f for f in files if f.path == 'link.py').kind, 'symlink')
        self.assertEqual(len([u for u in calls if '/git/blobs/' in u]), 1)

    def test_incremental_plan_selects_omitted_paths_before_blob_download(self):
        from quotientguard.survey import Repository, SurveyLimits, AcquisitionPlan
        calls = []
        def transport(url, headers):
            calls.append(url)
            if '/git/trees/' in url:
                return self.response({'tree': [
                    {'path': path, 'type': 'blob', 'mode': '100644', 'size': 3,
                     'sha': hashlib.sha1(b'blob 3\0x=1').hexdigest()}
                    for path in ['trainer/ppo_loss.py', 'trainer/policy.py', 'z.py']]})
            return self.response({'encoding': 'base64', 'content': 'eD0x'})
        client = self.g.GitHubClient(transport=transport, limits=SurveyLimits(max_files=1))
        files = list(client.fetch_files(Repository('1', 'a/b', 'c'*40),
            acquisition_plan=AcquisitionPlan(('trainer/ppo_loss.py',), None, 1)))
        self.assertEqual([f.path for f in files if hasattr(f, 'content')], ['trainer/policy.py'])
        self.assertEqual(len([u for u in calls if '/git/blobs/' in u]), 1)
        self.assertIn('z.py', files[0].uninspected_paths)

    def test_blob_identity_mismatch_is_not_source_evidence(self):
        from quotientguard.survey import Repository, SourceFetchError
        def transport(url, headers):
            if '/git/trees/' in url:
                return self.response({'tree': [{'path': 'x.py', 'type': 'blob', 'mode': '100644', 'size': 3, 'sha': 'a'*40}]})
            return self.response({'encoding': 'base64', 'content': 'eD0x'})
        with self.assertRaisesRegex(SourceFetchError, 'BLOB_IDENTITY_MISMATCH'):
            list(self.g.GitHubClient(transport=transport).fetch_files(Repository('1', 'a/b', 'c'*40)))

    def test_request_budget_and_transient_failure_do_not_loop(self):
        calls = []
        def transport(url, headers):
            calls.append(url)
            return self.response({}, 503)
        report = self.g.GitHubClient(transport=transport).discover('PPO', '2026-01-01', '2026-01-01')
        self.assertEqual(report['failure_count'], 1)
        self.assertEqual(len(calls), 1)
        self.assertFalse(report['coverage_complete'])

    def test_truncated_tree_is_explicit(self):
        from quotientguard.survey import Repository
        client = self.g.GitHubClient(transport=lambda u, h: self.response({'tree': [], 'truncated': True}))
        files = list(client.fetch_files(Repository('1', 'a/b', 'c'*40)))
        self.assertEqual(files[-1].kind, 'TREE_TRUNCATED')

    def test_learning_paths_are_selected_before_alphabetical_utilities(self):
        from quotientguard.survey import Repository, SurveyLimits
        def transport(url, headers):
            if '/git/trees/' in url:
                return self.response({'tree': [
                    {'path': path, 'type': 'blob', 'mode': '100644', 'size': 3,
                     'sha': hashlib.sha1(b'blob 3\0x=1').hexdigest()}
                    for path in ['a_utils.py', 'trainer/grpo_loss.py']], 'truncated': False})
            return self.response({'encoding': 'base64', 'content': 'eD0x'})
        files = list(self.g.GitHubClient(transport=transport, limits=SurveyLimits(max_files=1)).fetch_files(Repository('1', 'a/b', 'c'*40)))
        sources = [f for f in files if getattr(f, 'kind', None) == 'file']
        self.assertEqual(sources[0].path, 'trainer/grpo_loss.py')
        inventory = next(f for f in files if hasattr(f, 'selected_paths'))
        self.assertEqual(inventory.uninspected_paths, ('a_utils.py',))


if __name__ == '__main__':
    unittest.main()
