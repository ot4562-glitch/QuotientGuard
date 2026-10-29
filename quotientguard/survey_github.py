"""Bounded GitHub public-source discovery and acquisition, without execution.

Transport injection permits offline tests. The default transport uses only the
GitHub API, never clones repositories and never records a token or response body
on failures. Search counts describe query results, not the population of RL repos.
"""
from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
import json
import math
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .survey import Repository, SourceFetchError, SourceFile, SourceInventory, SurveyLimits, safe_source_path


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    headers: dict
    body: bytes


class GitHubClient:
    def __init__(self, token=None, transport=None, limits=None, max_requests=500,
                 max_response_bytes=8_000_000, timeout=20):
        self._token = token
        self.transport = transport or self._transport
        self.limits = limits or SurveyLimits()
        self.max_requests = max_requests
        self.max_response_bytes = max_response_bytes
        self.timeout = timeout
        self.request_count = 0
        if max_requests <= 0 or max_response_bytes <= 0 or timeout <= 0:
            raise ValueError('HTTP limits must be positive')

    def _transport(self, url, headers):
        try:
            with urlopen(Request(url, headers=headers), timeout=self.timeout) as response:
                return HTTPResponse(response.status, dict(response.headers), response.read(self.max_response_bytes + 1))
        except HTTPError as exc:
            # No response body is needed to distinguish acquisition failures.
            return HTTPResponse(exc.code, dict(exc.headers), b'')
        except (OSError, URLError, TimeoutError):
            raise SourceFetchError('NETWORK_ERROR') from None

    def _request(self, path, params=None):
        url = 'https://api.github.com' + path
        if params:
            url += '?' + urlencode(params)
        headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'QuotientGuard-survey',
                   'X-GitHub-Api-Version': '2022-11-28'}
        if self._token:
            headers['Authorization'] = 'Bearer ' + self._token
        # No automatic retries. A resumed invocation can retry after owner review.
        if self.request_count >= self.max_requests:
            raise SourceFetchError('REQUEST_LIMIT')
        self.request_count += 1
        response = self.transport(url, headers)
        response_headers = {k.lower(): v for k, v in response.headers.items()}
        if response.status in (403, 429) and (str(response_headers.get('x-ratelimit-remaining')) == '0'
                                              or response.status == 429 or 'retry-after' in response_headers):
            raise SourceFetchError('RATE_LIMIT')
        if response.status != 200:
            raise SourceFetchError('HTTP_' + str(response.status))
        if len(response.body) > self.max_response_bytes:
            raise SourceFetchError('RESPONSE_SIZE_LIMIT')
        try:
            result = json.loads(response.body)
        except (ValueError, UnicodeError):
            raise SourceFetchError('INVALID_JSON') from None
        if not isinstance(result, dict):
            raise SourceFetchError('INVALID_RESPONSE')
        return result

    def discover(self, query: str, start_date: str, end_date: str, max_repositories=1000):
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        if start > end or type(max_repositories) is not int or max_repositories <= 0:
            raise ValueError('ordered dates and positive discovery limit required')
        if not query.strip() or len(query) > 500 or re.search(r'\bcreated:', query, re.I):
            raise ValueError('provide a bounded query without a created qualifier')
        found, partitions, failures = {}, [], []
        pending = [(start, end)]
        coverage_complete = True
        while pending and len(found) < max_repositories:
            lo, hi = pending.pop()
            q = query + f' created:{lo.isoformat()}..{hi.isoformat()} is:public'
            partition = {'start_date': lo.isoformat(), 'end_date': hi.isoformat(), 'query': q, 'retrieved': 0}
            try:
                first = self._request('/search/repositories', {'q': q, 'per_page': 100, 'page': 1, 'sort': 'created', 'order': 'asc'})
                count = first.get('total_count')
                if type(count) is not int or count < 0 or not isinstance(first.get('items'), list):
                    raise SourceFetchError('INVALID_RESPONSE')
                if count > 1000 and lo < hi:
                    mid = lo + timedelta(days=(hi - lo).days // 2)
                    pending.extend([(mid + timedelta(days=1), hi), (lo, mid)])
                    continue
                partition.update(total_count=count, status='SEARCH_CAP' if count > 1000 else 'COMPLETE')
                if count > 1000:
                    coverage_complete = False
                pages = min(10, max(1, math.ceil(count / 100)))
                for page in range(1, pages + 1):
                    response = first if page == 1 else self._request('/search/repositories', {
                        'q': q, 'per_page': 100, 'page': page, 'sort': 'created', 'order': 'asc'})
                    items = response.get('items')
                    if not isinstance(items, list):
                        raise SourceFetchError('INVALID_RESPONSE')
                    if response.get('incomplete_results'):
                        partition['status'] = 'INCOMPLETE_RESULTS'
                        coverage_complete = False
                    for item in items[:100]:
                        if not isinstance(item, dict) or item.get('id') is None or not item.get('full_name'):
                            continue
                        if str(item['id']) not in found:
                            found[str(item['id'])] = {k: item[k] for k in (
                                'id', 'full_name', 'default_branch', 'html_url', 'description',
                                'stargazers_count', 'fork', 'archived', 'created_at', 'pushed_at', 'language') if k in item}
                        partition['retrieved'] += 1
                        if len(found) >= max_repositories:
                            if page < pages or partition['retrieved'] < min(count, 1000):
                                partition['status'] = 'DISCOVERY_LIMIT'
                                coverage_complete = False
                            break
                    if len(found) >= max_repositories:
                        break
                    if len(items) < 100 and page < pages:
                        partition['status'] = 'SHORT_PAGE'
                        coverage_complete = False
                        break
                if partition['retrieved'] < min(count, 1000) and partition['status'] == 'COMPLETE':
                    partition['status'] = 'SHORT_PAGE'
                    coverage_complete = False
                partitions.append(partition)
            except SourceFetchError as exc:
                coverage_complete = False
                partition['status'] = 'FAILED'
                partitions.append(partition)
                failures.append({'query': q, 'kind': exc.kind})
                if exc.kind in ('RATE_LIMIT', 'REQUEST_LIMIT', 'HTTP_403', 'HTTP_401'):
                    break
        if pending:
            coverage_complete = False
        return {'schema': 'quotientguard.survey.discovery.v1', 'query': query,
                'start_date': start_date, 'end_date': end_date, 'repositories': sorted(found.values(), key=lambda r: str(r['id'])),
                'discovered_count': len(found), 'partitions': partitions, 'failure_count': len(failures),
                'failures': failures, 'coverage_complete': coverage_complete,
                'unqueried_partitions': len(pending), 'request_count': self.request_count,
                'limitations': ['Search query hits are not a complete RL/RLVR repository population.',
                                'GitHub search caps each date partition at 1000 results.']}

    def resolve_repository(self, item):
        full_name = item.get('full_name', '')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', full_name) or item.get('id') is None:
            raise ValueError('GitHub repository identity required')
        branch = item.get('default_branch', 'HEAD')
        response = self._request('/repos/' + full_name + '/commits/' + quote(branch, safe=''))
        try:
            return Repository(str(item['id']), full_name, response['sha'], item.get('html_url', ''))
        except (ValueError, KeyError, TypeError):
            raise SourceFetchError('INVALID_COMMIT') from None

    def fetch_files(self, repo: Repository, acquisition_plan=None):
        tree = self._request('/repos/' + repo.full_name + '/git/trees/' + repo.commit_sha, {'recursive': '1'})
        entries = tree.get('tree')
        if not isinstance(entries, list):
            raise SourceFetchError('INVALID_TREE')
        if any(not isinstance(item, dict) for item in entries):
            raise SourceFetchError('INVALID_TREE')
        if len(entries) > 100000:
            raise SourceFetchError('INVENTORY_PATH_LIMIT')
        entries = [item for item in entries if isinstance(item.get('path'), str)
                   and item['path'].endswith('.py') and item.get('type') == 'blob']
        def priority(item):
            path = item['path'].lower()
            terms = ('loss', 'ppo', 'grpo', 'objective', 'trainer', 'policy', 'actor', 'advantage')
            score = sum((len(terms) - i) * 10 for i, term in enumerate(terms) if term in path)
            if any(part in path.split('/') for part in ('test', 'tests', 'examples', 'docs', 'vendor')):
                score -= 200
            return (-score, path)
        entries.sort(key=priority)
        exclude = set(acquisition_plan.exclude_paths) if acquisition_plan else set()
        remaining = (set(acquisition_plan.remaining_paths)
                     if acquisition_plan and acquisition_plan.remaining_paths is not None else None)
        eligible = [i for i in entries if i['path'] not in exclude
                    and (remaining is None or i['path'] in remaining)]
        budget = min(self.limits.max_files, acquisition_plan.max_files) if acquisition_plan else self.limits.max_files
        selected = eligible[:budget]
        selected_names = {i['path'] for i in selected}
        yield SourceInventory(tuple(i['path'] for i in selected),
                              tuple(i['path'] for i in entries if i['path'] not in selected_names), bool(tree.get('truncated')))
        count, total = 0, 0
        for item in selected:
            if not isinstance(item, dict):
                raise SourceFetchError('INVALID_TREE')
            path = item.get('path', '')
            if not isinstance(path, str) or not path.endswith('.py') or item.get('type') != 'blob':
                continue
            if count >= self.limits.max_files:
                yield SourceFile('__file_limit__.py', b'', 'FILE_COUNT_LIMIT')
                return
            count += 1
            if not safe_source_path(path):
                yield SourceFile(path, b'')
                continue
            if item.get('mode') != '100644' and item.get('mode') != '100755':
                yield SourceFile(path, b'', 'symlink')
                continue
            size = item.get('size')
            if type(size) is not int or size < 0:
                raise SourceFetchError('INVALID_TREE')
            if size > self.limits.max_file_bytes:
                yield SourceFile(path, b'', 'SIZE_LIMIT')
                continue
            total += size
            if total > self.limits.max_total_bytes:
                yield SourceFile(path, b'', 'TOTAL_BYTES_LIMIT')
                return
            sha = item.get('sha', '')
            if not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', sha):
                raise SourceFetchError('INVALID_TREE')
            blob = self._request('/repos/' + repo.full_name + '/git/blobs/' + sha)
            if blob.get('encoding') != 'base64' or not isinstance(blob.get('content'), str):
                raise SourceFetchError('INVALID_BLOB')
            try:
                content = base64.b64decode(''.join(blob['content'].split()), validate=True)
            except ValueError:
                raise SourceFetchError('INVALID_BLOB') from None
            if len(content) > self.limits.max_file_bytes:
                yield SourceFile(path, b'', 'SIZE_LIMIT')
                continue
            git_content = b'blob ' + str(len(content)).encode('ascii') + b'\0' + content
            actual_sha = (hashlib.sha1(git_content).hexdigest() if len(sha) == 40
                          else hashlib.sha256(git_content).hexdigest())
            if actual_sha != sha or len(content) != size:
                raise SourceFetchError('BLOB_IDENTITY_MISMATCH')
            yield SourceFile(path, content)
        if tree.get('truncated'):
            yield SourceFile('__truncated_tree__.py', b'', 'TREE_TRUNCATED')
