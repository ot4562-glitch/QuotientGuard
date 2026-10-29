"""Bounded, offline static triage with resumable source-bound evidence.

This module never imports or executes inspected source. A candidate identifies
syntax needing human investigation; it is not an objective verdict or a bug.
The caller owns source acquisition and authorization for any later reproduction.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tokenize
import tempfile
from typing import Callable, Iterable


ENGINE_VERSION = 'quotientguard.survey.v1'
ANALYSIS_VERSION = 'quotientguard.survey.analysis.v3'

# SQLite's pager caps sector size at 64 KiB. With cache spilling disabled a
# single-main-database transaction needs one journal segment plus end padding.
# Reserve original pages, 8-byte record overhead, and two maximum-size sectors.
_JOURNAL_PADDING_BYTES = 2 * 65536


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def _hash(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def safe_source_path(path: str) -> bool:
    """Allow only portable relative paths; never interpret source as local paths."""
    return (isinstance(path, str) and bool(path) and len(path) <= 4096
            and '\\' not in path and ':' not in path and '\x00' not in path
            and not path.startswith('/')
            and all(part not in ('', '.', '..') for part in path.split('/')))


@dataclass(frozen=True)
class Repository:
    repo_id: str
    full_name: str
    commit_sha: str
    html_url: str = ''

    def __post_init__(self):
        object.__setattr__(self, 'repo_id', str(self.repo_id))
        if not self.repo_id or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', self.full_name):
            raise ValueError('repository requires a stable ID and owner/name')
        if not re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}', self.commit_sha):
            raise ValueError('repository requires an immutable full commit SHA')
        object.__setattr__(self, 'commit_sha', self.commit_sha.lower())


@dataclass(frozen=True)
class SourceFile:
    path: str
    content: bytes
    kind: str = 'file'


@dataclass(frozen=True)
class SourceInventory:
    """An acquisition provider's bounded tree inventory, without source content."""
    selected_paths: tuple[str, ...]
    uninspected_paths: tuple[str, ...]
    tree_truncated: bool = False
    selection_policy: str = 'Learning/loss/trainer paths first, then lexical order.'


@dataclass(frozen=True)
class AcquisitionPlan:
    """Verified cached paths and bounded additional acquisition for one commit."""
    exclude_paths: tuple[str, ...] = ()
    remaining_paths: tuple[str, ...] | None = None
    max_files: int = 100


@dataclass(frozen=True)
class SurveyLimits:
    max_files: int = 100
    max_file_bytes: int = 256_000
    max_total_bytes: int = 4_000_000
    max_candidates_per_file: int = 50
    max_result_bytes: int = 4_000_000
    max_inventory_bytes: int = 4_000_000

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError('survey limits must be positive integers')


class SourceFetchError(Exception):
    """Structured acquisition failure. Message must not contain credentials."""
    def __init__(self, kind: str, message: str = ''):
        self.kind = kind
        super().__init__(message or kind)


def _bounded_inventory(source: SourceInventory, max_bytes: int):
    """Admit an inventory before copying/serializing its complete path arrays."""
    if len(source.selected_paths) + len(source.uninspected_paths) > 100000:
        raise SourceFetchError('INVENTORY_PATH_LIMIT')
    if not isinstance(source.selection_policy, str) or len(source.selection_policy) > min(max_bytes, 4096):
        raise SourceFetchError('INVENTORY_BYTES_LIMIT')
    size = 256 + len(canonical_json(source.selection_policy))
    if size > max_bytes:
        raise SourceFetchError('INVENTORY_BYTES_LIMIT')
    for paths in (source.selected_paths, source.uninspected_paths):
        for path in paths:
            # Inventory labels are inert remote metadata: Linux names may
            # contain ':' or other characters excluded from acquired sources.
            # Preserve omitted labels without interpreting them as local paths.
            if not isinstance(path, str) or len(path) > 4096:
                raise SourceFetchError('INVENTORY_PATH_LIMIT')
            size += len(canonical_json(path)) + 1
            if size > max_bytes:
                raise SourceFetchError('INVENTORY_BYTES_LIMIT')
    return asdict(source)


class SurveyStore:
    """A local SQLite checkpoint and immutable content-addressed source store.

    Place this directory outside a public checkout: reports contain source and
    triage material. JSON exports intentionally omit wall clocks and secrets.
    """
    def __init__(self, directory, max_disk_bytes=None, max_objects=100000,
                 max_export_bytes=64_000_000):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.objects = self.directory / 'objects'
        if self.objects.is_symlink():
            raise ValueError('evidence directory must not be a link')
        self.objects.mkdir(exist_ok=True)
        # Evidence stores have one writer; network acquisition may be parallel.
        # Account existing objects once instead of rescanning N objects per file.
        self._object_count = 0
        self._object_bytes = 0
        for path in self.objects.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError('source evidence integrity failure')
            self._object_count += 1
            self._object_bytes += path.stat().st_size
        self.max_disk_bytes = max_disk_bytes
        self.max_objects = max_objects
        self.max_export_bytes = max_export_bytes
        self._reserved_disk_bytes = 0
        if (max_disk_bytes is not None and (type(max_disk_bytes) is not int or max_disk_bytes <= 0)
                or type(max_objects) is not int or max_objects <= 0
                or type(max_export_bytes) is not int or max_export_bytes <= 0):
            raise ValueError('positive store resource limits required')
        self.db = sqlite3.connect(self.directory / 'survey.sqlite3')
        try:
            self._initialize_database()
        except sqlite3.OperationalError as exc:
            self.db.close()
            if 'full' in str(exc).lower():
                raise SourceFetchError('DISK_QUOTA') from None
            raise
        except BaseException:
            self.db.close()
            raise

    def _initialize_database(self):
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA cache_spill=OFF')
        self.db.execute('PRAGMA temp_store=MEMORY')
        self.db.execute('PRAGMA mmap_size=0')
        # Enforce admission before schema writes, including a new/partial store.
        self._apply_quota()
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS inventory(
                run_id TEXT NOT NULL REFERENCES runs(id), repo_id TEXT NOT NULL,
                commit_sha TEXT NOT NULL, repository TEXT NOT NULL,
                config TEXT, status TEXT NOT NULL DEFAULT 'PENDING', result TEXT,
                PRIMARY KEY(run_id,repo_id,commit_sha));
            CREATE TABLE IF NOT EXISTS cache(
                repo_id TEXT NOT NULL,commit_sha TEXT NOT NULL,config TEXT NOT NULL,result TEXT NOT NULL,
                PRIMARY KEY(repo_id,commit_sha,config));
            CREATE TABLE IF NOT EXISTS source_index(
                repo_id TEXT NOT NULL, commit_sha TEXT NOT NULL, path TEXT NOT NULL,
                digest TEXT NOT NULL, size INTEGER NOT NULL,
                PRIMARY KEY(repo_id,commit_sha,path));
            CREATE TABLE IF NOT EXISTS analysis_cache(
                repo_id TEXT NOT NULL, commit_sha TEXT NOT NULL, path TEXT NOT NULL,
                digest TEXT NOT NULL, policy TEXT NOT NULL, engine TEXT NOT NULL,
                entry TEXT NOT NULL, candidates TEXT NOT NULL,
                PRIMARY KEY(repo_id,commit_sha,path,digest,policy,engine));
            CREATE TABLE IF NOT EXISTS file_checkpoints(
                run_id TEXT NOT NULL, repo_id TEXT NOT NULL, commit_sha TEXT NOT NULL,
                path TEXT NOT NULL, entry TEXT NOT NULL, candidates TEXT NOT NULL,
                PRIMARY KEY(run_id,repo_id,commit_sha,path));
            CREATE TABLE IF NOT EXISTS source_inventories(
                repo_id TEXT NOT NULL, commit_sha TEXT NOT NULL, inventory TEXT NOT NULL,
                complete INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(repo_id,commit_sha));
            CREATE TABLE IF NOT EXISTS run_configs(
                run_id TEXT PRIMARY KEY, config TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS run_extensions(
                parent_run_id TEXT PRIMARY KEY REFERENCES runs(id),
                child_run_id TEXT NOT NULL UNIQUE REFERENCES runs(id));
        ''')
        self._apply_quota()

    def disk_usage_bytes(self):
        return sum(p.stat().st_size for directory in (self.directory, self.objects)
                   for p in directory.iterdir() if p.is_file())

    def _apply_quota(self, additional=0):
        if self.max_disk_bytes is None:
            return
        page_size = self.db.execute('PRAGMA page_size').fetchone()[0]
        pages = self.db.execute('PRAGMA page_count').fetchone()[0]
        if (self.db.execute('PRAGMA journal_mode').fetchone()[0] != 'delete'
                or self.db.execute('PRAGMA cache_spill').fetchone()[0]):
            raise SourceFetchError('UNSUPPORTED_STORE_MODE')
        other = self._object_bytes + sum(p.stat().st_size for p in self.directory.iterdir()
            if p.is_file() and p.name not in ('survey.sqlite3', 'survey.sqlite3-journal'))
        available = (self.max_disk_bytes - other - additional - self._reserved_disk_bytes
                     - _JOURNAL_PADDING_BYTES)
        # Both the main DB and every original journal page must fit at once.
        # max_page_count cannot shrink below the current DB; fail before writing.
        max_pages = available // (2 * page_size + 8)
        if max_pages < max(pages, 1):
            raise SourceFetchError('DISK_QUOTA')
        self.db.execute('PRAGMA max_page_count=' + str(max_pages))

    def read_source(self, digest, max_bytes=None):
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('source evidence integrity failure')
        path = self.objects / digest
        if path.is_symlink() or not path.is_file():
            raise ValueError('source evidence integrity failure')
        if max_bytes is not None and path.stat().st_size > max_bytes:
            raise SourceFetchError('SIZE_LIMIT')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('source evidence integrity failure')
        return data

    def verify_source(self, digest):
        """Verify skipped cached bytes with constant-size memory."""
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('source evidence integrity failure')
        path = self.objects / digest
        if path.is_symlink() or not path.is_file():
            raise ValueError('source evidence integrity failure')
        actual = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(65536), b''):
                actual.update(chunk)
        if actual.hexdigest() != digest:
            raise ValueError('source evidence integrity failure')

    def import_verified_run(self, previous_directory, run_id):
        """Explicit read-only import of immutable prior bytes, never old analyses.

        Legacy SQLite files are opened in read-only mode; no schema migrations or
        report exports are performed against the previous store.
        """
        previous = Path(previous_directory).resolve()
        if previous == self.directory:
            raise ValueError('import requires a distinct prior store')
        self._apply_quota()
        db_path = previous / 'survey.sqlite3'
        if db_path.is_symlink() or (previous / 'objects').is_symlink():
            raise ValueError('prior evidence directory must not be a link')
        prior = sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)
        count, total = 0, 0
        try:
            if prior.execute('SELECT 1 FROM runs WHERE id=?', (run_id,)).fetchone() is None:
                raise ValueError('unknown prior survey run')
            record_limit = self.max_export_bytes // 2
            for repo_id, commit, size, raw in prior.execute(
                    'SELECT repo_id,commit_sha,length(result),CASE WHEN length(result)<=? THEN result ELSE NULL END '
                    'FROM inventory WHERE run_id=? ORDER BY repo_id,commit_sha', (record_limit, run_id)):
                if size is not None and size > record_limit:
                    raise SourceFetchError('IMPORT_RECORD_BYTES_LIMIT')
                if not raw:
                    continue
                result = json.loads(raw)
                if result.get('evidence_sha256') != _hash({k: v for k, v in result.items() if k != 'evidence_sha256'}):
                    raise ValueError('inspection evidence integrity failure')
                for entry in result['files']:
                    digest = entry.get('source_sha256')
                    if not digest:
                        continue
                    if not re.fullmatch('[0-9a-f]{64}', digest) or not safe_source_path(entry['path']):
                        raise ValueError('source evidence integrity failure')
                    path = previous / 'objects' / digest
                    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16_000_000:
                        raise ValueError('source evidence integrity failure')
                    content = path.read_bytes()
                    if hashlib.sha256(content).hexdigest() != digest:
                        raise ValueError('source evidence integrity failure')
                    self.put_source(content)
                    self.db.execute('INSERT OR IGNORE INTO source_index VALUES (?,?,?,?,?)',
                                    (repo_id, commit, entry['path'], digest, len(content)))
                    self.db.commit()
                    count += 1
                    total += len(content)
                inventory = result.get('source_inventory')
                if inventory is None and result['status'] == 'COMPLETED':
                    inventory = asdict(SourceInventory(tuple(e['path'] for e in result['files']), ()))
                if inventory:
                    self.db.execute('INSERT OR IGNORE INTO source_inventories VALUES (?,?,?,?)',
                        (repo_id, commit, canonical_json(inventory), int(result['status'] == 'COMPLETED')))
                    self.db.commit()
        finally:
            prior.close()
        return {'imported_files': count, 'imported_bytes': total, 'prior_run_id': run_id}

    def close(self):
        self.db.close()

    def create_run(self, run_id: str, metadata=None):
        if not isinstance(run_id, str) or not run_id or len(run_id) > 200:
            raise ValueError('nonempty bounded run ID required')
        payload = canonical_json(metadata or {})
        old = self.db.execute('SELECT metadata FROM runs WHERE id=?', (run_id,)).fetchone()
        if old and old[0] != payload:
            raise ValueError('run metadata is immutable; choose another run ID')
        if old:
            return
        self._apply_quota()
        try:
            self.db.execute('INSERT OR IGNORE INTO runs VALUES (?,?)', (run_id, payload))
            self.db.commit()
        except sqlite3.OperationalError as exc:
            self.db.rollback()
            if 'full' in str(exc).lower():
                raise SourceFetchError('DISK_QUOTA') from None
            raise

    def put_source(self, content: bytes) -> str:
        digest = hashlib.sha256(content).hexdigest()
        path = self.objects / digest
        if path.exists():
            self.verify_source(digest)
            return digest
        if self._object_count >= self.max_objects:
            raise SourceFetchError('OBJECT_COUNT_LIMIT')
        self._apply_quota(len(content))
        # Publish a content-addressed object only after its bytes are durable.
        # A crash while writing leaves an unreferenced pending file, never a
        # corrupted object at the final digest path. Pending files count in quota.
        pending = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.objects, prefix='.pending-', delete=False) as output:
                pending = Path(output.name)
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(pending, path)
            pending = None
            self._object_count += 1
            self._object_bytes += len(content)
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)
        self._apply_quota()
        return digest

    def _verify(self, result):
        for entry in result.get('files', []):
            digest = entry.get('source_sha256')
            if digest:
                self.verify_source(digest)
        expected = result.get('evidence_sha256')
        if expected != _hash({k: v for k, v in result.items() if k != 'evidence_sha256'}):
            raise ValueError('inspection evidence integrity failure')

    def export(self, run_id: str):
        # Bound materialization before loading even the metadata or one raw row.
        # Families duplicate a subset of candidate identities; 2x stored JSON,
        # per-row wrappers and fixed headers form a conservative admission bound.
        size = self.db.execute('SELECT length(metadata) FROM runs WHERE id=?', (run_id,)).fetchone()
        if size is None:
            raise ValueError('unknown survey run')
        stored = self.db.execute('SELECT coalesce(sum(length(repository)+coalesce(length(result),0)+1024),0) '
                                 'FROM inventory WHERE run_id=?', (run_id,)).fetchone()[0]
        if 1024 + 2 * (size[0] + stored) > getattr(self, 'max_export_bytes', 64_000_000):
            raise SourceFetchError('REPORT_BYTES_LIMIT')
        row = self.db.execute('SELECT metadata FROM runs WHERE id=?', (run_id,)).fetchone()
        if row is None:
            raise ValueError('unknown survey run')
        counts = dict(repositories=0, pending=0, completed=0, partial=0, failed=0,
                      candidates=0, inspected_files=0)
        repositories, source_hashes, families = [], set(), {}
        for repository, status, result in self.db.execute(
                'SELECT repository,status,result FROM inventory WHERE run_id=? ORDER BY repo_id,commit_sha', (run_id,)):
            result = json.loads(result) if result else None
            if result:
                self._verify(result)
                counts['candidates'] += len(result['candidates'])
                counts['inspected_files'] += sum(f['status'] == 'INSPECTED' for f in result['files'])
                source_hashes.update(f['source_sha256'] for f in result['files'] if f.get('source_sha256'))
                for candidate in result['candidates']:
                    family_identity = {k: candidate[k] for k in ('source_sha256', 'line', 'pattern')}
                    family_id = _hash(family_identity)
                    family = families.setdefault(family_id, dict(family_identity, family_id=family_id,
                        candidate_ids=[], independent_implementation_confirmed=False))
                    family['candidate_ids'].append(candidate['candidate_id'])
            counts['repositories'] += 1
            counts[status.lower()] += 1
            repositories.append({'repository': json.loads(repository), 'status': status, 'result': result})
        counts['unique_source_hashes'] = len(source_hashes)
        counts['candidate_families'] = len(families)
        for family in families.values():
            family['candidate_ids'].sort()
        report = {'schema': ENGINE_VERSION, 'run_id': run_id, 'metadata': json.loads(row[0]),
                  'classification': 'STATIC_RISK_CANDIDATE', 'counts': counts,
                  'progress': {'processed': counts['repositories'] - counts['pending'], 'total': counts['repositories']},
                  'limitations': ['Python AST patterns only; no dataflow or semantic proof.',
                                  'No inspected source was executed; no bug is confirmed.',
                                  'Bounded file inventory may omit relevant source.'],
                  'repositories': repositories, 'candidate_families': sorted(families.values(), key=lambda f: f['family_id'])}
        report['report_sha256'] = _hash(report)
        return report

    def export_json(self, run_id: str, path=None) -> str:
        output = canonical_json(self.export(run_id)) + '\n'
        if path is not None:
            Path(path).write_text(output, encoding='utf-8')
        return output


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _call_name(node.value) + '.' + node.attr
    return ''


def _has_name(node, fragment):
    return any(fragment in child.id.lower() for child in ast.walk(node) if isinstance(child, ast.Name))


def _patterns(tree):
    """Bounded local dataflow risks; absence is not proof of numerical safety."""
    from .survey_patterns import iter_patterns
    yield from iter_patterns(tree)


def inspect_source(repo: Repository, source: SourceFile, store: SurveyStore, limits: SurveyLimits, *, persist=True):
    entry = {'path': source.path}
    if not safe_source_path(source.path):
        return dict(entry, status='UNSAFE_PATH'), []
    if source.kind != 'file':
        return dict(entry, status='LINK_SKIPPED' if source.kind == 'symlink' else source.kind.upper()), []
    if not source.path.endswith('.py'):
        return dict(entry, status='UNSUPPORTED_LANGUAGE'), []
    if len(source.content) > limits.max_file_bytes:
        return dict(entry, status='SIZE_LIMIT'), []
    digest = store.put_source(source.content) if persist else hashlib.sha256(source.content).hexdigest()
    entry['source_sha256'] = digest
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(source.content).readline)
        text = source.content.decode(encoding)
    except (UnicodeError, SyntaxError, LookupError):
        return dict(entry, status='DECODE_ERROR'), []
    try:
        tree = ast.parse(text, filename=source.path)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        return dict(entry, status='SYNTAX_ERROR', error_type=type(exc).__name__), []
    from .survey_patterns import analyze_tree, POLICY_VERSION
    try:
        analysis = analyze_tree(tree)
    except (ValueError, RecursionError, MemoryError) as exc:
        return dict(entry, status='ANALYSIS_LIMIT', error_type=type(exc).__name__), []
    observation_limit = limits.max_candidates_per_file * 4
    observations = [item for _, item in sorted(analysis.observations.items())]
    entry['analysis_policy'] = POLICY_VERSION
    entry['static_observations'] = observations[:observation_limit]
    entry['observations_truncated'] = len(observations) > observation_limit
    candidates = {}
    for _, finding in sorted(analysis.candidates.items()):
        node, pattern, priority, note = (finding['_node'], finding['pattern'], finding['priority'], finding['reason'])
        identity = {'repo_id': repo.repo_id, 'commit_sha': repo.commit_sha, 'path': source.path,
                    'source_sha256': digest, 'line': node.lineno, 'pattern': pattern, 'engine': ENGINE_VERSION}
        cid = _hash(identity)
        candidates[cid] = dict(identity, candidate_id=cid, classification='STATIC_RISK_CANDIDATE',
            analysis_policy=POLICY_VERSION, operations=finding['operations'], dependencies=finding['dependencies'],
            priority=priority, priority_basis='Static syntax review order; not defect probability or severity.',
            reason=note, reproduction_plan={'execution_authorized': False,
                'required_evidence': ['Declare the exact objective, masking and reduction contract.',
                    'Independently reconstruct the source-bound production loss and derivatives.',
                    'Obtain explicit authorization before installing or executing upstream code.',
                    'Test extreme finite values, all-masked inputs and normal finite controls.',
                    'Capture actual framework backward twice under the objective evidence gates.'],
                'expected_verdict_until_runtime_evidence': 'INSUFFICIENT_EVIDENCE'})
    ordered = sorted(candidates.values(), key=lambda c: (-c['priority'], c['line'], c['pattern']))
    return dict(entry, status='INSPECTED', candidate_limit_reached=len(ordered) > limits.max_candidates_per_file), ordered[:limits.max_candidates_per_file]


class SurveyEngine:
    def __init__(self, store: SurveyStore, limits: SurveyLimits | None = None):
        from . import survey_patterns
        self.store = store
        self.limits = limits or SurveyLimits()
        self.pattern_policy = {'version': survey_patterns.POLICY_VERSION,
            'implementation_sha256': hashlib.sha256(Path(survey_patterns.__file__).read_bytes()).hexdigest()}
        self.analysis_implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        # ast.parse follows the running interpreter's grammar and optimization.
        # Verified bytes remain portable; a parse result is runtime-specific.
        self.parser_runtime = {'implementation': sys.implementation.name,
            'cache_tag': sys.implementation.cache_tag, 'version': sys.version,
            'optimize': sys.flags.optimize}
        self.config = _hash({'engine': ANALYSIS_VERSION, 'patterns': self.pattern_policy,
            'analysis_implementation_sha256': self.analysis_implementation,
            'parser_runtime': self.parser_runtime, 'limits': asdict(self.limits)})

    def run(self, run_id: str, repositories: Iterable[Repository],
            fetch_files: Callable[[Repository], Iterable[SourceFile]], max_repositories: int | None = None,
            retry_failed: bool = False, *, incremental: bool = False,
            remaining_paths: Iterable[str] | None = None, resume_partial: bool = False):
        if resume_partial or (incremental and retry_failed and self.store.db.execute(
                "SELECT 1 FROM inventory WHERE run_id=? AND status='FAILED' LIMIT 1", (run_id,)).fetchone()):
            if not incremental:
                raise ValueError('resume_partial requires incremental mode')
            return self._resume_partial(run_id, repositories, fetch_files, max_repositories,
                                        retry_failed, remaining_paths)
        if incremental:
            return self._run_incremental(run_id, repositories, fetch_files, max_repositories,
                                         retry_failed, remaining_paths)
        if max_repositories is not None and (type(max_repositories) is not int or max_repositories < 0):
            raise ValueError('max_repositories must be nonnegative')
        db = self.store.db
        if db.execute('SELECT 1 FROM runs WHERE id=?', (run_id,)).fetchone() is None:
            raise ValueError('create the survey run before inspecting')
        configurations = {row[0] for row in db.execute(
            'SELECT DISTINCT config FROM inventory WHERE run_id=? AND config IS NOT NULL', (run_id,))}
        if configurations - {self.config}:
            raise ValueError('run configuration is immutable; choose another run ID')
        self.store._apply_quota()
        for repo in repositories:
            db.execute('INSERT OR IGNORE INTO inventory(run_id,repo_id,commit_sha,repository) VALUES (?,?,?,?)',
                       (run_id, repo.repo_id, repo.commit_sha, canonical_json(asdict(repo))))
        db.commit()
        if retry_failed:
            transient = {'NETWORK_ERROR', 'RATE_LIMIT', 'REQUEST_LIMIT', 'GH_API_FAILURE',
                         'HTTP_429', 'HTTP_500', 'HTTP_502', 'HTTP_503', 'HTTP_504'}
            for repo_id, sha, raw_result in db.execute(
                    "SELECT repo_id,commit_sha,result FROM inventory WHERE run_id=? AND status='FAILED'", (run_id,)).fetchall():
                if (json.loads(raw_result).get('failure') or {}).get('kind') in transient:
                    db.execute("UPDATE inventory SET status='PENDING',result=NULL WHERE run_id=? AND repo_id=? AND commit_sha=?",
                               (run_id, repo_id, sha))
            db.commit()
        processed = 0
        rows = db.execute('SELECT repo_id,commit_sha,repository,config,status FROM inventory WHERE run_id=? ORDER BY repo_id,commit_sha', (run_id,)).fetchall()
        for repo_id, sha, raw_repo, config, status in rows:
            if status != 'PENDING':
                if config != self.config:
                    raise ValueError('run configuration is immutable; choose another run ID')
                continue
            if max_repositories is not None and processed >= max_repositories:
                break
            repo = Repository(**json.loads(raw_repo))
            cached = db.execute('SELECT result FROM cache WHERE repo_id=? AND commit_sha=? AND config=?',
                                (repo_id, sha, self.config)).fetchone()
            reserved = self.store._reserved_disk_bytes
            try:
                if cached:
                    self._reserve_terminal(len(cached[0]), 0, reserved)
                    result = json.loads(cached[0])
                    self.store._verify(result)
                else:
                    result = self._inspect(repo, fetch_files)
            finally:
                self.store._reserved_disk_bytes = reserved
            self.store._apply_quota()
            if not cached and result['status'] == 'COMPLETED':
                db.execute('INSERT OR IGNORE INTO cache VALUES (?,?,?,?)', (repo_id, sha, self.config, canonical_json(result)))
            db.execute('UPDATE inventory SET status=?,config=?,result=? WHERE run_id=? AND repo_id=? AND commit_sha=?',
                       (result['status'], self.config, canonical_json(result), run_id, repo_id, sha))
            db.commit()  # One durable checkpoint per repository; interruption leaves remaining inventory pending.
            processed += 1
            if (result.get('failure') or {}).get('kind') in ('RATE_LIMIT', 'REQUEST_LIMIT', 'HTTP_401', 'HTTP_403'):
                break
        return self.store.export(run_id)

    def _resume_partial(self, run_id, repositories, fetch_files, max_repositories, retry_failed, paths):
        """Advance an explicit handle through immutable, per-step snapshots."""
        db = self.store.db
        paths = None if paths is None else tuple(sorted(set(paths)))
        latest = run_id
        for _ in range(100000):
            row = db.execute('SELECT child_run_id FROM run_extensions WHERE parent_run_id=?', (latest,)).fetchone()
            if row is None:
                break
            latest = row[0]
        else:
            raise ValueError('snapshot extension chain exceeds limit')
        report = self.store.export(latest)
        if report['counts']['pending'] or not report['counts']['repositories']:
            return self._run_incremental(latest, repositories, fetch_files, max_repositories, retry_failed, paths)
        extendable = any(item['status'] == 'PARTIAL' and (
            not item['result'].get('coverage') or item['result']['coverage'].get('uninspected_paths')
            or item['result']['coverage'].get('tree_truncated') or item['result']['coverage'].get('known_paths') is None)
            or (retry_failed and item['status'] == 'FAILED' and (item['result'].get('failure') or {}).get('kind') in
                {'NETWORK_ERROR', 'RATE_LIMIT', 'REQUEST_LIMIT', 'HTTP_429', 'HTTP_500', 'HTTP_502', 'HTTP_503', 'HTTP_504'})
            for item in report['repositories'])
        if not extendable:
            return report
        if paths is None:
            for item in report['repositories']:
                old = item.get('result') or {}
                key = {'RESULT_BYTES_LIMIT': 'max_result_bytes',
                       'INVENTORY_BYTES_LIMIT': 'max_inventory_bytes'}.get(old.get('limit_reason'))
                if key and getattr(self.limits, key) <= old.get('limits', {}).get(key, 0):
                    # Same-budget retries cannot advance the bounded buffer.
                    raise SourceFetchError(old['limit_reason'])
        child = latest[:180] + '~' + _hash({'parent': report['report_sha256'], 'config': self.config,
                                           'paths': sorted(paths) if paths is not None else None})[:12]
        self.store.create_run(child, {'extends_run_id': latest, 'parent_report_sha256': report['report_sha256'],
                                     'additional_files_per_repository': self.limits.max_files})
        db.execute('INSERT OR IGNORE INTO inventory(run_id,repo_id,commit_sha,repository) '
                   'SELECT ?,repo_id,commit_sha,repository FROM inventory WHERE run_id=?', (child, latest))
        db.execute('INSERT INTO run_extensions VALUES (?,?)', (latest, child))
        db.commit()
        return self._run_incremental(child, repositories, fetch_files, max_repositories, retry_failed, paths)

    def _run_incremental(self, run_id, repositories, fetch_files, max_repositories,
                         retry_failed, remaining_paths):
        if max_repositories is not None and (type(max_repositories) is not int or max_repositories < 0):
            raise ValueError('max_repositories must be nonnegative')
        paths = None if remaining_paths is None else tuple(sorted(set(remaining_paths)))
        if paths is not None and (len(paths) > 100000 or any(not safe_source_path(p) for p in paths)):
            raise ValueError('bounded safe remaining paths required')
        db = self.store.db
        if db.execute('SELECT 1 FROM runs WHERE id=?', (run_id,)).fetchone() is None:
            raise ValueError('create the survey run before inspecting')
        # File-count/byte/request budgets control acquisition, not analysis identity.
        policy = _hash({'patterns': self.pattern_policy,
                        'analysis_implementation_sha256': self.analysis_implementation,
                        'parser_runtime': self.parser_runtime,
                        'max_candidates_per_file': self.limits.max_candidates_per_file})
        config = _hash({'engine': ANALYSIS_VERSION, 'policy': policy,
                        'limits': asdict(self.limits), 'remaining_paths': paths, 'incremental': True})
        old_config = db.execute('SELECT config FROM run_configs WHERE run_id=?', (run_id,)).fetchone()
        old_inventory = db.execute('SELECT 1 FROM inventory WHERE run_id=? AND config IS NOT NULL LIMIT 1', (run_id,)).fetchone()
        if (old_config and old_config[0] != config) or (not old_config and old_inventory):
            raise ValueError('run configuration is immutable; choose another run ID')
        self.store._apply_quota()
        db.execute('INSERT OR IGNORE INTO run_configs VALUES (?,?)', (run_id, config))
        for repo in repositories:
            if old_config and db.execute('SELECT 1 FROM inventory WHERE run_id=? AND repo_id=? AND commit_sha=?',
                                        (run_id, repo.repo_id, repo.commit_sha)).fetchone() is None:
                raise ValueError('run inventory is immutable; choose another run ID')
            db.execute('INSERT OR IGNORE INTO inventory(run_id,repo_id,commit_sha,repository) VALUES (?,?,?,?)',
                (run_id, repo.repo_id, repo.commit_sha, canonical_json(asdict(repo))))
        db.commit()
        processed = 0
        transient = {'NETWORK_ERROR', 'RATE_LIMIT', 'REQUEST_LIMIT', 'HTTP_429',
                     'HTTP_500', 'HTTP_502', 'HTTP_503', 'HTTP_504'}
        # Cursor iteration keeps repository inventory out of Python memory.
        for repo_id, sha, raw_repo, status, raw_result in db.execute(
                'SELECT repo_id,commit_sha,repository,status,result FROM inventory WHERE run_id=? ORDER BY repo_id,commit_sha', (run_id,)):
            if status != 'PENDING':
                failure = (json.loads(raw_result).get('failure') or {}) if raw_result else {}
                if not (retry_failed and status == 'FAILED' and failure.get('kind') in transient):
                    continue
            if max_repositories is not None and processed >= max_repositories:
                break
            repo = Repository(**json.loads(raw_repo))
            db.execute('UPDATE inventory SET config=? WHERE run_id=? AND repo_id=? AND commit_sha=?',
                       (config, run_id, repo_id, sha))
            db.commit()
            reserved = self.store._reserved_disk_bytes
            try:
                result = self._inspect_incremental(run_id, repo, fetch_files, policy, paths)
            finally:
                self.store._reserved_disk_bytes = reserved
            # The file phase reserved pages for this terminal immutable snapshot.
            self.store._apply_quota()
            db.execute('UPDATE inventory SET status=?,result=? WHERE run_id=? AND repo_id=? AND commit_sha=?',
                (result['status'], canonical_json(result), run_id, repo_id, sha))
            db.commit()
            processed += 1
            if (result.get('failure') or {}).get('kind') in ('RATE_LIMIT', 'REQUEST_LIMIT', 'HTTP_401', 'HTTP_403', 'DISK_QUOTA'):
                break
        return self.store.export(run_id)

    def _reserve_terminal(self, buffered_bytes, inventory_bytes, base=0, copies=1):
        """Keep terminal result space reserved across object publication/SQL."""
        old = self.store._reserved_disk_bytes
        page_size = self.store.db.execute('PRAGMA page_size').fetchone()[0]
        self.store._reserved_disk_bytes = base + 2 * copies * (8192 + buffered_bytes + 2 * inventory_bytes) + 16 * page_size
        try:
            self.store._apply_quota()
        except BaseException:
            self.store._reserved_disk_bytes = old
            raise

    def _inspect_incremental(self, run_id, repo, fetch_files, policy, paths):
        db = self.store.db
        result = {'engine': ENGINE_VERSION, 'analysis_engine': ANALYSIS_VERSION,
                  'analysis_policy': policy, 'pattern_policy': self.pattern_policy,
                  'parser_runtime': dict(self.parser_runtime), 'limits': asdict(self.limits),
                  'status': 'COMPLETED', 'files': [], 'candidates': [], 'failure': None}
        completed, total, acquired, reused, acquired_bytes = set(), 0, 0, 0, 0
        buffered_bytes = 4  # Empty file/candidate arrays.
        cached_paths = set()
        inventory_row = db.execute('SELECT length(inventory),complete FROM source_inventories WHERE repo_id=? AND commit_sha=?',
                                   (repo.repo_id, repo.commit_sha)).fetchone()
        inventory = None
        inventory_bytes = 0
        base_reserve = self.store._reserved_disk_bytes
        self._reserve_terminal(buffered_bytes, inventory_bytes, base_reserve)

        def checkpoint(source, cached_source=False):
            nonlocal total, acquired_bytes, buffered_bytes
            if source.path in completed:
                return
            if len(source.content) > self.limits.max_file_bytes:
                entry, candidates = {'path': source.path, 'status': 'SIZE_LIMIT'}, []
            elif not cached_source and acquired_bytes + len(source.content) > self.limits.max_total_bytes:
                result['limit_reason'] = 'TOTAL_BYTES_LIMIT'
                return False
            else:
                digest = hashlib.sha256(source.content).hexdigest()
                cached = db.execute('SELECT entry,candidates FROM analysis_cache WHERE repo_id=? AND commit_sha=? AND path=? AND digest=? AND policy=? AND engine=?',
                    (repo.repo_id, repo.commit_sha, source.path, digest, policy, ANALYSIS_VERSION)).fetchone()
                if cached:
                    self.store.read_source(digest, self.limits.max_file_bytes)
                    entry, candidates = json.loads(cached[0]), json.loads(cached[1])
                else:
                    entry, candidates = inspect_source(repo, source, self.store, self.limits, persist=False)
                if entry.get('source_sha256'):
                    # Keep room for index-page growth, the immutable repository
                    # snapshot, and a structured terminal quota diagnostic.
                    record_bytes = len(canonical_json(entry)) + len(canonical_json(candidates))
                    next_buffered = buffered_bytes + record_bytes + 2
                    self._reserve_terminal(next_buffered if next_buffered <= self.limits.max_result_bytes else buffered_bytes,
                                           inventory_bytes, base_reserve)
                    object_bytes = 0 if (self.store.objects / digest).exists() else len(source.content)
                    self.store._apply_quota(object_bytes + 16384 + 8 * record_bytes)
                    self.store.put_source(source.content)
                    if not cached:
                        db.execute('INSERT OR IGNORE INTO analysis_cache VALUES (?,?,?,?,?,?,?,?)',
                            (repo.repo_id, repo.commit_sha, source.path, digest, policy, ANALYSIS_VERSION,
                             canonical_json(entry), canonical_json(candidates)))
                    db.execute('INSERT OR IGNORE INTO source_index VALUES (?,?,?,?,?)',
                        (repo.repo_id, repo.commit_sha, source.path, digest, len(source.content)))
                total += len(source.content)
                if not cached_source:
                    acquired_bytes += len(source.content)
            record_bytes = len(canonical_json(entry)) + len(canonical_json(candidates)) + 2
            if buffered_bytes + record_bytes > self.limits.max_result_bytes:
                # Keep the verified source/index/analysis for a larger-budget or
                # scoped run, without growing this report or downloading again.
                db.commit()
                result['limit_reason'] = 'RESULT_BYTES_LIMIT'
                return False
            self._reserve_terminal(buffered_bytes + record_bytes, inventory_bytes, base_reserve)
            db.execute('INSERT OR REPLACE INTO file_checkpoints VALUES (?,?,?,?,?,?)',
                (run_id, repo.repo_id, repo.commit_sha, source.path, canonical_json(entry), canonical_json(candidates)))
            db.commit()  # Must finish before advancing the source generator / next network request.
            result['files'].append(entry)
            result['candidates'].extend(candidates)
            completed.add(source.path)
            buffered_bytes += record_bytes
            return True

        try:
            if inventory_row:
                if inventory_row[0] > self.limits.max_inventory_bytes:
                    raise SourceFetchError('INVENTORY_BYTES_LIMIT')
                self._reserve_terminal(buffered_bytes, inventory_row[0], base_reserve)
                inventory = json.loads(db.execute('SELECT inventory FROM source_inventories WHERE repo_id=? AND commit_sha=?',
                                       (repo.repo_id, repo.commit_sha)).fetchone()[0])
                inventory_bytes = inventory_row[0]
            # Stored bytes are verified and reanalysed under the current analysis policy.
            # This includes durable files from interrupted runs and explicit imports.
            for path, digest, size in db.execute('SELECT path,digest,size FROM source_index WHERE repo_id=? AND commit_sha=? ORDER BY path',
                                                (repo.repo_id, repo.commit_sha)):
                # Verify every excluded mapping, including out-of-scope and
                # over-budget cached paths, before asking acquisition to skip it.
                if (result.get('limit_reason') or size > self.limits.max_file_bytes
                        or (paths is not None and path not in paths)):
                    self.store.verify_source(digest)
                    cached_paths.add(path)
                    continue
                content = self.store.read_source(digest, self.limits.max_file_bytes)
                cached_paths.add(path)
                if checkpoint(SourceFile(path, content), cached_source=True) is False:
                    continue
                reused += 1
            known = set(inventory.get('selected_paths', ())) | set(inventory.get('uninspected_paths', ())) if inventory else set()
            wanted = set(paths) if paths is not None else known
            need_fetch = ((not inventory or inventory.get('tree_truncated') or wanted - completed)
                          and (not inventory or inventory.get('tree_truncated') or wanted - cached_paths))
            if paths is not None:
                need_fetch = bool(set(paths) - cached_paths)
            if result.get('limit_reason'):
                need_fetch = False
            if need_fetch:
                plan = AcquisitionPlan(tuple(sorted(cached_paths)), paths, self.limits.max_files)
                try:
                    supports_plan = 'acquisition_plan' in inspect.signature(fetch_files).parameters
                except (ValueError, TypeError):
                    supports_plan = False
                sources = fetch_files(repo, acquisition_plan=plan) if supports_plan else fetch_files(repo)
                iterator = iter(sources)
                try:
                    while acquired < self.limits.max_files:
                        try:
                            source = next(iterator)
                        except StopIteration:
                            break
                        if isinstance(source, SourceInventory):
                            new_inventory = _bounded_inventory(source, self.limits.max_inventory_bytes)
                            encoded = canonical_json(new_inventory)
                            self._reserve_terminal(buffered_bytes, len(encoded), base_reserve)
                            self.store._apply_quota(16384 + 3 * len(encoded))
                            inventory = new_inventory
                            inventory_bytes = len(encoded)
                            db.execute('INSERT OR REPLACE INTO source_inventories VALUES (?,?,?,0)',
                                       (repo.repo_id, repo.commit_sha, encoded))
                            db.commit()
                            continue
                        if source.path in cached_paths or source.path in completed or (paths is not None and source.path not in paths):
                            continue
                        if checkpoint(source) is False:
                            break
                        acquired += 1
                finally:
                    close = getattr(iterator, 'close', None)
                    if close:
                        close()
            if result.get('limit_reason') or any(f['status'] != 'INSPECTED' or f.get('candidate_limit_reached') for f in result['files']):
                result['status'] = 'PARTIAL'
        except SourceFetchError as exc:
            result['status'] = 'FAILED'
            result['failure'] = {'kind': exc.kind}
            if exc.kind in ('INVENTORY_BYTES_LIMIT', 'INVENTORY_PATH_LIMIT'):
                result['status'] = 'PARTIAL'
                result['limit_reason'] = exc.kind
        except (OSError, TimeoutError):
            result['status'] = 'FAILED'
            result['failure'] = {'kind': 'NETWORK_ERROR'}
        except sqlite3.OperationalError as exc:
            if 'full' not in str(exc).lower():
                raise
            db.rollback()
            result['status'] = 'FAILED'
            result['failure'] = {'kind': 'DISK_QUOTA'}
        # Even failed acquisitions retain the known inventory and exact omissions.
        known = set(inventory['selected_paths']) | set(inventory['uninspected_paths']) if inventory else set()
        missing = sorted(known - completed)
        if inventory:
            result['source_inventory'] = inventory
        result['coverage'] = {'known_paths': len(known) if inventory else None,
            'inspected_paths': sum(f['status'] == 'INSPECTED' for f in result['files']),
            'uninspected_paths': missing, 'tree_truncated': bool(inventory.get('tree_truncated')) if inventory else None,
            'complete': bool(inventory) and not missing and not inventory.get('tree_truncated')
                and all(f['status'] == 'INSPECTED' and not f.get('candidate_limit_reached') for f in result['files'])}
        if not result['coverage']['complete'] and result['status'] == 'COMPLETED':
            result['status'] = 'PARTIAL'
        result['resources'] = {'reused_files': reused, 'acquired_files': acquired, 'source_bytes': total,
                               'acquired_source_bytes': acquired_bytes}
        result['files'].sort(key=lambda f: f['path'])
        result['candidates'].sort(key=lambda c: (-c['priority'], c['candidate_id']))
        result['evidence_sha256'] = _hash(result)
        return result

    def _inspect(self, repo, fetch_files):
        result = {'engine': ENGINE_VERSION, 'limits': asdict(self.limits), 'status': 'COMPLETED',
                  'analysis_engine': ANALYSIS_VERSION, 'parser_runtime': dict(self.parser_runtime),
                  'files': [], 'candidates': [], 'failure': None}
        seen, total, buffered_bytes = set(), 0, 4
        inventory_bytes = 0
        base_reserve = self.store._reserved_disk_bytes
        self._reserve_terminal(buffered_bytes, inventory_bytes, base_reserve, copies=2)
        try:
            for source in fetch_files(repo):
                if isinstance(source, SourceInventory):
                    inventory = _bounded_inventory(source, self.limits.max_inventory_bytes)
                    encoded_size = len(canonical_json(inventory))
                    self._reserve_terminal(buffered_bytes, encoded_size, base_reserve, copies=2)
                    result['source_inventory'] = inventory
                    inventory_bytes = encoded_size
                    if source.uninspected_paths or source.tree_truncated:
                        result['status'] = 'PARTIAL'
                    continue
                if source.path in seen:
                    continue
                seen.add(source.path)
                if len(result['files']) >= self.limits.max_files:
                    result['status'] = 'PARTIAL'
                    result['limit_reason'] = 'FILE_COUNT_LIMIT'
                    break
                total += len(source.content)
                if total > self.limits.max_total_bytes:
                    result['status'] = 'PARTIAL'
                    result['limit_reason'] = 'TOTAL_BYTES_LIMIT'
                    break
                entry, candidates = inspect_source(repo, source, self.store, self.limits, persist=False)
                record_bytes = len(canonical_json(entry)) + len(canonical_json(candidates)) + 2
                if buffered_bytes + record_bytes > self.limits.max_result_bytes:
                    result['status'] = 'PARTIAL'
                    result['limit_reason'] = 'RESULT_BYTES_LIMIT'
                    break
                self._reserve_terminal(buffered_bytes + record_bytes, inventory_bytes, base_reserve, copies=2)
                if entry.get('source_sha256'):
                    self.store.put_source(source.content)
                result['files'].append(entry)
                result['candidates'].extend(candidates)
                buffered_bytes += record_bytes
                if entry['status'] != 'INSPECTED' or entry.get('candidate_limit_reached'):
                    result['status'] = 'PARTIAL'
        except SourceFetchError as exc:
            result['status'] = 'FAILED'
            result['failure'] = {'kind': exc.kind}  # Never persist exception text or authorization headers.
        except (OSError, TimeoutError) as exc:
            result['status'] = 'FAILED'
            result['failure'] = {'kind': 'NETWORK_ERROR', 'error_type': type(exc).__name__}
        result['files'].sort(key=lambda f: f['path'])
        result['candidates'].sort(key=lambda c: (-c['priority'], c['candidate_id']))
        result['evidence_sha256'] = _hash(result)
        return result


def compare_runs(store: SurveyStore, before: str, after: str):
    old, new = store.export(before), store.export(after)
    def identities(report):
        repos = {}
        candidates = set()
        for item in report['repositories']:
            repo = item['repository']
            repos.setdefault(repo['repo_id'], set()).add(repo['commit_sha'])
            if item['result']:
                candidates.update(c['candidate_id'] for c in item['result']['candidates'])
        return repos, candidates
    a, ac = identities(old)
    b, bc = identities(new)
    result = {'schema': 'quotientguard.survey.compare.v1', 'before': before, 'after': after,
              'before_report_sha256': old['report_sha256'], 'after_report_sha256': new['report_sha256'],
              'added_repository_ids': sorted(b.keys() - a.keys()), 'removed_repository_ids': sorted(a.keys() - b.keys()),
              'changed_commits': sorted(k for k in a.keys() & b.keys() if a[k] != b[k]),
              'added_candidate_ids': sorted(bc - ac), 'removed_candidate_ids': sorted(ac - bc),
              'before_counts': old['counts'], 'after_counts': new['counts'],
              'interpretation': 'Static candidate differences may reflect source, coverage or configuration changes; no fix or regression is confirmed.'}
    result['comparison_sha256'] = _hash(result)
    return result
