"""CLI wiring for bounded source-only surveys; acquisition never executes code."""
from dataclasses import asdict
import json
import sqlite3
from pathlib import Path


def add_survey_parsers(sub):
    discover = sub.add_parser('survey-discover', help='bounded public GitHub repository discovery')
    discover.add_argument('--query', required=True)
    discover.add_argument('--start-date', required=True)
    discover.add_argument('--end-date', required=True)
    discover.add_argument('--max-repositories', type=int, default=1000)
    discover.add_argument('--max-requests', type=int, default=50)
    discover.add_argument('--out', type=Path, required=True)
    inventory = sub.add_parser('survey-inventory', help='pin discovered default branches to immutable commits')
    inventory.add_argument('input', type=Path)
    inventory.add_argument('--max-repositories', type=int, default=100)
    inventory.add_argument('--max-requests', type=int, default=100)
    inventory.add_argument('--out', type=Path, required=True)
    screen = sub.add_parser('survey-screen', help='resume bounded AST screening of pinned inventory')
    screen.add_argument('input', type=Path)
    screen.add_argument('--max-repositories', type=int, default=25)
    screen.add_argument('--max-requests', type=int, default=100)
    screen.add_argument('--max-files', type=int, default=100)
    screen.add_argument('--max-file-bytes', type=int, default=256000)
    screen.add_argument('--max-total-bytes', type=int, default=4000000)
    screen.add_argument('--max-result-bytes', type=int, default=4000000,
                        help='per-repository serialized file/candidate buffer budget')
    screen.add_argument('--max-inventory-bytes', type=int, default=4000000,
                        help='per-repository cached and acquired path-inventory budget')
    screen.add_argument('--retry-failed', action='store_true', help='explicitly retry transient acquisition failures once this invocation')
    screen.add_argument('--incremental', action='store_true', help='reuse verified per-file bytes across new immutable run snapshots')
    screen.add_argument('--resume-partial', action='store_true',
                        help='advance partial run handle via immutable child snapshots; max-files grants additional files per step')
    screen.add_argument('--max-store-bytes', type=int, default=100000000,
                        help='cumulative local evidence store quota, including SQLite and objects')
    screen.add_argument('--max-store-objects', type=int, default=100000)
    screen.add_argument('--remaining-path', action='append', help='explicit additional path (repeatable; coverage still includes all known paths)')
    screen.add_argument('--import-store', type=Path, help='read-only previous store to explicitly import verified source bytes')
    screen.add_argument('--import-run-id', help='run in the previous store; requires --import-store')
    screen.add_argument('--out', type=Path)
    export = sub.add_parser('survey-export', help='verify and export source survey evidence')
    export.add_argument('--out', type=Path, required=True)
    compare = sub.add_parser('survey-compare', help='compare repository commits and static candidate identities')
    compare.add_argument('before')
    compare.add_argument('after')
    compare.add_argument('--out', type=Path)
    for parser in (screen, export, compare):
        parser.add_argument('--max-report-bytes', type=int, default=64000000,
                            help='conservative report materialization and input JSON admission budget')
        parser.add_argument('--store', type=Path, required=True,
                            help='local evidence directory; keep internal investigations outside public Git')
    for parser in (screen, export):
        parser.add_argument('--run-id', required=True)


def _write(path, value):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _read_input(path, max_bytes=64000000):
    from .survey import SourceFetchError
    with path.open('rb') as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise SourceFetchError('INPUT_BYTES_LIMIT')
    return json.loads(data)


def run_survey_command(args):
    from .survey import Repository, SurveyStore, SurveyEngine, SurveyLimits, SourceFetchError, compare_runs
    from .survey_github import GitHubClient
    store = None
    try:
        if args.command == 'survey-discover':
            report = GitHubClient(max_requests=args.max_requests).discover(
                args.query, args.start_date, args.end_date, args.max_repositories)
            _write(args.out, report)
            print(json.dumps({k: report[k] for k in ('discovered_count', 'failure_count', 'coverage_complete')}))
            return 1 if report['failure_count'] else 0
        if args.command == 'survey-inventory':
            if args.max_repositories < 0:
                raise ValueError('max-repositories must be nonnegative')
            discovered = _read_input(args.input)
            client = GitHubClient(max_requests=args.max_requests)
            pinned, failures, seen = [], [], set()
            items = discovered['repositories']
            for item in items:
                identity = str(item['id'])
                if identity in seen or len(seen) >= args.max_repositories:
                    continue
                seen.add(identity)
                try:
                    pinned.append(asdict(client.resolve_repository(item)))
                except SourceFetchError as exc:
                    failures.append({'repo_id': identity, 'full_name': item.get('full_name'), 'kind': exc.kind})
                    if exc.kind in ('RATE_LIMIT', 'REQUEST_LIMIT', 'HTTP_403', 'HTTP_401'):
                        break
            report = {'schema': 'quotientguard.survey.inventory.v1', 'repositories': pinned,
                      'failures': failures, 'unresolved_count': len({str(x['id']) for x in items}) - len(pinned)}
            _write(args.out, report)
            print(json.dumps({'pinned': len(pinned), 'failed': len(failures), 'unresolved': report['unresolved_count']}))
            return 1 if failures else 0
        store = SurveyStore(args.store, max_disk_bytes=getattr(args, 'max_store_bytes', None),
                            max_objects=getattr(args, 'max_store_objects', 100000),
                            max_export_bytes=args.max_report_bytes)
        if args.command == 'survey-screen':
            inventory = _read_input(args.input, args.max_report_bytes)
            repos = [Repository(**item) for item in inventory['repositories']]
            limits = SurveyLimits(max_files=args.max_files, max_file_bytes=args.max_file_bytes,
                                  max_total_bytes=args.max_total_bytes, max_result_bytes=args.max_result_bytes,
                                  max_inventory_bytes=args.max_inventory_bytes)
            if bool(args.import_store) != bool(args.import_run_id):
                raise ValueError('import-store and import-run-id must be supplied together')
            if args.import_store:
                store.import_verified_run(args.import_store, args.import_run_id)
            store.create_run(args.run_id)
            report = SurveyEngine(store, limits).run(args.run_id, repos,
                GitHubClient(limits=limits, max_requests=args.max_requests).fetch_files,
                max_repositories=args.max_repositories, retry_failed=args.retry_failed,
                incremental=args.incremental, remaining_paths=args.remaining_path, resume_partial=args.resume_partial)
        elif args.command == 'survey-export':
            report = store.export(args.run_id)
        else:
            report = compare_runs(store, args.before, args.after)
        _write(args.out, report)
        summary = dict(report['counts'], run_id=report['run_id']) if 'counts' in report else report
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return 1 if report.get('counts', {}).get('failed') else 0
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, SourceFetchError) as exc:
        # Do not echo arbitrary HTTP payloads, source bytes, or local environment.
        resource_kinds = {'DISK_QUOTA', 'OBJECT_COUNT_LIMIT', 'RESULT_BYTES_LIMIT',
                          'INVENTORY_BYTES_LIMIT', 'INVENTORY_PATH_LIMIT', 'REPORT_BYTES_LIMIT',
                          'IMPORT_RECORD_BYTES_LIMIT', 'INPUT_BYTES_LIMIT', 'UNSUPPORTED_STORE_MODE'}
        kind = exc.kind if isinstance(exc, SourceFetchError) and exc.kind in resource_kinds else None
        print('Survey failed: ' + type(exc).__name__ + (f' ({kind})' if kind else ''))
        return 2
    finally:
        if store is not None:
            store.close()
