"""Command-line interface for QuotientGuard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .adapters.trl_sequence import trl_sequence_batch_to_receipt
from .benchmark import benchmark_sampled_audit
from .certification import certification_text, run_builtin_certification_suite, run_certification_suite
from .check import check_path
from .compare import compare_certificates
from .core import audit_receipt
from .demo import run_paper_a1_demo
from .explain import explain_trl_payload, explanation_text
from .observations import observations_to_receipt, read_jsonl
from .report import comparison_text, markdown_report, text_report
from .sentinel import sentinel_markdown, summarize_certificates
from .source_manifests import SOURCE_MANIFESTS, verify_source_file
from .version import __version__


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _certificate_outputs(args: argparse.Namespace, certificate: dict) -> None:
    if getattr(args, "out", None):
        _write_json(args.out, certificate)
    if getattr(args, "markdown", None):
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(certificate), encoding="utf-8")
    if getattr(args, "json", False):
        print(json.dumps(certificate, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(text_report(certificate))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quotientguard",
        description="Audit estimator homomorphism compatibility for RL/RLVR training.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "check",
        help="auto-detect a receipt, sampled observation log, or supported trainer fixture",
    )
    check.add_argument("input", type=Path)
    check.add_argument("--out", type=Path, help="write the JSON certificate")
    check.add_argument("--markdown", type=Path, help="write a Markdown report")
    check.add_argument("--json", action="store_true", help="print full JSON instead of concise text")

    demo = sub.add_parser(
        "demo",
        help="run the paper A1 equal-length clipping demonstration",
    )
    demo.add_argument("--out", type=Path, help="write the JSON certificate")
    demo.add_argument("--receipt-out", type=Path, help="write the derived receipt")
    demo.add_argument("--markdown", type=Path, help="write a Markdown report")
    demo.add_argument("--json", action="store_true", help="print full JSON instead of concise text")

    compare = sub.add_parser(
        "compare",
        help="compare baseline and candidate QuotientGuard certificates",
    )
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--out", type=Path, help="write comparison JSON")
    compare.add_argument("--json", action="store_true", help="print full comparison JSON")
    compare.add_argument("--fail-on-new-gap", action="store_true")
    compare.add_argument("--max-gap-ratio-increase", type=float)
    compare.add_argument("--fail-on-evidence-downgrade", action="store_true")

    explain = sub.add_parser(
        "explain",
        help="show mechanical coefficient traces for a supported TRL fixture",
    )
    explain.add_argument("input", type=Path)
    explain.add_argument("--out", type=Path, help="write explanation JSON")
    explain.add_argument("--json", action="store_true", help="print full explanation JSON")

    audit = sub.add_parser("audit", help="audit an exact/coefficient receipt")
    audit.add_argument("receipt", type=Path)
    audit.add_argument("--out", type=Path, help="write the JSON certificate")
    audit.add_argument("--markdown", type=Path, help="write a Markdown report")
    audit.add_argument("--json", action="store_true", help="print full JSON instead of concise text")

    aggregate = sub.add_parser(
        "aggregate",
        help="aggregate sampled trainer observations into a coefficient-only receipt",
    )
    aggregate.add_argument("observations", type=Path, help="JSONL observation stream")
    aggregate.add_argument("--out", type=Path, required=True, help="write the receipt JSON")
    aggregate.add_argument("--run-id")
    aggregate.add_argument("--estimator-name", default="unspecified")
    aggregate.add_argument(
        "--score-kind",
        choices=("full", "sketch"),
        default="sketch",
        help="validate source score vectors as full or sketch; aggregate receipts omit scores",
    )
    aggregate.add_argument("--contract-description", default="")

    trl = sub.add_parser(
        "trl-receipt",
        help="convert a supported TRL sequence-level batch description into a receipt",
    )
    trl.add_argument("batch", type=Path)
    trl.add_argument("--out", type=Path, required=True)

    sentinel = sub.add_parser(
        "sentinel",
        help="summarize persistence across multiple QuotientGuard certificates",
    )
    sentinel.add_argument("certificates", type=Path, nargs="+")
    sentinel.add_argument("--out", type=Path, required=True)
    sentinel.add_argument("--markdown", type=Path)
    sentinel.add_argument("--max-measured-gap-events", type=int)
    sentinel.add_argument("--max-uncertified-events", type=int)
    sentinel.add_argument("--max-consecutive-gap-events", type=int)
    sentinel.add_argument("--max-gap-ratio", type=float)
    sentinel.add_argument("--require-same-estimator", action="store_true")

    benchmark = sub.add_parser(
        "benchmark",
        help="run a local QuotientGuard audit microbenchmark",
    )
    benchmark.add_argument("--samples", type=int, default=256)
    benchmark.add_argument("--dimension", type=int, default=64)
    benchmark.add_argument("--repeats", type=int, default=7)
    benchmark.add_argument("--out", type=Path)

    certify = sub.add_parser(
        "certify",
        help="run a deterministic QuotientGuard certification fixture suite",
    )
    certify.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        help="optional suite manifest; omitted uses the built-in v1 suite",
    )
    certify.add_argument("--out", type=Path)
    certify.add_argument("--json", action="store_true")

    verify_source = sub.add_parser(
        "verify-source",
        help="verify a source file against a frozen QuotientGuard source manifest",
    )
    verify_source.add_argument("source_file", type=Path)
    verify_source.add_argument(
        "--manifest",
        default="trl-grpo-research-frozen-v1",
        choices=tuple(sorted(SOURCE_MANIFESTS)),
    )
    verify_source.add_argument("--out", type=Path)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "check":
        certificate = check_path(args.input)
        _certificate_outputs(args, certificate)
        return 1 if certificate["summary"]["policy_status"] == "FAIL" else 0

    if args.command == "demo":
        receipt, certificate = run_paper_a1_demo()
        if args.receipt_out:
            _write_json(args.receipt_out, receipt)
        _certificate_outputs(args, certificate)
        return 1 if certificate["summary"]["policy_status"] == "FAIL" else 0

    if args.command == "compare":
        result = compare_certificates(
            _load_json(args.baseline),
            _load_json(args.candidate),
            fail_on_new_gap=args.fail_on_new_gap,
            max_gap_ratio_increase=args.max_gap_ratio_increase,
            fail_on_evidence_downgrade=args.fail_on_evidence_downgrade,
        )
        if args.out:
            _write_json(args.out, result)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(comparison_text(result))
        return 1 if result["summary"]["policy_status"] == "FAIL" else 0

    if args.command == "explain":
        result = explain_trl_payload(_load_json(args.input))
        if args.out:
            _write_json(args.out, result)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(explanation_text(result))
        return 0

    if args.command == "audit":
        certificate = audit_receipt(_load_json(args.receipt))
        _certificate_outputs(args, certificate)
        return 1 if certificate["summary"]["policy_status"] == "FAIL" else 0

    if args.command == "aggregate":
        observations = read_jsonl(args.observations)
        receipt = observations_to_receipt(
            observations,
            run_id=args.run_id,
            estimator={"name": args.estimator_name},
            equivalence_contract={"description": args.contract_description},
            score_kind=args.score_kind,
        )
        _write_json(args.out, receipt)
        return 0

    if args.command == "trl-receipt":
        receipt = trl_sequence_batch_to_receipt(_load_json(args.batch))
        _write_json(args.out, receipt)
        return 0

    if args.command == "sentinel":
        certificates = [_load_json(path) for path in args.certificates]
        result = summarize_certificates(
            certificates,
            max_measured_gap_events=args.max_measured_gap_events,
            max_uncertified_events=args.max_uncertified_events,
            max_consecutive_gap_events=args.max_consecutive_gap_events,
            max_gap_ratio=args.max_gap_ratio,
            require_same_estimator=args.require_same_estimator,
        )
        _write_json(args.out, result)
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(sentinel_markdown(result), encoding="utf-8")
        return 1 if result["summary"]["outcome"] == "ALERT" else 0

    if args.command == "benchmark":
        result = benchmark_sampled_audit(
            sample_count=args.samples,
            score_dimension=args.dimension,
            repeats=args.repeats,
        )
        if args.out:
            _write_json(args.out, result)
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    if args.command == "certify":
        result = (
            run_certification_suite(args.manifest)
            if args.manifest is not None
            else run_builtin_certification_suite()
        )
        if args.out:
            _write_json(args.out, result)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(certification_text(result))
        return 1 if result["summary"]["outcome"] == "FAIL" else 0

    if args.command == "verify-source":
        result = verify_source_file(
            args.source_file,
            manifest_id=args.manifest,
        )
        if args.out:
            _write_json(args.out, result)
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
