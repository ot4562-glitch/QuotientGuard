"""Reproducible local release gate. Run from an environment with torch + build.

Writes bounded public evidence under benchmarks/ and detailed logs in ignored
.venv/release-validation/. Does not push, tag, or publish anything.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import venv


from check_release_docs import check_release_docs

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".venv" / "release-validation"


def run(name, command, cwd=ROOT):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.run([str(x) for x in command], cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=flags)
    (WORK / (name + ".log")).write_text(process.stdout, encoding="utf-8")
    if process.returncode:
        print(process.stdout[-4000:])
        raise RuntimeError(f"release gate failed: {name} (exit {process.returncode})")
    print(f"PASS {name}", flush=True)
    return process.stdout


def python_in(path):
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main(argv=None):
    global WORK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, help='isolated local validation logs/environments')
    parser.add_argument('--evidence-dir', type=Path, default=ROOT / 'benchmarks')
    parser.add_argument('--dist-dir', type=Path, help='build directory (default: dist/VERSION)')
    args = parser.parse_args(argv)
    version, _ = check_release_docs()
    WORK = args.work_dir.resolve() if args.work_dir else ROOT / '.venv' / ('release-validation-' + version)
    evidence = args.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    dist = args.dist_dir.resolve() if args.dist_dir else ROOT / 'dist' / version
    dist.mkdir(parents=True, exist_ok=True)
    import torch
    WORK.mkdir(parents=True, exist_ok=True)
    full = run("full-suite", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"])
    if "skipped=" in full:
        raise RuntimeError("required full suite must not skip objective tests")
    count = int(re.search(r"Ran (\d+) tests", full).group(1))
    run("compile", [sys.executable, "-m", "compileall", "-q", "quotientguard", "tests", "tools"])
    run("certification", [sys.executable, "-m", "quotientguard", "certify", "--out",
        evidence / f"certification-v{version}.json"])
    run("objective-benchmark", [sys.executable, "-m", "quotientguard", "objective-benchmark", "--out",
        evidence / f"objective-v{version}.json"])
    smokes = [["--version"], ["demo"], ["check", "fixtures/controls/exact_compatible.json"],
        ["check", "fixtures/controls/sampled_projected_gap.jsonl"],
        ["explain", "fixtures/paper/a1_equal_length_clipping.json"],
        ["objective-audit", "examples/ppo-objective.json", "--oracle", "--out",
         str(WORK / "objective.json"), "--markdown", str(WORK / "objective.md")]]
    for index, args in enumerate(smokes):
        run(f"readme-smoke-{index}", [sys.executable, "-m", "quotientguard", *args])
    for index, (command, filename) in enumerate((('full-objective-audit', 'full-objective.json'),
            ('full-objective-audit', 'ppo-rollout-correction.json'),
            ('full-objective-audit', 'gspo-rollout-correction.json'),
            ('objective-window-audit', 'objective-window.json'))):
        run(f'v12-cli-{index}', [sys.executable, '-m', 'quotientguard', command,
                                  ROOT / 'examples' / filename, '--json'])
    run("build", [sys.executable, "-m", "build", '--outdir', dist])
    wheel = dist / f"quotientguard-{version}-py3-none-any.whl"
    sdist = dist / f"quotientguard-{version}.tar.gz"
    for kind, package in (("wheel", wheel), ("sdist", sdist)):
        env = Path(tempfile.mkdtemp(prefix=kind + "-env-", dir=WORK))
        venv.EnvBuilder(with_pip=True).create(env)
        exe = python_in(env)
        run(kind + "-install", [exe, "-m", "pip", "install", "--no-deps", package], cwd=WORK)
        for command in ("demo", "certify"):
            run(kind + "-" + command, [exe, "-m", "quotientguard", command], cwd=WORK)
        # Imported package must be the installed copy, not the source checkout.
        run(kind + "-identity", [exe, "-c",
            "import quotientguard as q; from pathlib import Path; "
            f"assert q.__version__ == '{version}'; "
            "assert 'site-packages' in str(Path(q.__file__)); print(q.__version__)"], cwd=WORK)
        run(kind + "-objective-json", [exe, "-m", "quotientguard", "objective-audit",
            ROOT / "examples/ppo-objective.json", "--json"], cwd=WORK)
        run(kind + '-survey-help', [exe, '-m', 'quotientguard', 'survey-screen', '--help'], cwd=WORK)
        run(kind + '-native-api', [exe, '-c',
            'from quotientguard.native_validation import not_executed_report; '
            "assert not_executed_report(framework='smoke', path_id='unavailable', reason='no execution')['trainer_executed'] is False"], cwd=WORK)
    # Install wheel in the torch validation environment; execute outside checkout.
    run("torch-wheel-install", [sys.executable, "-m", "pip", "install", "--force-reinstall",
        "--no-deps", wheel], cwd=WORK)
    run("torch-wheel-oracle", [sys.executable, "-m", "quotientguard", "objective-audit",
        ROOT / "examples/ppo-objective.json", "--oracle", "--json"], cwd=WORK)
    run("torch-wheel-identity", [sys.executable, "-c",
        "import quotientguard as q; assert 'site-packages' in q.__file__; print(q.__version__)"], cwd=WORK)
    run('torch-wheel-development', [sys.executable, '-c',
        'from quotientguard import ObjectiveSpec,ObjectiveTrace,audit_objective; '
        "r=audit_objective(ObjectiveSpec(dtype='float64', ratio_level='sequence_masked_geometric_mean', "
        "reduction='per_sequence_active_mean', masking='select_inputs'), "
        'ObjectiveTrace([[0,0,1000]],[[0,0,0]],[[1,1,1]],[[1,1,0]], '
        'observed_coefficients=[[-.5,-.5,0]]),oracle=True); '
        "assert r.verdict=='EXPECTED_OBJECTIVE_EFFECT',r.to_dict(); print(r.verdict)"], cwd=WORK)
    run('torch-wheel-v12', [sys.executable, ROOT / 'tools/check_installed_v12.py', ROOT], cwd=WORK)
    benchmark = json.loads((evidence / f"objective-v{version}.json").read_text())
    receipt = {"outcome": "PASS", "scope": "local_package_validation", "release_authorized": False,
        "version": version, "python": sys.version.split()[0],
        "torch": torch.__version__, "platform": sys.platform, "tests_passed": count,
        "tests_skipped": 0, "certification_cases_passed": 5,
        "controls": benchmark["controls"]["metrics"], "holdout": benchmark["holdout"]["metrics"],
        "development_controls": {k: benchmark['development_controls'][k] for k in ('seed', 'case_count', 'passed')},
        "development_holdout": {k: benchmark['development_holdout'][k] for k in ('seed', 'case_count', 'passed')},
        "cuda_executed": False,
        "dtypes": benchmark["environment"]["dtypes"], "wheel_install": "PASS",
        "sdist_install": "PASS", "installed_wheel_oracle": "PASS",
        "readme_cli_smokes": len(smokes), "v12_cli_smokes": 4,
        "installed_v12_full_window_parameter_training_survey": "PASS", "build": "PASS"}
    (evidence / f"release-validation-v{version}.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # These hashes describe the validated binaries, not subsequently rebuilt copies.
    sums = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in (wheel, sdist))
    (dist / "SHA256SUMS").write_text(sums, encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
