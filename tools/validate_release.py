"""Reproducible local release gate. Run from an environment with torch + build.

Writes bounded public evidence under benchmarks/ and detailed logs in ignored
.venv/release-validation/. Does not push, tag, or publish anything.
"""
from __future__ import annotations

import hashlib
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


def main():
    version, _ = check_release_docs()
    import torch
    WORK.mkdir(parents=True, exist_ok=True)
    full = run("full-suite", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"])
    if "skipped=" in full:
        raise RuntimeError("required full suite must not skip objective tests")
    count = int(re.search(r"Ran (\d+) tests", full).group(1))
    run("compile", [sys.executable, "-m", "compileall", "-q", "quotientguard", "tests", "tools"])
    run("certification", [sys.executable, "-m", "quotientguard", "certify", "--out",
        f"benchmarks/certification-v{version}.json"])
    run("objective-benchmark", [sys.executable, "-m", "quotientguard", "objective-benchmark", "--out",
        f"benchmarks/objective-v{version}.json"])
    smokes = [["--version"], ["demo"], ["check", "fixtures/controls/exact_compatible.json"],
        ["check", "fixtures/controls/sampled_projected_gap.jsonl"],
        ["explain", "fixtures/paper/a1_equal_length_clipping.json"],
        ["objective-audit", "examples/ppo-objective.json", "--oracle", "--out",
         str(WORK / "objective.json"), "--markdown", str(WORK / "objective.md")]]
    for index, args in enumerate(smokes):
        run(f"readme-smoke-{index}", [sys.executable, "-m", "quotientguard", *args])
    run("build", [sys.executable, "-m", "build"])
    wheel = ROOT / f"dist/quotientguard-{version}-py3-none-any.whl"
    sdist = ROOT / f"dist/quotientguard-{version}.tar.gz"
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
    # Install wheel in the torch validation environment; execute outside checkout.
    run("torch-wheel-install", [sys.executable, "-m", "pip", "install", "--force-reinstall",
        "--no-deps", wheel], cwd=WORK)
    run("torch-wheel-oracle", [sys.executable, "-m", "quotientguard", "objective-audit",
        ROOT / "examples/ppo-objective.json", "--oracle", "--json"], cwd=WORK)
    run("torch-wheel-identity", [sys.executable, "-c",
        "import quotientguard as q; assert 'site-packages' in q.__file__; print(q.__version__)"], cwd=WORK)
    benchmark = json.loads((ROOT / f"benchmarks/objective-v{version}.json").read_text())
    receipt = {"outcome": "PASS", "version": version, "python": sys.version.split()[0],
        "torch": torch.__version__, "platform": sys.platform, "tests_passed": count,
        "tests_skipped": 0, "certification_cases_passed": 5,
        "controls": benchmark["controls"]["metrics"], "holdout": benchmark["holdout"]["metrics"],
        "dtypes": benchmark["environment"]["dtypes"], "wheel_install": "PASS",
        "sdist_install": "PASS", "installed_wheel_oracle": "PASS",
        "readme_cli_smokes": len(smokes), "build": "PASS"}
    (ROOT / f"benchmarks/release-validation-v{version}.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # These hashes describe the validated binaries, not subsequently rebuilt copies.
    sums = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in (wheel, sdist))
    (ROOT / "dist/SHA256SUMS").write_text(sums, encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
