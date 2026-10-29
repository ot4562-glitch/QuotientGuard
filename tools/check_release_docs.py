"""Check published stable or explicitly unpublished local candidate metadata.

Dependency-free Python 3.10+ check. A tag-release workflow can also generate
release notes from the versioned CHANGELOG section, avoiding stale notes files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = r"\d+\.\d+\.\d+"
PACKAGE_VERSION_RE = VERSION_RE + r"(?:\.dev\d+)?"


def _match(pattern: str, text: str, description: str) -> str:
    found = re.search(pattern, text, flags=re.MULTILINE)
    if not found:
        raise ValueError(f"missing or malformed {description}")
    return found.group(1)


def check_release_docs(*, tag: str | None = None) -> tuple[str, str]:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = pyproject.split("[project]", 1)
    if len(section) != 2:
        raise ValueError("missing [project] metadata in pyproject.toml")
    version = _match(
        rf'^version\s*=\s*"({PACKAGE_VERSION_RE})"\s*$',
        section[1].split("\n[", 1)[0],
        "pyproject.toml project version",
    )
    version_py = (ROOT / "quotientguard/version.py").read_text(encoding="utf-8")
    code_version = _match(
        rf'^__version__\s*=\s*"({PACKAGE_VERSION_RE})"\s*$',
        version_py,
        "quotientguard/version.py",
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_version = _match(
        rf'^Current stable release:\s+\*\*v({VERSION_RE})\*\*',
        readme,
        "README.md current stable release",
    )
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    development = '.dev' in version
    candidate = re.search(rf'^Current release candidate:\s+\*\*v({VERSION_RE})\*\* \(unpublished\)\s*$',
                          readme, flags=re.MULTILINE)
    if candidate:
        if development or candidate.group(1) != version:
            raise ValueError("release candidate must match the final local package version")
        if tuple(map(int, version.split('.'))) <= tuple(map(int, readme_version.split('.'))):
            raise ValueError("unpublished candidate must be newer than the published stable release")
        if tag is not None:
            raise ValueError("unpublished candidate cannot pass a publishing tag gate")
    if development:
        if tag is not None:
            raise ValueError("development builds cannot pass a stable release tag gate")
        base = tuple(map(int, version.split('.dev', 1)[0].split('.')))
        if base <= tuple(map(int, readme_version.split('.'))):
            raise ValueError("development version must be newer than the declared current stable release")
        if not re.search(r'^## Unreleased\b', changelog, flags=re.MULTILINE):
            raise ValueError("development build requires an Unreleased changelog section")
    title = re.search(
        rf'^## {re.escape(readme_version if development else version)}\s+-\s+\d{{4}}-\d{{2}}-\d{{2}}\s*$',
        changelog,
        flags=re.MULTILINE,
    )
    if not title:
        raise ValueError(f"CHANGELOG.md lacks a dated {version} section")
    if code_version != version or (not development and not candidate and readme_version != version):
        raise ValueError(
            f"version disagreement: pyproject={version}, "
            f"python={code_version}, README={readme_version}"
        )
    expected_url = (
        f"https://github.com/ot4562-glitch/QuotientGuard/releases/tag/v{readme_version}"
    )
    if expected_url not in readme or f"[v{readme_version}]" not in readme:
        raise ValueError("README.md must link the current release and version table")
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"release tag {tag!r} != package version v{version}")

    # All canonical Markdown links in the README should point to real files.
    for target in re.findall(r"\]\((docs/[^)#?]+\.md)(?:#[^)]*)?\)", readme):
        if not (ROOT / target).is_file():
            raise ValueError(f"broken README documentation link: {target}")

    if development:
        title = re.search(r'^## Unreleased[^\n]*$', changelog, flags=re.MULTILINE)
    start = title.end()
    next_section = re.search(r"^## ", changelog[start:], flags=re.MULTILINE)
    end = start + next_section.start() if next_section else len(changelog)
    section_notes = changelog[start:end].strip()
    if not section_notes:
        raise ValueError(f"empty changelog section for {version}")
    return version, section_notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Expected pushed Git tag, e.g. v1.2.0")
    parser.add_argument("--notes-out", type=Path, help="Write release notes from CHANGELOG")
    args = parser.parse_args()
    try:
        version, section = check_release_docs(tag=args.tag)
        if args.notes_out is not None:
            args.notes_out.write_text(
                f"# QuotientGuard v{version}\n\n{section}\n", encoding="utf-8"
            )
    except (OSError, ValueError) as exc:
        print(f"FAIL release documentation: {exc}", file=sys.stderr)
        return 1
    label = "development metadata (stable release unchanged)" if '.dev' in version else "local/published release metadata"
    print(f"PASS {label}: v{version}; README, code, changelog and links agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
