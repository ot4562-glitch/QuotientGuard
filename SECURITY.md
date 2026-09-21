# Security Policy

## Supported version

QuotientGuard v1.0.0 is the frozen public release.

## Reporting a vulnerability

Please use GitHub's private vulnerability-reporting/security-advisory mechanism for the repository rather than opening a public issue with exploit details.

Include:
- affected file/function;
- reproduction steps;
- expected versus observed behavior;
- whether the issue can alter certificate contents, hashes, policy exits, or source verification.

## Security model

QuotientGuard is a local deterministic analysis tool.

The core:
- does not require network access;
- does not require API keys;
- does not execute model-generated code;
- does not send receipts/certificates to a hosted service.

Input files should still be treated as untrusted data. QuotientGuard parses JSON/JSONL and source files but does not intentionally execute them.

Source verification compares bytes against frozen SHA-256 manifests.

## Scientific integrity issues

Incorrect certification, evidence-level promotion, source-provenance bypass, or cross-snapshot pooling should be reported with the same priority as ordinary correctness/security defects because they can invalidate audit conclusions.
