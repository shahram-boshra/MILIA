# Security Policy

Thank you for helping keep MILIA and its users safe. We take security vulnerabilities seriously and appreciate responsible disclosure.

## Supported Versions

| Version | Python | Supported |
|---------|--------|-----------|
| 1.1.x   | 3.10, 3.11, 3.12 | ✅ Yes — security updates provided |

Only the latest released version receives security updates. When a new minor or major version is released, the previous version will no longer receive patches. Refer to the [Changelog](https://github.com/shahram-boshra/MILIA/blob/main/CHANGELOG.md) for release history.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues, pull requests, or discussions.**

Instead, use one of the following private channels:

1. **GitHub Private Vulnerability Reporting** (preferred): Navigate to the [Security Advisories](https://github.com/shahram-boshra/MILIA/security/advisories) page and click **"Report a vulnerability"**. This keeps the report confidential between you and the maintainers until a fix is ready.

2. **Email**: Send your report to **a.boshra@gmail.com** with the subject line `[SECURITY] MILIA — <brief description>`.

### What to Include

When reporting a vulnerability, please provide as much of the following as possible so we can reproduce and assess the issue:

- A description of the vulnerability and its potential impact.
- Detailed steps to reproduce the issue, including any configuration (YAML files, CLI arguments, or plugin configurations) involved.
- The MILIA version, Python version, operating system, and versions of key dependencies. You can retrieve this programmatically:
  ```python
  from milia_pipeline import get_package_info, check_dependencies
  print(get_package_info())
  print(check_dependencies())
  ```
- Any proof-of-concept code or logs demonstrating the issue.
- Your assessment of the severity (e.g., data exposure, arbitrary code execution, denial of service).

## Response Timeline

We are committed to the following response targets:

| Stage | Target |
|-------|--------|
| **Acknowledgment** | Within 48 hours of receiving your report |
| **Initial assessment** | Within 7 days — we will confirm whether the report is accepted, request additional information, or explain why it does not qualify |
| **Fix development** | Varies by complexity — we will keep you informed of progress |
| **Security release** | A dedicated patch release as promptly as feasible after the fix is validated |
| **Public disclosure** | Coordinated with you after the fix is released (see below) |

## Disclosure Policy

We follow **coordinated vulnerability disclosure**:

1. When we receive a report, we will work to validate and reproduce the issue.
2. We will develop and test a fix privately.
3. We will coordinate a disclosure timeline with you before publishing.
4. Once the fix is released, we will publish a [GitHub Security Advisory](https://github.com/shahram-boshra/MILIA/security/advisories) crediting the reporter (unless you prefer to remain anonymous).
5. We will add a `Security` entry to the `[Unreleased]` section of [CHANGELOG.md](CHANGELOG.md) per our Keep a Changelog format.

We ask that you:

- Allow us a reasonable timeframe to address the issue before any public disclosure.
- Avoid exploiting the vulnerability beyond what is necessary to demonstrate it.
- Avoid accessing, modifying, or deleting data belonging to other users.

## Scope

The following are examples of issues we consider security vulnerabilities in MILIA:

- **Plugin system abuse**: Malicious or untrusted plugins that bypass the plugin validation and security checks in the plugin architecture.
- **YAML configuration injection**: Crafted YAML inputs that exploit the configuration loading or schema validation system to execute unintended operations.
- **Dependency chain issues**: Vulnerabilities in MILIA's direct dependencies (PyTorch, PyTorch Geometric, RDKit, etc.) that are exploitable through MILIA's interfaces.
- **Path traversal**: File path manipulation through configuration parameters, CLI arguments, or dataset handler paths that allows reading or writing outside intended directories.
- **Arbitrary code execution**: Any mechanism through which untrusted input leads to execution of arbitrary code.
- **Sensitive information exposure**: Unintended logging, error messages, or outputs that leak filesystem paths, credentials, or internal state.

The following are **not** in scope for this security policy:

- Bugs that do not have a security impact (please report these as [regular issues](https://github.com/shahram-boshra/MILIA/issues) per the [Contributing Guide](CONTRIBUTING.md)).
- Vulnerabilities in upstream dependencies that are not exploitable through MILIA's interfaces (please report these to the respective upstream projects).
- Issues requiring physical access to the machine running MILIA.

## Security-Related Configuration

MILIA users should be aware of the following security considerations:

- **Plugin trust**: Only enable plugins from trusted sources. MILIA's plugin system includes validation checks, but plugins execute arbitrary Python code by design.
- **Configuration validation**: Always run with configuration schema validation enabled (the default) to catch malformed or unexpected inputs early.
- **File paths**: Review all file paths in YAML configuration files, especially when processing data from untrusted sources.

## References

- [GitHub — Adding a security policy to your repository](https://docs.github.com/en/code-security/getting-started/adding-a-security-policy-to-your-repository)
- [GitHub — Privately reporting a security vulnerability](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
- [OpenSSF OSPS Baseline — Vulnerability Management](https://baseline.openssf.org/)
