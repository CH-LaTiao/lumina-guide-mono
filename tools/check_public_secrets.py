"""Report secret and private-artifact locations without printing matched values."""
from pathlib import Path
import fnmatch
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    "provider-token": re.compile(
        rb"\b(?:sk-[A-Za-z0-9_.-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
        rb"github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})\b"
    ),
    "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "credential-literal": re.compile(
        rb"""(?i)\b(?:api[_-]?key|password|passwd|secret[_-]?key|auth[_-]?token)\b"""
        rb"""\s*[:=]\s*["']([^"'\r\n]+)["']"""
    ),
}
PLACEHOLDERS = {b"test", b"test-only", b"fake", b"YOUR_API_KEY", b"YOUR_PASSWORD"}
IGNORE = (".git", ".venv", "runtime", "__pycache__", "models")
PRIVATE_NAMES = (
    ".env", ".env.*", "*.key", "*.pem", "*.p12", "*.pfx",
    "*.onnx", "*.engine", "*.sqlite*", "*.db", "*.log",
    "*.jpg", "*.png", "*.mp4", "*.zip", "*.tar.gz", "*.bundle",
)


def private_name(path):
    return path != ".env.example" and any(
        fnmatch.fnmatch(Path(path).name.lower(), pattern)
        for pattern in PRIVATE_NAMES
    )


def inspect(data):
    findings = []
    for name, pattern in RULES.items():
        for match in pattern.finditer(data):
            if name == "credential-literal" and match.group(1) in PLACEHOLDERS:
                continue
            findings.append((name, data[:match.start()].count(b"\n") + 1))
    return findings


def tracked_files():
    try:
        output = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            stderr=subprocess.DEVNULL,
        )
        return [item.decode("utf-8") for item in output.split(b"\0") if item]
    except (OSError, subprocess.CalledProcessError):
        return [
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*")
            if path.is_file() and not set(path.relative_to(ROOT).parts) & set(IGNORE)
        ]


def main():
    findings = []
    for name in tracked_files():
        if private_name(name):
            findings.append((name, 1, "private-file"))
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        for category, line in inspect(path.read_bytes()):
            findings.append((name, line, category))
    for name, line, category in findings:
        print(f"{name}:{line}: {category} [value hidden]")
    print(f"Public content check: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
