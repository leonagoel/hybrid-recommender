"""Unified setup diagnostics for Hybrid Recommender contributors."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib import error, request

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - exercised only before dependencies are installed.
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKEND_URL = "http://localhost:8000/api/status"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
MIN_PYTHON = (3, 10)

REQUIRED_ENV_VARS = (
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "SUPABASE_SERVICE_KEY",
)

CRITICAL_PATHS = (
    "README.md",
    "requirements.txt",
    ".env.example",
    "backend/main.py",
    "celery_app.py",
    "src/model/hybrid_model.py",
    "src/data/data_adapter.py",
    "frontend/index.html",
    "scripts/check_env.py",
    "scripts/health_check.py",
)

DATASET_PATHS = (
    "datasets",
    "data",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    message: str
    suggestion: str | None = None


def pass_result(name: str, message: str) -> CheckResult:
    return CheckResult(name=name, status="pass", message=message)


def warn_result(name: str, message: str, suggestion: str | None = None) -> CheckResult:
    return CheckResult(name=name, status="warn", message=message, suggestion=suggestion)


def fail_result(name: str, message: str, suggestion: str | None = None) -> CheckResult:
    return CheckResult(name=name, status="fail", message=message, suggestion=suggestion)


def load_environment(root: Path) -> None:
    env_path = root / ".env"
    if load_dotenv is not None and env_path.exists():
        load_dotenv(env_path)


def check_python_version() -> CheckResult:
    current = sys.version_info[:3]
    current_text = ".".join(str(part) for part in current)
    required_text = ".".join(str(part) for part in MIN_PYTHON)

    if current >= MIN_PYTHON:
        return pass_result("Python", f"Python {current_text} is supported")

    return fail_result(
        "Python",
        f"Python {current_text} is below the supported version {required_text}+",
        f"Install Python {required_text} or newer and recreate your virtual environment.",
    )


def parse_requirements(requirements_path: Path) -> list[str]:
    packages: list[str] = []

    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#") or cleaned.startswith("-"):
            continue

        package = cleaned
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
            if separator in package:
                package = package.split(separator, 1)[0]
        packages.append(package.strip())

    return packages


def check_dependencies(root: Path) -> CheckResult:
    requirements_path = root / "requirements.txt"
    if not requirements_path.exists():
        return fail_result(
            "Dependencies",
            "requirements.txt is missing",
            "Restore requirements.txt from the repository or reinstall from a clean checkout.",
        )

    missing = []
    for package in parse_requirements(requirements_path):
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)

    if not missing:
        return pass_result("Dependencies", "All packages from requirements.txt are installed")

    preview = ", ".join(missing[:5])
    suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
    return warn_result(
        "Dependencies",
        f"Missing Python packages: {preview}{suffix}",
        "Run python -m pip install -r requirements.txt inside your virtual environment.",
    )


def check_env_file(root: Path) -> CheckResult:
    env_path = root / ".env"
    if not env_path.exists():
        return warn_result(
            "Environment",
            ".env file was not found",
            "Copy .env.example to .env and fill in your Supabase credentials.",
        )

    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        return warn_result(
            "Environment",
            "Missing required environment variables: " + ", ".join(missing),
            "Update .env with values from your Supabase project settings.",
        )

    placeholder_vars = [
        name
        for name in REQUIRED_ENV_VARS
        if os.getenv(name, "").startswith(("your-", "https://your-project-ref"))
    ]
    if placeholder_vars:
        return warn_result(
            "Environment",
            "Placeholder values detected for: " + ", ".join(placeholder_vars),
            "Replace placeholder values with real Supabase credentials.",
        )

    return pass_result("Environment", ".env contains the required Supabase variables")


def check_backend(url: str, timeout: float = 3.0) -> CheckResult:
    started_at = time.monotonic()
    try:
        with request.urlopen(url, timeout=timeout) as response:
            elapsed_ms = round((time.monotonic() - started_at) * 1000)
            if 200 <= response.status < 300:
                return pass_result("Backend", f"Backend reachable at {url} ({elapsed_ms} ms)")
            return warn_result(
                "Backend",
                f"Backend returned HTTP {response.status} at {url}",
                "Review backend logs and confirm the API is listening on the expected port.",
            )
    except (error.URLError, TimeoutError, OSError) as exc:
        return warn_result(
            "Backend",
            f"Backend is not reachable at {url}: {exc}",
            "Start the API with python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000.",
        )


def parse_redis_target(redis_url: str) -> tuple[str, int]:
    if "://" in redis_url:
        redis_url = redis_url.split("://", 1)[1]
    host_port = redis_url.split("/", 1)[0]
    if "@" in host_port:
        host_port = host_port.rsplit("@", 1)[1]
    host, _, port_text = host_port.partition(":")
    return host or "localhost", int(port_text or "6379")


def check_redis(redis_url: str, timeout: float = 2.0) -> CheckResult:
    host, port = parse_redis_target(redis_url)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return pass_result("Redis", f"Redis socket reachable at {host}:{port}")
    except OSError as exc:
        return warn_result(
            "Redis",
            f"Redis is not reachable at {host}:{port}: {exc}",
            "Start Redis or set REDIS_URL to the broker used by Celery.",
        )


def check_datasets(root: Path) -> CheckResult:
    existing_dirs = [path for path in DATASET_PATHS if (root / path).is_dir()]
    dataset_files = []
    for directory in existing_dirs:
        dataset_files.extend((root / directory).glob("*"))

    data_files = [path for path in dataset_files if path.is_file()]
    if data_files:
        return pass_result(
            "Datasets",
            f"Found {len(data_files)} dataset file(s) in " + ", ".join(existing_dirs),
        )

    return warn_result(
        "Datasets",
        "No dataset files were found in datasets/ or data/",
        "Add sample data or run python scripts/generate_sample_data.py before model testing.",
    )


def check_repository_structure(root: Path) -> CheckResult:
    missing = [path for path in CRITICAL_PATHS if not (root / path).exists()]
    if not missing:
        return pass_result("Repository", "Required project files and directories are present")

    return fail_result(
        "Repository",
        "Missing required project paths: " + ", ".join(missing),
        "Sync your branch with main or restore the missing files from the repository.",
    )


def run_diagnostics(
    root: Path = PROJECT_ROOT,
    backend_url: str = DEFAULT_BACKEND_URL,
    redis_url: str | None = None,
    skip_services: bool = False,
) -> list[CheckResult]:
    load_environment(root)
    redis_target = redis_url or os.getenv("REDIS_URL", DEFAULT_REDIS_URL)

    checks = [
        check_python_version(),
        check_dependencies(root),
        check_env_file(root),
        check_repository_structure(root),
        check_datasets(root),
    ]

    if not skip_services:
        checks.extend([
            check_backend(backend_url),
            check_redis(redis_target),
        ])

    return checks


def summarize(results: Iterable[CheckResult]) -> tuple[int, int, int]:
    passed = warned = failed = 0
    for result in results:
        if result.status == "pass":
            passed += 1
        elif result.status == "warn":
            warned += 1
        else:
            failed += 1
    return passed, warned, failed


def print_report(results: list[CheckResult]) -> None:
    print("Running Hybrid Recommender Diagnostics...")
    print()

    symbols = {
        "pass": "[PASS]",
        "warn": "[WARN]",
        "fail": "[FAIL]",
    }

    for result in results:
        print(f"{symbols[result.status]} {result.name}: {result.message}")
        if result.suggestion:
            print(f"       Suggested fix: {result.suggestion}")

    passed, warned, failed = summarize(results)
    print()
    print(f"Result: {passed} Passed | {warned} Warning(s) | {failed} Failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run setup diagnostics for the Hybrid Recommender project.",
    )
    parser.add_argument(
        "--backend-url",
        default=DEFAULT_BACKEND_URL,
        help=f"Backend status endpoint to check. Default: {DEFAULT_BACKEND_URL}",
    )
    parser.add_argument(
        "--redis-url",
        default=None,
        help="Redis URL to check. Defaults to REDIS_URL or redis://localhost:6379/0.",
    )
    parser.add_argument(
        "--skip-services",
        action="store_true",
        help="Skip backend and Redis connectivity checks.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results = run_diagnostics(
        backend_url=args.backend_url,
        redis_url=args.redis_url,
        skip_services=args.skip_services,
    )
    print_report(results)
    return 1 if any(result.status == "fail" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
