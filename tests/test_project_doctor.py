from __future__ import annotations

from pathlib import Path

from scripts import project_doctor


def write_minimal_project(root: Path) -> None:
    for path in project_doctor.CRITICAL_PATHS:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")


def test_parse_requirements_handles_version_specifiers(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "\n".join([
            "pandas>=2.0.0,<3.0.0",
            "uvicorn[standard]>=0.23.0,<1.0.0",
            "# ignored",
            "",
            "python-dotenv~=1.0",
        ]),
        encoding="utf-8",
    )

    assert project_doctor.parse_requirements(requirements) == [
        "pandas",
        "uvicorn",
        "python-dotenv",
    ]


def test_repository_structure_reports_missing_paths(tmp_path):
    write_minimal_project(tmp_path)
    (tmp_path / "backend/main.py").unlink()

    result = project_doctor.check_repository_structure(tmp_path)

    assert result.status == "fail"
    assert "backend/main.py" in result.message


def test_environment_check_warns_for_missing_env_file(tmp_path, monkeypatch):
    for name in project_doctor.REQUIRED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    result = project_doctor.check_env_file(tmp_path)

    assert result.status == "warn"
    assert ".env file was not found" in result.message
    assert "Copy .env.example" in result.suggestion


def test_environment_check_accepts_required_variables(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-key")

    result = project_doctor.check_env_file(tmp_path)

    assert result.status == "pass"


def test_dataset_check_warns_when_no_data_files_exist(tmp_path):
    (tmp_path / "datasets").mkdir()

    result = project_doctor.check_datasets(tmp_path)

    assert result.status == "warn"
    assert "No dataset files" in result.message


def test_parse_redis_target_supports_auth_and_database_path():
    assert project_doctor.parse_redis_target("redis://:secret@redis.local:6380/1") == (
        "redis.local",
        6380,
    )


def test_run_diagnostics_can_skip_service_checks(tmp_path, monkeypatch):
    write_minimal_project(tmp_path)
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets/sample.csv").write_text("title\nExample\n", encoding="utf-8")

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-key")
    monkeypatch.setattr(
        project_doctor,
        "check_dependencies",
        lambda root: project_doctor.pass_result("Dependencies", "ok"),
    )

    results = project_doctor.run_diagnostics(root=tmp_path, skip_services=True)

    assert [result.name for result in results] == [
        "Python",
        "Dependencies",
        "Environment",
        "Repository",
        "Datasets",
    ]
    assert all(result.status == "pass" for result in results)
