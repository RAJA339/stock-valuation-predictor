"""Every job entry point must be importable and runnable as a module.

Deliberately does NOT use the `spark` fixture. A module that only imports once
a SparkSession already exists is a module that crashes on a job cluster: the
driver imports the wheel's entry point before any session is built. This suite
is what catches, for example, a module-level `F.udf(..., "string")` whose
string return type can only be parsed with an active SparkContext.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

try:  # tomllib is stdlib from 3.11; 3.10 is still a supported target.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only hit on Python 3.10
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_MODULES = [
    "pipelines.bronze.autoloader_documents",
    "pipelines.bronze.kinesis_market_events",
    "pipelines.silver.documents_silver",
    "pipelines.silver.market_events_silver",
    "pipelines.silver.instrument_scd2",
    "pipelines.gold.gold_aggregates",
    "pipelines.gold.vector_index",
]


@pytest.mark.parametrize("module_name", PIPELINE_MODULES)
def test_module_imports_without_a_spark_session(module_name):
    # A subprocess guarantees no session leaked in from another test.
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{module_name} failed to import:\n{result.stderr}"


@pytest.mark.parametrize("module_name", PIPELINE_MODULES)
def test_module_exposes_a_main(module_name):
    module = importlib.import_module(module_name)
    assert callable(getattr(module, "main", None)), f"{module_name} has no main()"


def test_declared_entry_points_resolve():
    """Every console script in pyproject.toml points at something real."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    assert scripts, "no entry points declared"

    for name, target in scripts.items():
        module_name, _, attribute = target.partition(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute, None)), (
            f"entry point {name} -> {target} does not resolve; "
            "jobs.yml references these by name and would fail at run time"
        )


def test_jobs_yml_entry_points_match_pyproject():
    """The bundle must not reference an entry point the wheel does not build."""
    yaml = pytest.importorskip("yaml")

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(pyproject["project"]["scripts"])

    jobs = yaml.safe_load((REPO_ROOT / "pipelines/resources/jobs.yml").read_text())
    referenced = {
        task["python_wheel_task"]["entry_point"]
        for job in jobs["resources"]["jobs"].values()
        for task in job["tasks"]
        if "python_wheel_task" in task
    }

    missing = referenced - declared
    assert not missing, f"jobs.yml references undeclared entry points: {sorted(missing)}"
