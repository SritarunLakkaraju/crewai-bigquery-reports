"""Generates the synthetic dataset before the tests if it is not there yet (it is seeded, so always identical)."""
import os
import subprocess
import sys

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
ROOT = os.path.dirname(os.path.abspath(__file__))
GENERATORS = [('fde3_crewai_bigquery.data_gen', 'fde3_crewai_bigquery/data/warehouse.duckdb')]


def pytest_sessionstart(session):
    for module, marker in GENERATORS:
        if not os.path.exists(os.path.join(ROOT, marker)):
            subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True, capture_output=True)
