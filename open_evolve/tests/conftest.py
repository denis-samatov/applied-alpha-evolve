"""Tests execute only repository-authored local fixtures, never generated provider code."""
import pytest


@pytest.fixture(autouse=True)
def consent_to_reviewed_test_fixtures(monkeypatch):
    monkeypatch.setenv("OPENEVOLVE_ALLOW_UNSANDBOXED_EXECUTION", "1")
