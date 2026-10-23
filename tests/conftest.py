"""Report the case IDs declared in each test body."""

from __future__ import annotations

import inspect
import re

import pytest


CASE_ID_LINE = re.compile(r"#\s*case_id:\s*(.+)")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Emit stable PASS/FAILED lines without changing test outcomes."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or item.function is None:
        return
    source = inspect.getsource(item.function)
    case_ids = []
    for line in source.splitlines():
        match = CASE_ID_LINE.search(line)
        if match:
            case_ids.extend(match.group(1).split())
    if not case_ids:
        return
    reporter = item.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        status = "PASS" if report.passed else "FAILED"
        for case_id in case_ids:
            reporter.write_line(f"\n{status} {case_id}")
