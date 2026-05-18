import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
# Allow tests to import Chapter 5 application modules directly.
sys.path.insert(0, str(ROOT / "AganticAssistant"))
sys.path.insert(1, str(ROOT / "McpServer"))

module_results = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0})


def pytest_configure(config):
	# Disable pytest output capture so print() appears in real time.
	config.option.capture = "no"


def pytest_runtest_logreport(report):
	if report.when != "call":
		return
	module = report.location[0]
	test_name = report.location[2]
	if report.passed:
		status = "PASS"
		count_key = "passed"
	elif report.failed:
		status = "FAIL"
		count_key = "failed"
	elif report.skipped:
		status = "SKIP"
		count_key = "skipped"
	else:
		status = report.outcome.upper()
		count_key = report.outcome.lower()

	# Print in real time for educational visibility
	print(f"[{module}] {test_name} - {status}")
	sys.stdout.flush()
	module_results[module][count_key] += 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
	if not module_results:
		return
	terminalreporter.write_sep("-", "Chapter 5 Test Module Summary")
	for module, counts in module_results.items():
		terminalreporter.write_line(
			f"{module}: passed={counts['passed']} failed={counts['failed']} skipped={counts['skipped']}"
		)
