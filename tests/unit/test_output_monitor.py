from firewall.config import load_rules_config
from firewall.output_monitor.monitor import OutputMonitor, OutputStatus


def test_redacts_email_in_output():
    monitor = OutputMonitor(load_rules_config().get("output_rules", []))
    result = monitor.evaluate("Contact me at user@example.com for details.")
    assert result.status == OutputStatus.REDACT
    assert "<EMAIL>" not in result.text or "[REDACTED]" in result.text


def test_ok_for_clean_output():
    monitor = OutputMonitor(load_rules_config().get("output_rules", []))
    result = monitor.evaluate("Paris is the capital of France.")
    assert result.status == OutputStatus.OK
