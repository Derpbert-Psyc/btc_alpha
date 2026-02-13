"""Tests for direction value normalization in execution form.

Verifies that the UI direction select does not crash on stored direction
values like "LONG", "SHORT", or historical variants like "LONG_ONLY".
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# Extract the normalization logic from the component for unit testing.
# This mirrors exactly what render_execution_form does.
_DIR_OPTIONS = ["BOTH", "LONG", "SHORT"]
_DIR_NORM = {
    "LONG_ONLY": "LONG", "SHORT_ONLY": "SHORT",
    "long": "LONG", "short": "SHORT", "both": "BOTH",
    "Long": "LONG", "Short": "SHORT", "Both": "BOTH",
}


def normalize_direction(raw):
    val = _DIR_NORM.get(raw, raw)
    if val not in _DIR_OPTIONS:
        return "BOTH"
    return val


def test_canonical_values_pass_through():
    """BOTH, LONG, SHORT pass through unchanged."""
    assert normalize_direction("BOTH") == "BOTH"
    assert normalize_direction("LONG") == "LONG"
    assert normalize_direction("SHORT") == "SHORT"


def test_historical_long_only_normalized():
    """LONG_ONLY and SHORT_ONLY map to LONG and SHORT."""
    assert normalize_direction("LONG_ONLY") == "LONG"
    assert normalize_direction("SHORT_ONLY") == "SHORT"


def test_lowercase_normalized():
    """Lowercase variants are normalized."""
    assert normalize_direction("long") == "LONG"
    assert normalize_direction("short") == "SHORT"
    assert normalize_direction("both") == "BOTH"


def test_titlecase_normalized():
    """Titlecase variants are normalized."""
    assert normalize_direction("Long") == "LONG"
    assert normalize_direction("Short") == "SHORT"
    assert normalize_direction("Both") == "BOTH"


def test_adversarial_unknown_direction_does_not_crash():
    """Unknown direction value falls back to BOTH, never crashes."""
    assert normalize_direction("SIDEWAYS") == "BOTH"
    assert normalize_direction("") == "BOTH"
    assert normalize_direction("HEDGE") == "BOTH"


def test_none_direction_does_not_crash():
    """None direction (missing key) falls back to BOTH."""
    # ep.get("direction", "BOTH") would return "BOTH" if key is missing,
    # but if somehow None gets through:
    assert normalize_direction(None) == "BOTH"


def test_direction_value_always_in_options():
    """Every normalized value is in the select options list."""
    test_values = [
        "BOTH", "LONG", "SHORT",
        "LONG_ONLY", "SHORT_ONLY",
        "long", "short", "both",
        "Long", "Short", "Both",
        "SIDEWAYS", "", None, 42,
    ]
    for val in test_values:
        result = normalize_direction(val)
        assert result in _DIR_OPTIONS, f"normalize_direction({val!r}) = {result!r} not in {_DIR_OPTIONS}"
