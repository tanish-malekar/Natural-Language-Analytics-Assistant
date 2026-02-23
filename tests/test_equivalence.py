from decimal import Decimal
from analytics_assistant.evaluate import equivalent
from analytics_assistant.warehouse import QueryResult


def result(rows, truncated=False):
    return QueryResult(["label", "value"], rows, "fixture", 0, truncated)


def test_unordered_matching_preserves_duplicates():
    expected = result([["a", 1], ["a", 1], ["b", 2]])
    assert equivalent(result([["b", 2], ["a", 1], ["a", 1]]), expected, ordered=False)
    assert not equivalent(result([["b", 2], ["a", 1], ["a", 2]]), expected, ordered=False)


def test_numeric_precision_null_and_truncation_contracts():
    expected = result([[None, Decimal("1.0000000")]])
    assert equivalent(result([[None, Decimal("1.0000001")]]), expected)
    assert not equivalent(result([[None, Decimal("1.1")]]), expected)
    assert not equivalent(result([[None, Decimal("1")]], truncated=True), expected)
