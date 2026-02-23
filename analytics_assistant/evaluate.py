from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import hashlib
import json
from .graph import ask


def same_value(actual, expected, relative=1e-9, absolute=1e-6):
    if actual is None or expected is None:
        return actual is None and expected is None
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    numeric = (int, float, Decimal)
    if isinstance(actual, numeric) and isinstance(expected, numeric):
        try:
            a, b = Decimal(str(actual)), Decimal(str(expected))
            if not a.is_finite() or not b.is_finite():
                return False
            tolerance = max(Decimal(str(absolute)), Decimal(str(relative)) * abs(b))
            return abs(a - b) <= tolerance
        except InvalidOperation:
            return False
    if isinstance(actual, (date, datetime)):
        actual = actual.isoformat()
    if isinstance(expected, (date, datetime)):
        expected = expected.isoformat()
    return type(actual) is type(expected) and actual == expected


def equivalent(actual, expected, ordered=True, relative=1e-9, absolute=1e-6):
    if actual.truncated or expected.truncated or len(actual.columns) != len(expected.columns) or len(actual.rows) != len(expected.rows):
        return False
    def matches(a, b):
        return len(a) == len(b) and all(same_value(x, y, relative, absolute) for x, y in zip(a, b))
    if ordered:
        return all(matches(a, b) for a, b in zip(actual.rows, expected.rows))
    # Bipartite matching preserves duplicate row multiplicities under tolerances.
    adjacency = [[j for j, expected_row in enumerate(expected.rows) if matches(row, expected_row)] for row in actual.rows]
    assignments = [-1] * len(expected.rows)
    def augment(index, seen):
        for target in adjacency[index]:
            if target in seen:
                continue
            seen.add(target)
            if assignments[target] == -1 or augment(assignments[target], seen):
                assignments[target] = index
                return True
        return False
    return all(augment(i, set()) for i in range(len(actual.rows)))


def load_questions(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    required = {"id", "family", "question", "reference_sql", "ordered", "relative_tolerance", "absolute_tolerance"}
    if not records or len({row["id"] for row in records}) != len(records) or len({row["question"] for row in records}) != len(records):
        raise ValueError("Evaluation questions and IDs must be nonempty and unique")
    if any(not required.issubset(row) or not isinstance(row["ordered"], bool) or row["relative_tolerance"] < 0 or row["absolute_tolerance"] < 0 for row in records):
        raise ValueError("Invalid evaluation question schema")
    return records


def evaluate(questions, modes, graph, warehouse, config, output, source_path):
    if len(set(modes)) != len(modes) or not modes or not set(modes).issubset({"base", "sft"}):
        raise ValueError("Provide unique supported evaluation modes")
    for case in questions:
        warehouse.policy.validate(case["reference_sql"])
    records = []
    with (Path(output) / "cases.jsonl").open("w") as ledger:
        for index, case in enumerate(questions):
            reference = warehouse.execute(case["reference_sql"])
            if reference.truncated:
                raise ValueError(f"Reference result exceeds row cap: {case['id']}")
            order = modes if index % 2 == 0 else list(reversed(modes))
            for mode in order:
                state = ask(graph, case["question"], mode, config.max_attempts)
                result = state.get("result")
                correct = bool(result is not None and not state.get("error") and equivalent(
                    result, reference, case["ordered"], case["relative_tolerance"], case["absolute_tolerance"]))
                record = {"id": case["id"], "family": case["family"], "mode": mode, "question": case["question"],
                          "correct": correct, "attempts": state["attempts"], "error": state.get("error"),
                          "sql": state.get("sql"), "elapsed_seconds": state["elapsed_seconds"],
                          "input_tokens": state.get("input_tokens", 0), "output_tokens": state.get("output_tokens", 0),
                          "trace": state["trace"], "reference": reference.as_dict(),
                          "result": result.as_dict() if result else None}
                ledger.write(json.dumps(record, allow_nan=False) + "\n"); ledger.flush()
                records.append(record)
    summary = {"questions": len(questions), "benchmark_sha256": hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
               "base_model": config.model, "revision": config.revision, "decoding": "greedy", "load_in_4bit": config.load_in_4bit, "max_attempts": config.max_attempts,
               "database": config.database, "schema": config.schema,
               "modes": {}, "accuracy_definition": "Execution-result equivalence, with declared ordering and numeric tolerances"}
    for mode in modes:
        rows = [row for row in records if row["mode"] == mode]
        families = defaultdict(list)
        for row in rows:
            families[row["family"]].append(row["correct"])
        summary["modes"][mode] = {"attempted": len(rows), "correct": sum(row["correct"] for row in rows),
            "accuracy_percent": 100 * sum(row["correct"] for row in rows) / len(rows),
            "first_attempt_correct_percent": 100 * sum(row["correct"] and row["attempts"] == 1 for row in rows) / len(rows),
            "family_accuracy_percent": {family: 100 * sum(values) / len(values) for family, values in families.items()}}
    if set(modes) == {"base", "sft"}:
        summary["sft_gain_percentage_points"] = summary["modes"]["sft"]["accuracy_percent"] - summary["modes"]["base"]["accuracy_percent"]
        paired = {case["id"]: {mode: next(row["correct"] for row in records if row["id"] == case["id"] and row["mode"] == mode) for mode in modes} for case in questions}
        summary["paired_cases"] = {"sft_only_correct": sum(row["sft"] and not row["base"] for row in paired.values()),
                                   "base_only_correct": sum(row["base"] and not row["sft"] for row in paired.values())}
    (Path(output) / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
