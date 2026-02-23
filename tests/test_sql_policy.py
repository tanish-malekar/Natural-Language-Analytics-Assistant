import json
from pathlib import Path
import pytest
from sqlglot.errors import SqlglotError
from analytics_assistant.config import ROOT
from analytics_assistant.sql_policy import SQLPolicy


@pytest.mark.parametrize("sql", [
    "DROP TABLE CUSTOMER", "SELECT * FROM CUSTOMER; DELETE FROM CUSTOMER",
    "SELECT * FROM OTHER_DATABASE.PUBLIC.CUSTOMER", "SELECT * FROM INFORMATION_SCHEMA.TABLES",
    "SELECT SYSTEM$SEND_EMAIL('x') FROM REGION", "SELECT * FROM IDENTIFIER('CUSTOMER')",
    "SELECT * FROM CUSTOMER CROSS JOIN ORDERS", "SELECT $secret FROM REGION",
])
def test_rejects_nonanalytic_queries(sql):
    with pytest.raises((ValueError, SqlglotError)):
        SQLPolicy().validate(sql)


def test_qualifies_physical_tables_and_preserves_cte_scope():
    sql = SQLPolicy(max_rows=20).validate("WITH totals AS (SELECT COUNT(*) AS n FROM REGION) SELECT n FROM totals")
    assert "SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION" in sql
    assert "TPCH_SF1.totals" not in sql
    assert "LIMIT 21" in sql


def test_reference_suite_has_full_valid_policy_coverage():
    rows = [json.loads(line) for line in (ROOT / "eval/questions.jsonl").read_text().splitlines()]
    assert len(rows) == 150 and len({row["family"] for row in rows}) == 15
    for row in rows:
        assert SQLPolicy().validate(row["reference_sql"])
