from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import time
import snowflake.connector
from .config import TABLES


class WarehouseError(RuntimeError):
    pass


def json_value(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


@dataclass
class QueryResult:
    columns: list
    rows: list
    query_id: str
    elapsed_seconds: float
    truncated: bool

    def as_dict(self):
        return {"columns": self.columns, "rows": [[json_value(value) for value in row] for row in self.rows],
                "query_id": self.query_id, "elapsed_seconds": self.elapsed_seconds, "truncated": self.truncated}


class Warehouse:
    def __init__(self, config, policy):
        self.config, self.policy, self.connection = config, policy, None

    def __enter__(self):
        arguments = self.config.connection_arguments()
        try:
            self.connection = snowflake.connector.connect(**arguments)
        except Exception as exc:
            raise WarehouseError(f"Snowflake connection failed ({type(exc).__name__})") from None
        return self

    def __exit__(self, *args):
        if self.connection is not None:
            self.connection.close()

    def schema(self):
        # Fixed trusted metadata query; generated SQL cannot access information_schema.
        query = """SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, COMMENT
                   FROM SNOWFLAKE_SAMPLE_DATA.INFORMATION_SCHEMA.COLUMNS
                   WHERE TABLE_SCHEMA = %s ORDER BY TABLE_NAME, ORDINAL_POSITION"""
        result = {}
        with self.connection.cursor() as cursor:
            cursor.execute(query, (self.config.schema,), timeout=self.config.timeout_seconds)
            for table, column, kind, comment in cursor.fetchall():
                if table in TABLES:
                    result.setdefault(table, []).append({"name": column, "type": kind, "comment": comment or ""})
        if set(result) != TABLES:
            raise WarehouseError("The reader role must see all eight TPCH_SF1 tables")
        return result

    def execute(self, sql):
        guarded = self.policy.validate(sql)
        started = time.perf_counter()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(guarded, timeout=self.config.timeout_seconds)
                rows = cursor.fetchmany(self.config.max_rows + 1)
                columns = [column.name for column in cursor.description]
                return QueryResult(columns, [list(row) for row in rows[:self.config.max_rows]], str(cursor.sfqid),
                                   time.perf_counter() - started, len(rows) > self.config.max_rows)
        except snowflake.connector.Error as exc:
            # SQL errors are useful repair feedback; connection/authentication details are excluded.
            code = getattr(exc, "errno", "unknown")
            message = str(getattr(exc, "msg", "Query failed"))[:700]
            raise WarehouseError(f"Snowflake SQL error {code}: {message}") from None
