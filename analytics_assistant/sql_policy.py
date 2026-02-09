import sqlglot
from sqlglot import expressions as exp
from sqlglot.optimizer.scope import traverse_scope
from .config import TABLES

SAFE_FUNCTIONS = {"COUNT", "SUM", "AVG", "MIN", "MAX", "ABS", "ROUND", "COALESCE", "NULLIF", "UPPER", "LOWER",
                  "CAST", "TRY_CAST", "EXTRACT", "YEAR", "MONTH", "DAY", "DATE", "DATE_ADD", "DATE_SUB", "DATE_DIFF",
                  "DATEDIFF", "DATEADD", "DATE_TRUNC", "TIMESTAMP_TRUNC", "TS_OR_DS_TO_DATE", "STR_TO_DATE",
                  "CASE", "IF", "IFF", "ROW_NUMBER", "RANK", "DENSE_RANK"}
FORBIDDEN = {"insert", "update", "delete", "merge", "create", "drop", "alter", "command", "copy", "into", "lock",
             "transaction", "commit", "rollback", "set", "use", "grant", "revoke", "pragma", "attach", "detach", "execute", "parameter", "sessionparameter", "placeholder"}


class SQLPolicy:
    def __init__(self, database="SNOWFLAKE_SAMPLE_DATA", schema="TPCH_SF1", max_rows=1000):
        self.database, self.schema, self.max_rows = database, schema, max_rows
        if database != "SNOWFLAKE_SAMPLE_DATA" or schema != "TPCH_SF1" or not 1 <= max_rows <= 10000:
            raise ValueError("Policy targets only Snowflake's TPCH_SF1 sample tables")

    def validate(self, sql):
        if not isinstance(sql, str) or not sql.strip() or len(sql) > 20000:
            raise ValueError("SQL must contain 1..20,000 characters")
        statements = sqlglot.parse(sql, read="snowflake")
        if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.Union, exp.Intersect, exp.Except)):
            raise ValueError("Exactly one read-only SELECT query is allowed")
        tree = statements[0]
        nodes = list(tree.walk())
        if len(nodes) > 2000 or any(node.key in FORBIDDEN for node in nodes):
            raise ValueError("SQL contains a prohibited operation or exceeds the AST budget")
        for node in tree.find_all(exp.Func):
            name = node.name.upper() if isinstance(node, exp.Anonymous) else node.sql_name().upper()
            if name not in SAFE_FUNCTIONS:
                raise ValueError(f"Function {name} is outside the read-only analytics allowlist")
        # Scope analysis distinguishes CTE names from physical table references.
        physical = {}
        for scope in traverse_scope(tree):
            for source in scope.sources.values():
                if isinstance(source, exp.Table):
                    physical[id(source)] = source
        if not physical:
            raise ValueError("Analytics SQL must read at least one TPC-H table")
        for join in tree.find_all(exp.Join):
            if not join.args.get("on") and not join.args.get("using"):
                raise ValueError("Use explicit joins with key predicates")
        for table in physical.values():
            if not isinstance(table.this, exp.Identifier):
                raise ValueError("Table functions and dynamic identifiers are prohibited")
            name = table.name.upper()
            if name not in TABLES or (table.db and table.db.upper() != self.schema) or (table.catalog and table.catalog.upper() != self.database):
                raise ValueError("Only the eight TPCH_SF1 sample tables are permitted")
            table.set("this", exp.to_identifier(name))
            table.set("db", exp.to_identifier(self.schema))
            table.set("catalog", exp.to_identifier(self.database))
        limit = tree.args.get("limit")
        cap = self.max_rows + 1
        if limit is not None:
            literal = limit.expression
            if not isinstance(literal, exp.Literal) or not literal.is_int or int(literal.this) < 0:
                raise ValueError("Result limits must be nonnegative integer literals")
            cap = min(cap, int(literal.this))
        return tree.limit(cap, copy=False).sql(dialect="snowflake", pretty=False)
