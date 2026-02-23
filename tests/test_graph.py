from types import SimpleNamespace
from analytics_assistant.graph import ask, build_graph
from analytics_assistant.llm import ChartSpec, Generation, SQLPlan
from analytics_assistant.sql_policy import SQLPolicy
from analytics_assistant.warehouse import QueryResult, WarehouseError


class Retriever:
    def retrieve(self, question):
        return {"schemas": {"REGION": []}, "join_hints": []}


class Model:
    def __init__(self, statements):
        self.statements = iter(statements)
        self.feedback = []
    def generate(self, question, context, previous_sql=None, error=None, mode="sft"):
        self.feedback.append(error)
        plan = SQLPlan(sql=next(self.statements), explanation="Fixture query",
                       chart=ChartSpec(kind="none", x_column=None, y_column=None, title=""))
        return Generation(plan, "fixture", "fixture-model", 10, 10)


class Warehouse:
    def __init__(self):
        self.calls = []
    def execute(self, sql):
        self.calls.append(sql)
        if "BAD_COLUMN" in sql:
            raise WarehouseError("Invalid identifier BAD_COLUMN")
        return QueryResult(["N"], [[7]], "fixture", .01, False)


def test_unsafe_first_plan_is_repaired_without_execution():
    config = SimpleNamespace(max_attempts=2)
    model = Model(["DROP TABLE REGION", "SELECT COUNT(*) AS N FROM REGION"])
    warehouse = Warehouse()
    graph = build_graph(config, Retriever(), model, warehouse, SQLPolicy())
    state = ask(graph, "Count regions", "sft", config.max_attempts)
    assert state["attempts"] == 2 and state["error"] is None
    assert len(warehouse.calls) == 1 and model.feedback[1]
    assert state["result"].rows == [[7]]


def test_sql_error_repairs_are_bounded_and_keep_feedback():
    config = SimpleNamespace(max_attempts=2)
    model = Model(["SELECT BAD_COLUMN FROM REGION", "SELECT COUNT(*) AS N FROM REGION"])
    warehouse = Warehouse()
    graph = build_graph(config, Retriever(), model, warehouse, SQLPolicy())
    state = ask(graph, "Count regions", "base", config.max_attempts)
    assert len(warehouse.calls) == 2 and state["attempts"] == 2
    assert "BAD_COLUMN" in model.feedback[1]
