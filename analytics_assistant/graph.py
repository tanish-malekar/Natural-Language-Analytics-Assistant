from pathlib import Path
import time
from typing import Any
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from sqlglot.errors import SqlglotError
from .charts import render_chart
from .llm import ModelError
from .warehouse import WarehouseError


class State(TypedDict, total=False):
    question: str
    mode: str
    context: dict
    attempts: int
    plan: dict | None
    sql: str
    error: str | None
    result: Any
    trace: list
    input_tokens: int
    output_tokens: int
    chart_path: str | None
    chart_error: str | None


def build_graph(config, retriever, model, warehouse, policy, output=None):
    def retrieve(state):
        context = retriever.retrieve(state["question"])
        return {"context": context, "trace": [{"event": "schema_retrieved", "tables": list(context["schemas"])}]}

    def generate(state):
        attempt = state.get("attempts", 0) + 1
        try:
            generated = model.generate(state["question"], state["context"], state.get("sql"), state.get("error"), mode=state["mode"])
            plan = generated.plan.model_dump()
            return {"attempts": attempt, "plan": plan, "sql": plan["sql"], "error": None,
                    "input_tokens": state.get("input_tokens", 0) + generated.input_tokens,
                    "output_tokens": state.get("output_tokens", 0) + generated.output_tokens,
                    "trace": state["trace"] + [{"event": "generated", "attempt": attempt, "sql": plan["sql"],
                                                "generation_id": generated.generation_id, "model": generated.model}]}
        except ModelError as exc:
            return {"attempts": attempt, "plan": None, "error": str(exc),
                    "trace": state["trace"] + [{"event": "generation_error", "attempt": attempt, "error": str(exc)}]}

    def validate(state):
        try:
            sql = policy.validate(state["sql"])
            return {"sql": sql, "error": None}
        except (ValueError, SqlglotError) as exc:
            return {"error": str(exc)[:700], "trace": state["trace"] + [{"event": "validation_error", "error": str(exc)[:700]}]}

    def execute(state):
        try:
            result = warehouse.execute(state["sql"])
            return {"result": result, "error": None,
                    "trace": state["trace"] + [{"event": "executed", "query_id": result.query_id,
                                                "rows": len(result.rows), "truncated": result.truncated}]}
        except (WarehouseError, ValueError) as exc:
            return {"error": str(exc)[:700], "result": None,
                    "trace": state["trace"] + [{"event": "execution_error", "error": str(exc)[:700]}]}

    def chart(state):
        if output is None:
            return {"chart_path": None}
        try:
            return {"chart_path": render_chart(state["result"], state["plan"]["chart"], Path(output)), "chart_error": None}
        except (ValueError, TypeError) as exc:
            return {"chart_path": None, "chart_error": str(exc)}

    def after_generation(state):
        if not state.get("error"):
            return "validate"
        return "retry" if state["attempts"] < config.max_attempts else "stop"

    def after_validation(state):
        if not state.get("error"):
            return "execute"
        return "retry" if state["attempts"] < config.max_attempts else "stop"

    def after_execution(state):
        if not state.get("error"):
            return "chart"
        return "retry" if state["attempts"] < config.max_attempts else "stop"

    graph = StateGraph(State)
    for name, function in (("retrieve", retrieve), ("generate", generate), ("validate", validate), ("execute", execute), ("chart", chart)):
        graph.add_node(name, function)
    graph.add_edge(START, "retrieve"); graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges("generate", after_generation, {"validate": "validate", "retry": "generate", "stop": END})
    graph.add_conditional_edges("validate", after_validation, {"execute": "execute", "retry": "generate", "stop": END})
    graph.add_conditional_edges("execute", after_execution, {"chart": "chart", "retry": "generate", "stop": END})
    graph.add_edge("chart", END)
    return graph.compile()


def ask(graph, question, mode, max_attempts):
    if mode not in {"base", "sft"} or not isinstance(question, str) or not 1 <= len(question.strip()) <= 4000:
        raise ValueError("Use a supported mode and a question of 1..4,000 characters")
    start = time.perf_counter()
    result = graph.invoke({"question": question, "mode": mode, "attempts": 0, "trace": [],
                           "input_tokens": 0, "output_tokens": 0, "error": None},
                          config={"recursion_limit": max_attempts * 4 + 8})
    result["elapsed_seconds"] = time.perf_counter() - start
    return result
