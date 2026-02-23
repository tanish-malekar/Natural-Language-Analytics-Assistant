import argparse
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import json
import pandas as pd
from .config import Config, ROOT
from .evaluate import evaluate, load_questions
from .graph import ask, build_graph
from .llm import QwenSQLModel
from .retrieval import Retriever
from .sql_policy import SQLPolicy
from .warehouse import Warehouse


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-Coder natural language analytics")
    sub = parser.add_subparsers(dest="command", required=True)
    trainer = sub.add_parser("train")
    trainer.add_argument("--config", type=Path, default=ROOT / "configs/sft.json")
    trainer.add_argument("--output", type=Path, required=True)
    schema_parser = sub.add_parser("schema")
    schema_parser.add_argument("--output", type=Path, required=True)
    question_parser = sub.add_parser("ask")
    question_parser.add_argument("question")
    question_parser.add_argument("--mode", choices=["base", "sft"], default="sft")
    question_parser.add_argument("--output", type=Path, required=True)
    evaluator = sub.add_parser("evaluate")
    evaluator.add_argument("--questions", type=Path, default=ROOT / "eval/questions.jsonl")
    evaluator.add_argument("--modes", nargs="+", choices=["base", "sft"], default=["base", "sft"])
    evaluator.add_argument("--limit", type=int)
    evaluator.add_argument("--output", type=Path, required=True)
    for child in (question_parser, evaluator):
        child.add_argument("--attempts", type=int, default=3)
        child.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        from .training import train_adapter
        print(json.dumps(train_adapter(args.config, args.output), indent=2))
        return
    defaults = Config()
    config = Config(max_attempts=getattr(args, "attempts", 3),
                    adapter_path=str(args.adapter) if getattr(args, "adapter", None) else defaults.adapter_path)
    config.validate()
    policy = SQLPolicy(config.database, config.schema, config.max_rows)
    need_adapter = (args.command == "ask" and args.mode == "sft") or (args.command == "evaluate" and "sft" in args.modes)
    model = QwenSQLModel(config, need_adapter) if args.command != "schema" else None
    args.output.mkdir(parents=True, exist_ok=False)
    with Warehouse(config, policy) as warehouse:
        schemas = warehouse.schema()
        (args.output / "schema.json").write_text(json.dumps(schemas, indent=2) + "\n")
        if args.command == "schema":
            return
        graph = build_graph(config, Retriever(schemas), model, warehouse, policy,
                            args.output if args.command == "ask" else None)
        if args.command == "ask":
            state = ask(graph, args.question, args.mode, config.max_attempts)
            result = state.get("result")
            output = {key: value for key, value in state.items() if key not in {"result", "context"}}
            output["result"] = result.as_dict() if result else None
            (args.output / "answer.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
            if result:
                pd.DataFrame(result.rows, columns=result.columns).to_csv(args.output / "result.csv", index=False)
            print(json.dumps(output, indent=2))
        else:
            questions = load_questions(args.questions)
            if args.limit is not None:
                if args.limit < 1:
                    raise ValueError("Limit must be positive")
                questions = questions[:args.limit]
            summary = evaluate(questions, args.modes, graph, warehouse, config, args.output, args.questions)
            summary["configuration"] = {key: value for key, value in asdict(config).items() if key not in {"account", "user", "warehouse", "role"}}
            summary["adapter_manifest"] = model.manifest
            summary["package_versions"] = {name: version(name) for name in ("torch", "transformers", "peft", "langgraph", "sqlglot", "snowflake-connector-python")}
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
