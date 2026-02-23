import json
from analytics_assistant.config import ROOT
from analytics_assistant.llm import extract_sql
from analytics_assistant.training import assert_disjoint, read_rows


def test_supervised_data_are_separate_from_the_execution_suite():
    train = read_rows(ROOT / "data/sft_train.jsonl")
    validation = read_rows(ROOT / "data/sft_validation.jsonl")
    evaluation = [json.loads(line) for line in (ROOT / "eval/questions.jsonl").read_text().splitlines()]
    assert len(train) == 480 and len(validation) == 120 and len(evaluation) == 150
    assert_disjoint(train, validation, evaluation)


def test_fenced_sql_is_extracted_without_accepting_trailing_prose():
    assert extract_sql("```sql\nSELECT COUNT(*) FROM REGION\n```") == "SELECT COUNT(*) FROM REGION"
    assert extract_sql("SELECT * FROM REGION; DROP TABLE REGION").endswith("DROP TABLE REGION")
