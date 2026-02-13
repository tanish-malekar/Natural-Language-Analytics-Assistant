# Authored SFT corpus

The corpus has 480 training and 120 validation examples from twelve target-query
families. A fixed seed splits parameters within each family. Rows contain IDs,
question text, a family label, and read-only Snowflake SQL over TPCH_SF1.
`corpus_manifest.json` records the authoring and split counts.

The seed schema in `resources/schema_seed.json` provides names and types for
constructing training prompts. Runtime inference retrieves the live Snowflake
schema instead. TRL consumes conversational prompt/completion pairs; only SQL
completion tokens contribute to supervised loss. Long examples are rejected
before model loading rather than silently truncating targets.

`eval/questions.jsonl` is a separate 150-question suite with other target-query
families. The training audit rejects duplicate IDs, normalized questions, and
canonical SQL across all three splits. Template correlations remain within each
suite, so results should be interpreted for their stated domain and workload.
No warehouse rows or hidden model answers are included in these source files.
