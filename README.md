# Natural Language Analytics Assistant

A text-to-SQL agent using **Python, LangGraph, PyTorch, Hugging Face, and
Snowflake**. It retrieves TPC-H schemas, generates SQL with a supervised
Qwen2.5-Coder adapter, validates queries, and repairs execution errors. A paired
150-question evaluation compares the off-the-shelf checkpoint with its SFT adapter
under identical prompts, schema context, decoding, and retry rules.

## Architecture

```text
question -> retrieve schemas and join keys -> Qwen2.5-Coder -> validate SQL AST
                                               ^                  |
                                               |                  v
                                      bounded repair <- Snowflake execution
                                                                  |
                                                         results / Plotly chart
```

LangGraph tracks retrieved tables, generated SQL, errors, attempt count, local
generation IDs, token counts, and Snowflake query IDs. The model produces SQL
only. Execution results come from the warehouse; optional Plotly charts use fixed
rendering functions and actual selected columns. The agent does not execute
model-generated Python or JavaScript.

## Qwen and supervised fine-tuning

The default foundation is `Qwen/Qwen2.5-Coder-7B-Instruct`, pinned to an immutable
Hugging Face revision. Here **base** means that off-the-shelf instruction checkpoint
before this project's domain adaptation. **SFT** means the same weights plus the
trained LoRA adapter; it does not denote a different foundation model.

The recipe uses Hugging Face Transformers, TRL SFTTrainer, PEFT LoRA, and PyTorch.
Default QLoRA training loads NF4 four-bit weights, uses a rank-16 adapter on
attention and MLP projections, and optimizes SQL completion tokens only. Prompts
and schema context are masked out of the loss. Qwen's own chat template and EOS
are retained. Training examples that exceed the configured token budget are
rejected rather than truncating their SQL targets.

`data/sft_train.jsonl` has **480 examples** and `data/sft_validation.jsonl` has
**120 examples**, authored as twelve parameterized SQL families. Validation
selects the adapter by completion loss. The separate evaluation file contains
**150 questions** in fifteen other target-query families. Split auditing rejects
overlapping IDs, normalized questions, and canonical SQL across all three files.
The audit reads evaluation references only for overlap checks; they do not enter
training prompts, labels, checkpoint selection, or repair feedback.

The training and validation examples share template families with disjoint
parameters. The evaluation suite also has correlated parameter variants, so
reported accuracy describes this project's defined workload rather than general
natural-language SQL competence.

## Training

Use Python 3.11–3.13 and install a PyTorch build compatible with your hardware.
The provided SFT recipe targets one CUDA GPU, with a configurable sequence length,
precision, accumulation count, and optional four-bit loading. GPU memory needs
depend on sequence length and model settings; the default seven-billion-parameter
recipe is intended for a suitably provisioned GPU environment.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,dev]'
nl-analytics train --config configs/sft.json --output artifacts/sft-01
```

`configs/sft.json` controls the base revision, epochs, learning rate, LoRA settings,
maximum length, precision, evaluation interval, and seed. Choose `fp16` where
`bf16` is unsupported. The trainer writes adapter weights, tokenizer files,
checkpoints, and `training_manifest.json`, including checkpoint identity, prompt
version, split hashes, and training metrics. Model weights and experiment outputs
are excluded from Git. There is no automatic upload to a model hub.

Training uses a static, versioned TPC-H schema seed and authored SQL labels, so it
is independent of Snowflake credentials. Interactive questions retrieve live
metadata from Snowflake instead. Keep model and revision settings aligned between
training and inference; loading an adapter with a different base identity is
rejected.

## Snowflake and inference

The fixed data target is `SNOWFLAKE_SAMPLE_DATA.TPCH_SF1`: CUSTOMER, ORDERS,
LINEITEM, PART, PARTSUPP, SUPPLIER, NATION, and REGION. A dedicated reader role
needs access to those sample tables and the selected warehouse. The application
does not create databases or grant roles. Schema retrieval uses the database's
information-schema column view, and join hints follow the TPC-H key graph.

```sh
cp .env.example .env
nl-analytics schema --output artifacts/schema-01
nl-analytics ask "Show discounted line-item revenue by ship mode." --mode sft --adapter artifacts/sft-01/adapter --output artifacts/answer-01
nl-analytics ask "How many customers are in each market segment?" --mode base --output artifacts/base-answer-01
```

Set the Snowflake account, user, warehouse, and restricted role. Configure key-pair,
external-browser, or supported password authentication according to your account
policy. `.env` and key files are ignored by Git. Qwen weights are downloaded through
Hugging Face and run locally; `HF_TOKEN` can be used where authorized downloads
require it. No model API service is required.

`QWEN_ADAPTER_PATH` can supply the adapter path instead of `--adapter`.
`QWEN_LOAD_4BIT=1` enables compatible CUDA quantized inference with the training
extra installed. Base and SFT evaluation share the same loaded model and
quantization: base mode temporarily disables the adapter. Without quantization,
CUDA inference uses a supported low-precision dtype; CPU inference uses float32
and requires substantial memory. SFT mode requires an explicit adapter with a
matching training manifest and cannot silently fall back to base weights.

Output directories must be new. `answer.json` stores query logic, SQL, attempts,
trace, result values, truncation status, and query ID. `result.csv` holds returned
rows; `chart.html` is a standalone Plotly visualization when the result supports
one. Decimal values are serialized as strings in JSON to preserve precision.

## SQL policy and repair

SQLGlot parses exactly one SELECT query, resolves CTE scopes, qualifies physical
tables, and restricts them to the eight sample tables. An analytical function
allowlist rejects external calls. Commands, mutations, dynamic identifiers,
session variables, and joins without explicit key predicates are prohibited.
Generated SQL cannot access information_schema; schema retrieval uses a separate
trusted statement. A result cap and statement timeouts bound output and execution.
Use a restricted warehouse role as an independent permission boundary.

Generation is greedy with a bounded token budget. There are three attempts by
default, including the first. Validation and Snowflake SQL errors can trigger a
repair using the same retrieved context. A wrong query that executes successfully
does not receive reference-answer feedback or a semantic retry.

## Base versus SFT: 150 questions

```sh
nl-analytics evaluate --adapter artifacts/sft-01/adapter --limit 10 --output artifacts/pilot-01
nl-analytics evaluate --adapter artifacts/sft-01/adapter --output artifacts/evaluation-01
nl-analytics evaluate --modes base sft --adapter artifacts/sft-01/adapter --attempts 1 --output artifacts/single-attempt-01
python -m pytest
```

Both modes use identical question text, live schema retrieval, SQL policy,
quantization, greedy decoding, token limits, and maximum attempts. The only
inference change is whether the LoRA adapter is enabled. Reference results are
computed once per question, and mode order alternates. A reference failure aborts
the comparison rather than removing a difficult case from the denominator.

Correctness is execution-result equivalence: column counts and positions, requested
row ordering, duplicate multiplicity, and declared numeric tolerances. Aliases may
differ. Truncated results are incorrect. The report records final accuracy,
first-attempt-correct percentage, per-family scores, and paired base-only/SFT-only
successes. Limited runs use their actual number of questions.

`cases.jsonl`, `summary.json`, and `schema.json` contain results, traces, model
revision, adapter manifest, benchmark hash, package versions, and settings.
Accuracy percentages and SFT gains are calculated from the recorded outcomes.
Snowflake queries use the configured warehouse and can incur compute charges.

## Layout

- `training.py` / `configs/sft.json` — completion-only SFT and adapter provenance
- `llm.py` / `prompts.py` — local Qwen generation and shared prompt construction
- `retrieval.py` — live schema ranking and join-key closure
- `sql_policy.py` / `warehouse.py` — read-only validation and Snowflake access
- `graph.py` — LangGraph generation, repair, execution, and rendering
- `evaluate.py` / `eval/` — paired execution-accuracy protocol
- `data/` — separate training/validation corpora and split documentation

Test specifications cover policy boundaries, CTEs, split disjointness, bounded
repair, numeric tolerances, and duplicate-row matching. Fake model and warehouse
adapters exercise graph behavior without model weights or database credentials.

References: [Qwen2.5-Coder model card](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct),
[Hugging Face SFTTrainer](https://huggingface.co/docs/trl/sft_trainer),
[PEFT quantization](https://huggingface.co/docs/peft/main/en/developer_guides/quantization),
[LangGraph](https://docs.langchain.com/oss/python/langgraph/graph-api), and
[Snowflake TPC-H](https://docs.snowflake.com/en/user-guide/sample-data-tpch).

Project code is MIT licensed. Qwen weights follow their upstream Apache-2.0 license.
TPC-H is a TPC benchmark; this repository does not redistribute its warehouse data.
