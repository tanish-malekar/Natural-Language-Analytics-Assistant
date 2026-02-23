# Paired base/SFT execution evaluation

`questions.jsonl` contains 150 project-authored questions in fifteen parameterized
families. Each case defines a reference SELECT query, column/row ordering contract,
and numeric tolerances. This is a project-specific suite over TPC-H data, distinct
from the official 22-query benchmark. Related variants are correlated.

Training and validation live in `data/`. Split checks compare IDs, normalized
question text, and canonical SQL. Evaluation references never become model
prompts, supervised targets, checkpoint-selection data, or repair hints.

Base mode disables the LoRA adapter on the same pinned Qwen2.5-Coder foundation.
SFT mode enables it. Both share schema retrieval, prompts, greedy decoding,
quantization, row caps, and retry budgets. Final and first-attempt execution
accuracies are reported separately. A reference error aborts the comparison.

Keep the base revision, adapter manifest, benchmark hash, retry policy, and scope
with any reported score. Subset runs use their real denominator.
