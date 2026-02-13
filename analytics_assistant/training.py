from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import re
from .config import ROOT
from .prompts import PROMPT_VERSION, messages
from .retrieval import Retriever
from .sql_policy import SQLPolicy


@dataclass
class SFT:
    base_model: str
    revision: str
    epochs: int = 3
    learning_rate: float = .0002
    max_length: int = 3072
    batch_size: int = 1
    gradient_accumulation: int = 16
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = .05
    load_in_4bit: bool = True
    precision: str = "bf16"
    eval_steps: int = 20
    seed: int = 42


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or any(not {"id", "family", "question", "sql"}.issubset(row) for row in rows):
        raise ValueError("Training rows require id, family, question, and SQL")
    return rows


def split_fingerprints(rows, sql_key="sql"):
    policy = SQLPolicy()
    texts, queries, ids = set(), set(), set()
    for row in rows:
        question = re.sub(r"\s+", " ", row["question"].strip().casefold())
        query = policy.validate(row[sql_key])
        if row["id"] in ids or question in texts or query in queries:
            raise ValueError("Duplicate training/evaluation identity, question, or canonical SQL")
        ids.add(row["id"]); texts.add(question); queries.add(query)
    return ids, texts, queries


def assert_disjoint(train, validation, evaluation):
    groups = [split_fingerprints(train), split_fingerprints(validation), split_fingerprints(evaluation, "reference_sql")]
    for left in range(3):
        for right in range(left + 1, 3):
            if any(groups[left][kind] & groups[right][kind] for kind in range(3)):
                raise ValueError("Training, validation, and evaluation overlap")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_adapter(config_path, output):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer
    config = SFT(**json.loads(Path(config_path).read_text()))
    if not torch.cuda.is_available() or int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This training recipe targets one CUDA GPU")
    if not re.fullmatch(r"[0-9a-f]{40}", config.revision) or not config.base_model.startswith("Qwen/Qwen2.5-Coder-"):
        raise ValueError("Pin a full Qwen2.5-Coder checkpoint revision")
    if config.precision not in {"bf16", "fp16"} or min(config.epochs, config.batch_size, config.gradient_accumulation, config.lora_rank, config.eval_steps) < 1 or config.max_length < 256 or not 0 < config.learning_rate < .1 or not 0 <= config.lora_dropout < 1:
        raise ValueError("Invalid SFT configuration")
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Choose fp16 on GPUs without bf16 support")
    train_path, validation_path, eval_path = ROOT / "data/sft_train.jsonl", ROOT / "data/sft_validation.jsonl", ROOT / "eval/questions.jsonl"
    train, validation = read_rows(train_path), read_rows(validation_path)
    evaluation = [json.loads(line) for line in eval_path.read_text().splitlines() if line.strip()]
    assert_disjoint(train, validation, evaluation)
    retriever = Retriever(json.loads((ROOT / "resources/schema_seed.json").read_text()))
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    def prepare(rows):
        prepared = []
        for row in rows:
            prompt = messages(row["question"], retriever.retrieve(row["question"]))
            completion = [{"role": "assistant", "content": row["sql"]}]
            size = len(tokenizer.apply_chat_template(prompt + completion, tokenize=True, add_generation_prompt=False))
            if size > config.max_length:
                raise ValueError(f"Refusing to truncate SQL supervision for {row['id']}: {size} tokens")
            prepared.append({"prompt": prompt, "completion": completion})
        return Dataset.from_list(prepared)
    training_data, validation_data = prepare(train), prepare(validation)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    set_seed(config.seed)
    dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
    kwargs = {"revision": config.revision, "torch_dtype": dtype, "trust_remote_code": False,
              "device_map": {"": torch.cuda.current_device()}, "attn_implementation": "sdpa"}
    if config.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    base = AutoModelForCausalLM.from_pretrained(config.base_model, **kwargs)
    if config.load_in_4bit:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    else:
        base.enable_input_require_grads()
    base.config.use_cache = False
    lora = LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)
    arguments = SFTConfig(output_dir=str(output), num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation, learning_rate=config.learning_rate,
        max_length=config.max_length, completion_only_loss=True, assistant_only_loss=False,
        packing=False, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=config.precision == "bf16", fp16=config.precision == "fp16", optim="adamw_torch",
        eval_strategy="steps", eval_steps=config.eval_steps, save_strategy="steps", save_steps=config.eval_steps,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=5, report_to=[], seed=config.seed, data_seed=config.seed,
        eos_token=tokenizer.eos_token, pad_token=tokenizer.pad_token)
    trainer = SFTTrainer(model=model, args=arguments, train_dataset=training_data,
                         eval_dataset=validation_data, processing_class=tokenizer)
    result = trainer.train()
    adapter = output / "adapter"
    trainer.save_model(str(adapter)); tokenizer.save_pretrained(adapter); trainer.save_state()
    manifest = {"base_model": config.base_model, "revision": config.revision, "prompt_version": PROMPT_VERSION,
                "schema_seed_sha256": file_hash(ROOT / "resources/schema_seed.json"),
                "configuration": asdict(config), "training_rows": len(train), "validation_rows": len(validation),
                "data_sha256": {"training": file_hash(train_path), "validation": file_hash(validation_path), "evaluation_split_audit": file_hash(eval_path)},
                "checkpoint_selection": "Validation completion loss; evaluation questions excluded from optimization",
                "metrics": result.metrics}
    (adapter / "training_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest
