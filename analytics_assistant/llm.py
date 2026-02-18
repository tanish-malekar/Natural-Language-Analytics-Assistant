from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import json
import re
import uuid
from pydantic import BaseModel, ConfigDict
from .prompts import PROMPT_VERSION, messages


class ChartSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["none", "auto", "bar", "line", "scatter"]
    x_column: str | None
    y_column: str | None
    title: str


class SQLPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sql: str
    explanation: str
    chart: ChartSpec


@dataclass
class Generation:
    plan: SQLPlan
    generation_id: str
    model: str
    input_tokens: int
    output_tokens: int


class ModelError(RuntimeError):
    pass


def extract_sql(text):
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:sql)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return fence.group(1).strip() if fence else stripped


class QwenSQLModel:
    def __init__(self, config, require_adapter=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        self.config, self.torch = config, torch
        self.has_adapter = bool(config.adapter_path)
        if require_adapter and not self.has_adapter:
            raise ValueError("SFT mode requires an explicit trained adapter path")
        self.manifest = None
        if self.has_adapter:
            adapter = Path(config.adapter_path)
            self.manifest = json.loads((adapter / "training_manifest.json").read_text())
            if (self.manifest["base_model"], self.manifest["revision"]) != (config.model, config.revision):
                raise ValueError("Adapter and base checkpoint identities do not match")
            if self.manifest.get("prompt_version") != PROMPT_VERSION:
                raise ValueError("Adapter prompt version does not match inference")
        if not config.model.startswith("Qwen/Qwen2.5-Coder-"):
            raise ValueError("This pipeline targets the Qwen2.5-Coder model family")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
        kwargs = {"revision": config.revision, "torch_dtype": dtype, "device_map": "auto", "trust_remote_code": False,
                  "attn_implementation": "sdpa"}
        if config.load_in_4bit:
            if not torch.cuda.is_available():
                raise ValueError("The optional 4-bit path requires a compatible CUDA device")
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(config.model, **kwargs)
        self.model = PeftModel.from_pretrained(base, config.adapter_path, is_trainable=False) if self.has_adapter else base
        self.model.eval(); self.model.config.use_cache = True

    def generate(self, question, context, previous_sql=None, error=None, mode="sft"):
        if mode not in {"base", "sft"} or (mode == "sft" and not self.has_adapter):
            raise ModelError("SFT mode requires a trained adapter; base mode disables it")
        prompt = self.tokenizer.apply_chat_template(messages(question, context, previous_sql, error), tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        size = encoded.input_ids.shape[1]
        if size + self.config.max_new_tokens > self.config.max_context_tokens:
            raise ModelError("Retrieved prompt exceeds the configured context budget")
        device = self.model.get_input_embeddings().weight.device
        encoded = encoded.to(device)
        disabled = self.model.disable_adapter() if mode == "base" and self.has_adapter else nullcontext()
        try:
            with disabled, self.torch.inference_mode():
                output = self.model.generate(**encoded, do_sample=False, max_new_tokens=self.config.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            generated = output[0, size:]
            sql = extract_sql(self.tokenizer.decode(generated, skip_special_tokens=True))
        except RuntimeError as exc:
            raise ModelError(f"Local Qwen generation failed ({type(exc).__name__})") from None
        plan = SQLPlan(sql=sql, explanation=f"SQL generated locally by the {mode} Qwen checkpoint.",
                       chart=ChartSpec(kind="auto", x_column=None, y_column=None, title=question[:200]))
        return Generation(plan, uuid.uuid4().hex, f"{self.config.model}@{self.config.revision}:{mode}", size, len(generated))
