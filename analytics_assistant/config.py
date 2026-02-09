from dataclasses import dataclass
from pathlib import Path
import os
import re
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
TABLES = {"CUSTOMER", "ORDERS", "LINEITEM", "PART", "PARTSUPP", "SUPPLIER", "NATION", "REGION"}


@dataclass(frozen=True)
class Config:
    account: str = os.getenv("SNOWFLAKE_ACCOUNT", "")
    user: str = os.getenv("SNOWFLAKE_USER", "")
    warehouse: str = os.getenv("SNOWFLAKE_WAREHOUSE", "")
    role: str = os.getenv("SNOWFLAKE_ROLE", "SAMPLE_READER")
    database: str = "SNOWFLAKE_SAMPLE_DATA"
    schema: str = "TPCH_SF1"
    model: str = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-Coder-7B-Instruct")
    revision: str = os.getenv("QWEN_REVISION", "c03e6d358207e414f1eca0bb1891e29f1db0e242")
    adapter_path: str = os.getenv("QWEN_ADAPTER_PATH", "")
    load_in_4bit: bool = os.getenv("QWEN_LOAD_4BIT", "0") == "1"
    max_new_tokens: int = 512
    max_context_tokens: int = 6144
    max_rows: int = 1000
    max_attempts: int = 3
    timeout_seconds: int = 30

    def validate(self):
        if not all((self.account, self.user, self.warehouse, self.role)):
            raise ValueError("Set Snowflake account, user, warehouse, and a restricted reader role")
        if not 1 <= self.max_attempts <= 10 or not 1 <= self.max_rows <= 10000 or not 1 <= self.timeout_seconds <= 300:
            raise ValueError("Invalid query, retry, or row limits")

        if not re.fullmatch(r"[0-9a-f]{40}", self.revision) or self.max_new_tokens < 1 or self.max_context_tokens <= self.max_new_tokens:
            raise ValueError("Pin a full checkpoint revision and valid generation budgets")

    def connection_arguments(self):
        self.validate()
        kwargs = {"account": self.account, "user": self.user, "warehouse": self.warehouse,
                  "role": self.role, "database": self.database, "schema": self.schema,
                  "login_timeout": 30, "network_timeout": self.timeout_seconds + 15,
                  "session_parameters": {"QUERY_TAG": "natural_language_analytics_assistant",
                    "STATEMENT_TIMEOUT_IN_SECONDS": self.timeout_seconds, "STATEMENT_QUEUED_TIMEOUT_IN_SECONDS": 15}}
        if os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE"):
            kwargs["private_key_file"] = os.environ["SNOWFLAKE_PRIVATE_KEY_FILE"]
            if os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSWORD"):
                kwargs["private_key_file_pwd"] = os.environ["SNOWFLAKE_PRIVATE_KEY_PASSWORD"]
        elif os.getenv("SNOWFLAKE_AUTHENTICATOR") == "externalbrowser":
            kwargs["authenticator"] = "externalbrowser"
        elif os.getenv("SNOWFLAKE_PASSWORD"):
            kwargs["password"] = os.environ["SNOWFLAKE_PASSWORD"]
        else:
            raise ValueError("Configure key-pair, external-browser, or supported password authentication")
        return kwargs
