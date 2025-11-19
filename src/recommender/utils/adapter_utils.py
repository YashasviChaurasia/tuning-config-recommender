import json
import yaml
import re
from pathlib import Path
from typing import Any, Dict, List

DYNAMIC_PATTERN = re.compile(r"^\$\{([A-Za-z0-9_]+)\}$")

def extract_runtime_fields(data_preprocessor: dict):
    runtime = {}

    for key in ["_dataset_text_field", "_response_template", "_conversation_column_name"]:
        if key in data_preprocessor:
            runtime[key.lstrip("_")] = data_preprocessor[key]
            data_preprocessor.pop(key)

    return runtime

def safe_serialize(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [safe_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    return str(obj)


def write_yaml_preserving_templates(obj, path):
    def strip_private(o):
        if isinstance(o, dict):
            return {k: strip_private(v) for k, v in o.items() if not k.startswith("_")}
        if isinstance(o, list):
            return [strip_private(x) for x in o]
        return o

    cleaned = strip_private(obj)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            cleaned,
            f,
            sort_keys=False,
            allow_unicode=True,
            width=10000,
        )


def split_static_and_dynamic(cfg: dict):
    static, dynamic = {}, []

    for k, v in cfg.items():
        if isinstance(v, str) and (m := DYNAMIC_PATTERN.match(v)):
            dynamic.append(f"--{k} \"${{{m.group(1)}}}\"")
        else:
            static[k] = v

    return static, dynamic


def fmt_cli_value(v):
    if isinstance(v, bool):
        return "'true'" if v else "'false'"
    if isinstance(v, (int, float)):
        return f"'{v}'"
    if isinstance(v, dict):
        return f"'{json.dumps(v)}'"
    if isinstance(v, (list, tuple)):
        return " ".join(f"'{str(x).lower()}'" for x in v)
    return f"'{v}'"


def prepare_ir_for_accelerate(ir: dict):
    static_dist, dynamic = split_static_and_dynamic(ir.get("dist_config", {}))
    ir["dist_config"] = static_dist
    return ir, dynamic


def build_launch_command(
    ir: Dict[str, Any],
    data_config: Path,
    accel_config: Path,
    dynamic_args: List[str] = None,
    runtime_args: Dict[str, Any] = None,
) -> str:

    cmd = [
        "accelerate launch",
        f"--config_file '{accel_config}'",
        *(dynamic_args or []),
        "-m 'tuning.sft_trainer'",
    ]

    for k, v in ir.get("train_config", {}).items():
        if v is not None and k != "training_data_path":
            cmd.append(f"--{k} {fmt_cli_value(v)}")

    if runtime_args:
        for k, v in runtime_args.items():
            cmd.append(f"--{k} {fmt_cli_value(v)}")

    cmd.append(f"--data_config '{data_config}'")

    return " \\\n".join(cmd)
