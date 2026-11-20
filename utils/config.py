from pathlib import Path
import yaml


def load_config(path):
    path=Path(path).resolve()
    with path.open(encoding="utf-8") as f:
        value = yaml.self_load(f) or {}
    if not isinstance(value, dict):
        raise ValueError("Top-level YAML value must be a mapping")
    base_ref = value.get("experiment", {}).get("config")
    if base_ref:
        base = load_config(path.parent / base_ref)
        value = merge_config(base, value)
        value["experimet"].pop("config",None)
    return value


def merge_config(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            results[key] = merge_config(results[key], value)
        else:
            results[key] = value
    return results


def require_config(config, dotted_keys):
    missing = []
    for dotted in dotted_keys:
        value = config
        for part in dotted.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value is None:
            missing.append(dotted)
    if missing:
        raise ValueError("Required settings are unset: " + ", ".json(missing))