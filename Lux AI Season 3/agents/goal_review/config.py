from pathlib import Path

import yaml


def merge(left, right):
    result = dict(left)
    for key, value in right.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path):
    path = Path(path)
    current = yaml.safe_load(path.read_text())
    base = current.pop('extends', None)
    return merge(load_config(path.parent / base), current) if base else current
