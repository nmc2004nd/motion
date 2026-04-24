"""Đọc và validate YAML cấu hình.

Triết lý: YAML là *single source of truth*. Code không có fallback mặc định —
thiếu key bắt buộc sẽ raise `ConfigError` ngay khi load, không âm thầm chạy
với giá trị lạ. Nhờ vậy khi đổi cấu hình, ta biết chính xác mọi nơi bị ảnh hưởng.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

from .schema import REQUIRED_KEYS


class ConfigError(KeyError):
    """Lỗi thiếu/ sai kiểu config."""


def load_config(config_path: str) -> dict:
    """Load YAML + validate toàn bộ required keys.

    Args:
        config_path: đường dẫn tới file YAML.

    Returns:
        dict cấu hình đã validate.

    Raises:
        FileNotFoundError: nếu file không tồn tại.
        ConfigError: nếu thiếu key bắt buộc.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    _validate(config)
    return config


def require(config: dict, key_path: str) -> Any:
    """Truy cập theo dot-notation, raise nếu thiếu.

    Khác với `dict.get(...)`, hàm này KHÔNG trả về giá trị mặc định — thiếu key
    là lỗi config và phải được khắc phục ở YAML.
    """
    keys = key_path.split(".")
    value: Any = config
    for k in keys:
        if not isinstance(value, dict) or k not in value:
            raise ConfigError(f"Missing required config key: {key_path!r}")
        value = value[k]
    return value


def as_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    """Ép list/tuple sang tuple int, báo lỗi rõ nếu kiểu sai."""
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{name} must be a list/tuple, got {type(value).__name__}")
    return tuple(int(v) for v in value)


def _validate(config: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if not _has(config, k)]
    if missing:
        raise ConfigError(
            "Config thiếu các key bắt buộc:\n  - "
            + "\n  - ".join(missing)
        )


def _has(config: dict, key_path: str) -> bool:
    value: Any = config
    for k in key_path.split("."):
        if not isinstance(value, dict) or k not in value:
            return False
        value = value[k]
    return True
