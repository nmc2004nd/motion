"""Config loader với validation nghiêm ngặt (không fallback im lặng)."""

from .loader import ConfigError, load_config, require, as_int_tuple

__all__ = ["ConfigError", "load_config", "require", "as_int_tuple"]
