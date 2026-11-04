import enum
from collections.abc import Mapping
from typing import Any

from rune import json

_BUILTIN_SCALARS = (str, int, float, bool)


def to_builtins(obj: Any, *, str_keys: bool = False, enc_hook: Any = None) -> Any:
    return _convert(obj, str_keys, enc_hook)


def _convert(obj: Any, str_keys: bool, enc_hook: Any) -> Any:
    if isinstance(obj, enum.Enum):
        return _convert(obj.value, str_keys, enc_hook)
    if obj is None or isinstance(obj, _BUILTIN_SCALARS):
        return obj
    if isinstance(obj, Mapping):
        return {
            _convert_key(key, str_keys, enc_hook): _convert(value, str_keys, enc_hook)
            for key, value in obj.items()
        }
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_convert(item, str_keys, enc_hook) for item in obj]
    return json.decode(json.encode(obj, enc_hook=enc_hook))


def _convert_key(key: Any, str_keys: bool, enc_hook: Any) -> Any:
    if isinstance(key, enum.Enum):
        key = key.value
    if isinstance(key, str) or (not str_keys and (key is None or isinstance(key, (int, float)))):
        return key
    return next(iter(json.decode(json.encode({key: None}, enc_hook=enc_hook))))
