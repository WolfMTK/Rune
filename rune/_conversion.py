import enum
from collections.abc import Iterable, Mapping
from functools import cache
from typing import Any

from rune import json, structs
from rune.struct import Struct

_BUILTIN_SCALARS = (str, int, float, bool)
_INVALID_BUILTIN_TYPES = "builtin_types must be an iterable of types"


def to_builtins(
    obj: Any,
    *,
    builtin_types: Iterable[type] | None = None,
    str_keys: bool = False,
    enc_hook: Any = None,
) -> Any:
    exact, subclassable = _builtin_types(builtin_types)
    return _Converter(exact, subclassable, str_keys, enc_hook).convert(obj)


@cache
def _passthrough_types() -> dict[type, bool]:
    import datetime
    import decimal
    import uuid

    return {
        bytes: False,
        bytearray: False,
        memoryview: False,
        datetime.datetime: False,
        datetime.date: False,
        datetime.time: False,
        datetime.timedelta: False,
        uuid.UUID: True,
        decimal.Decimal: False,
    }


def _builtin_types(types: Iterable[type] | None) -> tuple[frozenset[type], tuple[type, ...]]:
    if types is None:
        return frozenset(), ()
    try:
        items = tuple(types)
    except TypeError:
        raise TypeError(_INVALID_BUILTIN_TYPES) from None
    known = _passthrough_types()
    for item in items:
        if not isinstance(item, type):
            raise TypeError(_INVALID_BUILTIN_TYPES)
        if item not in known:
            raise TypeError(f"Cannot treat {item!r} as a builtin type")
    return frozenset(items), tuple(item for item in items if known[item])


class _Converter:
    __slots__ = ("builtin_types", "enc_hook", "str_keys", "subclassable_types")

    def __init__(
        self,
        builtin_types: frozenset[type],
        subclassable_types: tuple[type, ...],
        str_keys: bool,
        enc_hook: Any,
    ) -> None:
        self.builtin_types = builtin_types
        self.subclassable_types = subclassable_types
        self.str_keys = str_keys
        self.enc_hook = enc_hook

    def convert(self, obj: Any) -> Any:
        if isinstance(obj, enum.Enum):
            return self.convert(obj.value)
        if obj is None or isinstance(obj, _BUILTIN_SCALARS):
            return obj
        if self.builtin_types and self.passes_through(obj):
            return obj
        if isinstance(obj, Mapping):
            return {self.convert_key(key): self.convert(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple, set, frozenset)):
            return [self.convert(item) for item in obj]
        encoded = json.decode(json.encode(obj, enc_hook=self.enc_hook))
        if self.builtin_types:
            self.restore_fields(obj, encoded)
        return encoded

    def passes_through(self, obj: Any) -> bool:
        return type(obj) in self.builtin_types or isinstance(obj, self.subclassable_types)

    def restore_fields(self, obj: Any, encoded: Any) -> None:
        names = _encoded_field_names(obj)
        if isinstance(encoded, dict):
            for name, key in names:
                if key in encoded:
                    encoded[key] = self.convert(getattr(obj, name))
        elif isinstance(encoded, list):
            offset = 0 if getattr(type(obj), "__struct_tag__", None) is None else 1
            for index, (name, _) in enumerate(names[: len(encoded) - offset]):
                encoded[offset + index] = self.convert(getattr(obj, name))

    def convert_key(self, key: Any) -> Any:
        if isinstance(key, enum.Enum):
            key = key.value
        if isinstance(key, str) or (
            not self.str_keys and (key is None or isinstance(key, (int, float)))
        ):
            return key
        return next(iter(json.decode(json.encode({key: None}, enc_hook=self.enc_hook))))


def _encoded_field_names(obj: Any) -> list[tuple[str, str]]:
    cls = type(obj)
    if isinstance(obj, Struct):
        return [(info.name, info.encode_name) for info in structs.fields(cls)]
    if hasattr(cls, "__dataclass_fields__"):
        import dataclasses

        return [(field.name, field.name) for field in dataclasses.fields(cls)]
    attributes = getattr(cls, "__attrs_attrs__", None)
    if attributes is not None:
        return [(attribute.name, attribute.name) for attribute in attributes]
    return []
