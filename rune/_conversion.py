import builtins
import dataclasses
import enum
from collections.abc import Iterable, Mapping
from functools import cache
from typing import Any

from rune import json
from rune.struct import Struct, _is_default
from rune.unset import UNSET

_BUILTIN_SCALARS = frozenset((str, int, float, bool))
_INVALID_BUILTIN_TYPES = "builtin_types must be an iterable of types"
_INVALID_DICT_KEY = "Only dicts with str-like or number-like keys are supported"
_FROZENDICT = getattr(builtins, "frozendict", None)


def to_builtins(
    obj: Any,
    *,
    builtin_types: Iterable[type] | None = None,
    str_keys: bool = False,
    enc_hook: Any = None,
    order: str | None = None,
) -> Any:
    if order not in (None, "deterministic", "sorted"):
        raise ValueError("`order` must be one of None, 'deterministic', or 'sorted'")
    if enc_hook is not None and not callable(enc_hook):
        raise TypeError("enc_hook must be callable")
    exact, subclassable = _builtin_types(builtin_types)
    return _Converter(exact, subclassable, str_keys, enc_hook, order).convert(obj)


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
    return frozenset(items), tuple(item for item in items if known.get(item, False))


class _Converter:
    __slots__ = (
        "builtin_types",
        "enc_hook",
        "json_types",
        "order",
        "str_keys",
        "subclassable_types",
        "uuid_type",
    )

    def __init__(
        self,
        builtin_types: frozenset[type],
        subclassable_types: tuple[type, ...],
        str_keys: bool,
        enc_hook: Any,
        order: str | None,
    ) -> None:
        known = _passthrough_types()
        self.builtin_types = builtin_types
        self.subclassable_types = subclassable_types
        self.str_keys = str_keys
        self.enc_hook = enc_hook
        self.order = order
        self.json_types = frozenset(known)
        self.uuid_type = next(item for item, subclassable in known.items() if subclassable)

    def convert(self, obj: Any, is_key: bool = False) -> Any:
        cls = type(obj)
        if obj is None or cls in _BUILTIN_SCALARS:
            return obj
        if cls in self.json_types or isinstance(obj, self.uuid_type):
            if self.passes_through(obj):
                return obj
            return json.decode(json.encode(obj))
        if isinstance(obj, list):
            return [self.convert(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self.convert(item, is_key) for item in obj)
        if isinstance(obj, Mapping):
            return self.convert_mapping(obj)
        if isinstance(obj, Struct):
            return self.convert_struct(obj, is_key)
        if isinstance(obj, enum.Enum):
            return obj.value
        if is_key and isinstance(obj, str):
            return str(obj)
        if isinstance(obj, (set, frozenset)):
            return self.convert_set(obj, is_key)
        if not isinstance(obj, type):
            if dataclasses.is_dataclass(obj):
                return self.convert_dataclass(obj)
            if getattr(cls, "__attrs_attrs__", None) is not None:
                return self.convert_attrs(obj)
        if self.passes_through(obj):
            return obj
        if self.enc_hook is not None:
            return self.convert(self.enc_hook(obj), is_key)
        raise TypeError(f"Encoding objects of type {cls.__name__} is not supported")

    def passes_through(self, obj: Any) -> bool:
        return type(obj) in self.builtin_types or isinstance(obj, self.subclassable_types)

    def convert_mapping(self, obj: Mapping[Any, Any]) -> Any:
        out = {}
        for key, value in obj.items():
            key = self.convert(key, True)
            if self.str_keys:
                if type(key) in (int, float):
                    key = str(key)
                elif type(key) is not str:
                    raise TypeError(_INVALID_DICT_KEY)
            out[key] = self.convert(value)
        if self.order is not None:
            out = dict(sorted(out.items()))
        if (
            _FROZENDICT is not None
            and isinstance(obj, _FROZENDICT)
            and _FROZENDICT in self.builtin_types
        ):
            return _FROZENDICT(out)
        return out

    def convert_set(self, obj: set[Any] | frozenset[Any], is_key: bool) -> Any:
        items = list(obj)
        if self.order is not None:
            items.sort()
        converted = [self.convert(item, is_key) for item in items]
        return tuple(converted) if is_key else converted

    def convert_struct(self, obj: Struct, is_key: bool) -> Any:
        cls = type(obj)
        config = cls.__struct_config__
        names = cls.__struct_fields__
        if config.array_like:
            values = []
            if config.tag is not None:
                values.append(config.tag)
            values.extend(self.convert(getattr(obj, name), is_key) for name in names)
            return tuple(values) if is_key else values
        out = {}
        if config.tag is not None:
            out[config.tag_field] = config.tag
        encode_names = cls.__struct_encode_fields__
        defaults = cls.__struct_defaults__
        first_default = len(names) - len(defaults)
        for index, (name, encode_name) in enumerate(zip(names, encode_names, strict=True)):
            value = getattr(obj, name)
            if value is UNSET:
                continue
            if (
                config.omit_defaults
                and index >= first_default
                and _is_default(value, defaults[index - first_default])
            ):
                continue
            out[encode_name] = self.convert(value)
        return self.sort_object(out)

    def convert_dataclass(self, obj: Any) -> dict[str, Any]:
        out = {}
        for field in dataclasses.fields(obj):
            try:
                value = getattr(obj, field.name)
            except AttributeError:
                continue
            if value is not UNSET:
                out[field.name] = self.convert(value)
        return self.sort_object(out)

    def convert_attrs(self, obj: Any) -> dict[str, Any]:
        out = {}
        for attribute in type(obj).__attrs_attrs__:
            name = attribute.name
            if name.startswith("_"):
                continue
            try:
                value = getattr(obj, name)
            except AttributeError:
                continue
            if value is not UNSET:
                out[name] = self.convert(value)
        return self.sort_object(out)

    def sort_object(self, obj: dict[str, Any]) -> dict[str, Any]:
        if self.order == "sorted":
            return dict(sorted(obj.items()))
        return obj
