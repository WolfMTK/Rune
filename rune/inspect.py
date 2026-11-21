import builtins
import collections.abc
import dataclasses
import datetime
import decimal
import enum
import types
import typing
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import (
    Annotated,
    Any,
    Final,
    Literal,
    NewType,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import rune
from rune.meta import Meta
from rune.struct import (
    NODEFAULT,
    StructMeta,
    _DefaultFactory,
    _is_struct_type,
    _struct_field_types,
    _substitute,
    _type_mapping,
    resolve_generic_fields,
    resolve_namedtuple_fields,
    resolve_type_alias,
)
from rune.unset import UNSET, UnsetType

__all__ = [
    "AnyType",
    "BoolType",
    "ByteArrayType",
    "BytesType",
    "CustomType",
    "DataclassType",
    "DateTimeType",
    "DateType",
    "DecimalType",
    "DictType",
    "EnumType",
    "ExtType",
    "Field",
    "FloatType",
    "FrozenDictType",
    "FrozenSetType",
    "IntType",
    "ListType",
    "LiteralType",
    "MemoryViewType",
    "Metadata",
    "NamedTupleType",
    "NoneType",
    "RawType",
    "SetType",
    "StrType",
    "StructType",
    "TimeDeltaType",
    "TimeType",
    "TupleType",
    "TypedDictType",
    "UUIDType",
    "UnionType",
    "VarTupleType",
    "is_struct",
    "is_struct_type",
    "multi_type_info",
    "type_info",
]


@dataclass(frozen=True)
class AnyType:
    pass


@dataclass(frozen=True)
class NoneType:
    pass


@dataclass(frozen=True)
class BoolType:
    pass


@dataclass(frozen=True)
class IntType:
    gt: int | float | None = None
    ge: int | float | None = None
    lt: int | float | None = None
    le: int | float | None = None
    multiple_of: int | float | None = None


@dataclass(frozen=True)
class FloatType:
    gt: int | float | None = None
    ge: int | float | None = None
    lt: int | float | None = None
    le: int | float | None = None
    multiple_of: int | float | None = None


@dataclass(frozen=True)
class StrType:
    pattern: str | None = None
    min_length: int | None = None
    max_length: int | None = None


@dataclass(frozen=True)
class BytesType:
    min_length: int | None = None
    max_length: int | None = None


@dataclass(frozen=True)
class ByteArrayType:
    min_length: int | None = None
    max_length: int | None = None


@dataclass(frozen=True)
class MemoryViewType:
    min_length: int | None = None
    max_length: int | None = None


@dataclass(frozen=True)
class DateTimeType:
    tz: bool | None = None


@dataclass(frozen=True)
class TimeType:
    tz: bool | None = None


@dataclass(frozen=True)
class DateType:
    pass


@dataclass(frozen=True)
class TimeDeltaType:
    pass


@dataclass(frozen=True)
class UUIDType:
    pass


@dataclass(frozen=True)
class DecimalType:
    pass


@dataclass(frozen=True)
class RawType:
    pass


@dataclass(frozen=True)
class ExtType:
    pass


@dataclass(frozen=True)
class ListType:
    item_type: Any
    min_length: int | None = None
    max_length: int | None = None


@dataclass(frozen=True)
class SetType:
    item_type: Any
    min_length: int | None = None
    max_length: int | None = None


@dataclass(frozen=True)
class FrozenSetType:
    item_type: Any
    min_length: int | None = None
    max_length: int | None = None


@dataclass(frozen=True)
class VarTupleType:
    item_type: Any
    min_length: int | None = None
    max_length: int | None = None


@dataclass(frozen=True)
class TupleType:
    item_types: tuple[Any, ...]


@dataclass(frozen=True)
class DictType:
    key_type: Any
    value_type: Any
    min_length: int | None = None
    max_length: int | None = None


@dataclass(frozen=True)
class FrozenDictType:
    key_type: Any
    value_type: Any
    min_length: int | None = None
    max_length: int | None = None


@dataclass(frozen=True)
class UnionType:
    types: tuple[Any, ...]
    includes_none: bool = False


@dataclass(frozen=True)
class LiteralType:
    values: tuple[Any, ...]


@dataclass(frozen=True)
class EnumType:
    cls: type


@dataclass(frozen=True)
class CustomType:
    cls: Any


@dataclass(frozen=True)
class Field:
    name: str
    encode_name: str
    type: Any
    required: bool = True
    default: Any = NODEFAULT
    default_factory: Any = NODEFAULT


@dataclass(frozen=True)
class StructType:
    cls: Any
    fields: tuple[Field, ...]
    tag_field: str | None = None
    tag: str | int | None = None
    array_like: bool = False
    forbid_unknown_fields: bool = False


@dataclass(frozen=True)
class NamedTupleType:
    cls: Any
    fields: tuple[Field, ...]


@dataclass(frozen=True)
class TypedDictType:
    cls: Any
    fields: tuple[Field, ...]


@dataclass(frozen=True)
class DataclassType:
    cls: Any
    fields: tuple[Field, ...]


@dataclass(frozen=True)
class Metadata:
    type: Any
    extra_json_schema: dict[str, Any] | None = None
    extra: dict[str, Any] | None = None


_NONE_TYPE = type(None)
_TypeAliasType = getattr(typing, "TypeAliasType", None)
_FROZENDICT = getattr(builtins, "frozendict", None)

_SCALARS: dict[Any, Any] = {
    bool: BoolType,
    int: IntType,
    float: FloatType,
    str: StrType,
    bytes: BytesType,
    bytearray: ByteArrayType,
    memoryview: MemoryViewType,
    datetime.datetime: DateTimeType,
    datetime.time: TimeType,
    datetime.date: DateType,
    datetime.timedelta: TimeDeltaType,
    uuid.UUID: UUIDType,
    decimal.Decimal: DecimalType,
    rune.Raw: RawType,
    rune.msgpack.Ext: ExtType,
}

_SEQUENCES: dict[Any, Any] = {
    list: ListType,
    collections.abc.Collection: ListType,
    collections.abc.Sequence: ListType,
    collections.abc.MutableSequence: ListType,
    set: SetType,
    collections.abc.Set: SetType,
    collections.abc.MutableSet: SetType,
    frozenset: FrozenSetType,
}

_MAPPINGS: dict[Any, Any] = {
    dict: DictType,
    collections.abc.Mapping: DictType,
    collections.abc.MutableMapping: DictType,
}
if _FROZENDICT is not None:
    _MAPPINGS[_FROZENDICT] = FrozenDictType

_QUALIFIERS = frozenset(
    qualifier
    for qualifier in (
        Final,
        getattr(typing, "Required", None),
        getattr(typing, "NotRequired", None),
        getattr(typing, "ReadOnly", None),
    )
    if qualifier is not None
)

_NUMERIC_NAMES = ("gt", "ge", "lt", "le", "multiple_of")
_LENGTH_NAMES = ("min_length", "max_length")
_CONSTRAINT_NAMES = (*_NUMERIC_NAMES, "pattern", *_LENGTH_NAMES, "tz")
_DESCRIPTION_NAMES = ("title", "description", "examples")
_CONSTRAINTS: dict[Any, frozenset[str]] = {
    IntType: frozenset(_NUMERIC_NAMES),
    FloatType: frozenset(_NUMERIC_NAMES),
    StrType: frozenset(("pattern", *_LENGTH_NAMES)),
    BytesType: frozenset(_LENGTH_NAMES),
    ByteArrayType: frozenset(_LENGTH_NAMES),
    MemoryViewType: frozenset(_LENGTH_NAMES),
    ListType: frozenset(_LENGTH_NAMES),
    SetType: frozenset(_LENGTH_NAMES),
    FrozenSetType: frozenset(_LENGTH_NAMES),
    VarTupleType: frozenset(_LENGTH_NAMES),
    DictType: frozenset(_LENGTH_NAMES),
    FrozenDictType: frozenset(_LENGTH_NAMES),
    DateTimeType: frozenset(("tz",)),
    TimeType: frozenset(("tz",)),
}


class _Builder:
    def __init__(self) -> None:
        self.memo: dict[Any, Any] = {}

    def build(self, tp: Any) -> Any:
        if tp is Any or tp is object or tp is UnsetType:
            return AnyType()
        if tp is None or tp is _NONE_TYPE:
            return NoneType()
        if isinstance(tp, TypeVar):
            return self._type_var(tp)
        if isinstance(tp, NewType):
            return self.build(tp.__supertype__)
        origin = get_origin(tp)
        alias = tp if origin is None else origin
        if _TypeAliasType is not None and isinstance(alias, _TypeAliasType):
            return self.build(resolve_type_alias(tp))
        if origin is Annotated:
            base, *metadata = get_args(tp)
            return _apply_metadata(self.build(base), metadata)
        if origin in _QUALIFIERS:
            return self.build(get_args(tp)[0])
        if origin is Union or origin is types.UnionType:
            return self._union(get_args(tp))
        if origin is Literal:
            return LiteralType(_sort_literals(get_args(tp)))
        base = alias
        if not isinstance(base, type):
            raise TypeError(f"Unsupported type: {tp!r}")
        return self._class(tp, base, get_args(tp))

    def _class(self, tp: Any, base: type, args: tuple[Any, ...]) -> Any:
        if base is tuple:
            return self._tuple(tp, args)
        if base in _SEQUENCES:
            return _SEQUENCES[base](self._item(args))
        if base in _MAPPINGS:
            key, value = self._pair(args)
            return _MAPPINGS[base](key, value)
        if base in _SCALARS:
            return _SCALARS[base]()
        if issubclass(base, enum.Enum):
            return EnumType(base)
        if _is_struct_type(base):
            return self._struct(tp, base)
        if issubclass(base, tuple) and hasattr(base, "_fields"):
            return self._namedtuple(tp, base)
        if issubclass(base, dict) and hasattr(base, "__required_keys__"):
            return self._typeddict(tp, base)
        if dataclasses.is_dataclass(base):
            return self._dataclass(tp, base)
        if hasattr(base, "__attrs_attrs__"):
            return self._attrs(tp, base)
        return CustomType(tp)

    def _tuple(self, tp: Any, args: tuple[Any, ...]) -> Any:
        if not hasattr(tp, "__args__"):
            return VarTupleType(AnyType())
        if not args:
            return TupleType(())
        if len(args) == 2 and args[1] is Ellipsis:
            return VarTupleType(self.build(args[0]))
        return TupleType(tuple(self.build(arg) for arg in args))

    def _item(self, args: tuple[Any, ...]) -> Any:
        if not args:
            return AnyType()
        if len(args) != 1:
            raise TypeError(f"Expected 1 type argument, got {len(args)}")
        return self.build(args[0])

    def _pair(self, args: tuple[Any, ...]) -> tuple[Any, Any]:
        if not args:
            return AnyType(), AnyType()
        if len(args) != 2:
            raise TypeError(f"Expected 2 type arguments, got {len(args)}")
        return self.build(args[0]), self.build(args[1])

    def _union(self, args: tuple[Any, ...]) -> Any:
        includes_none = False
        members: list[Any] = []
        for arg in args:
            if arg is None or arg is _NONE_TYPE:
                includes_none = True
            elif arg is not UnsetType:
                members.append(self.build(arg))
        if not members:
            return NoneType() if includes_none else AnyType()
        if len(members) == 1 and not includes_none:
            return members[0]
        return UnionType(tuple(members), includes_none)

    def _type_var(self, tp: TypeVar) -> Any:
        if tp.__bound__ is not None:
            return self.build(tp.__bound__)
        if tp.__constraints__:
            return self._union(tp.__constraints__)
        return AnyType()

    def _field(
        self,
        name: str,
        encode_name: str,
        annotation: Any,
        default: Any,
        default_factory: Any,
    ) -> Field:
        if default is UNSET:
            default = NODEFAULT
        required = (
            default is NODEFAULT and default_factory is NODEFAULT and not _allows_unset(annotation)
        )
        return Field(name, encode_name, self.build(annotation), required, default, default_factory)

    def _struct(self, tp: Any, origin: StructMeta) -> Any:
        cached = self.memo.get(tp)
        if cached is not None:
            return cached
        config = origin.__struct_config__
        info = StructType(
            tp,
            (),
            config.tag_field,
            config.tag,
            config.array_like,
            config.forbid_unknown_fields,
        )
        self.memo[tp] = info
        raw_fields = _struct_field_types(origin) if tp is origin else resolve_generic_fields(tp)
        encode_names = origin.__struct_encode_fields__
        defaults = origin.__struct_defaults__
        first_default = len(raw_fields) - len(defaults)
        fields = []
        for index, (name, annotation) in enumerate(raw_fields):
            default = defaults[index - first_default] if index >= first_default else NODEFAULT
            factory = NODEFAULT
            if type(default) is _DefaultFactory:
                default, factory = NODEFAULT, default.factory
            fields.append(self._field(name, encode_names[index], annotation, default, factory))
        object.__setattr__(info, "fields", tuple(fields))
        return info

    def _namedtuple(self, tp: Any, origin: Any) -> Any:
        cached = self.memo.get(tp)
        if cached is not None:
            return cached
        info = NamedTupleType(tp, ())
        self.memo[tp] = info
        defaults = origin._field_defaults
        fields = tuple(
            self._field(name, name, annotation, defaults.get(name, NODEFAULT), NODEFAULT)
            for name, annotation in resolve_namedtuple_fields(tp)
        )
        object.__setattr__(info, "fields", fields)
        return info

    def _typeddict(self, tp: Any, origin: Any) -> Any:
        cached = self.memo.get(tp)
        if cached is not None:
            return cached
        info = TypedDictType(tp, ())
        self.memo[tp] = info
        required = origin.__required_keys__
        mapping = _mapping_for(tp, origin)
        hints = get_type_hints(origin, include_extras=True)
        fields = tuple(
            Field(name, name, self.build(_substitute(hint, mapping)), name in required)
            for name, hint in hints.items()
        )
        object.__setattr__(info, "fields", fields)
        return info

    def _dataclass(self, tp: Any, origin: type) -> Any:
        cached = self.memo.get(tp)
        if cached is not None:
            return cached
        info = DataclassType(tp, ())
        self.memo[tp] = info
        hints = get_type_hints(origin, include_extras=True)
        mapping = _mapping_for(tp, origin)
        fields = []
        for f in dataclasses.fields(origin):
            default = NODEFAULT if f.default is dataclasses.MISSING else f.default
            factory = NODEFAULT if f.default_factory is dataclasses.MISSING else f.default_factory
            annotation = _substitute(hints.get(f.name, f.type), mapping)
            fields.append(self._field(f.name, f.name, annotation, default, factory))
        object.__setattr__(info, "fields", tuple(fields))
        return info

    def _attrs(self, tp: Any, origin: Any) -> Any:
        import attrs

        cached = self.memo.get(tp)
        if cached is not None:
            return cached
        info = DataclassType(tp, ())
        self.memo[tp] = info
        hints = get_type_hints(origin, include_extras=True)
        mapping = _mapping_for(tp, origin)
        fields = []
        for attribute in origin.__attrs_attrs__:
            default = attribute.default
            factory = NODEFAULT
            if default is attrs.NOTHING:
                default = NODEFAULT
            elif isinstance(default, attrs.Factory):
                default, factory = NODEFAULT, default.factory
            annotation = _substitute(hints.get(attribute.name, Any), mapping)
            fields.append(self._field(attribute.name, attribute.name, annotation, default, factory))
        object.__setattr__(info, "fields", tuple(fields))
        return info


def _mapping_for(tp: Any, origin: type) -> dict[Any, Any]:
    return {} if tp is origin else _type_mapping(tp, origin)


def _allows_unset(tp: Any) -> bool:
    origin = get_origin(tp)
    if origin is Annotated or origin in _QUALIFIERS:
        return _allows_unset(get_args(tp)[0])
    if origin is Union or origin is types.UnionType:
        return any(_allows_unset(arg) for arg in get_args(tp))
    return tp is UnsetType


def _specified(item: Meta, names: tuple[str, ...]) -> dict[str, Any]:
    values = ((name, getattr(item, name)) for name in names)
    return {name: value for name, value in values if value is not None}


def _apply_metadata(info: Any, metadata: list[Any]) -> Any:
    constraints: dict[str, Any] = {}
    schema: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for item in metadata:
        if isinstance(item, Meta):
            constraints.update(_specified(item, _CONSTRAINT_NAMES))
            described = _specified(item, _DESCRIPTION_NAMES)
            schema = _merge_json(schema, _merge_json(described, item.extra_json_schema or {}))
            extra = _merge_json(extra, item.extra or {})
    base = info
    if isinstance(info, Metadata):
        base = info.type
        schema = _merge_json(info.extra_json_schema or {}, schema)
        extra = _merge_json(info.extra or {}, extra)
    if constraints:
        base = _apply_constraints(base, constraints)
    if schema or extra:
        return Metadata(base, schema or None, extra or None)
    return base


def _apply_constraints(info: Any, constraints: dict[str, Any]) -> Any:
    if isinstance(info, Metadata):
        return replace(info, type=_apply_constraints(info.type, constraints))
    allowed = _CONSTRAINTS.get(type(info), frozenset())
    unsupported = [name for name in constraints if name not in allowed]
    if unsupported:
        names = ", ".join(f"`{name}`" for name in unsupported)
        raise TypeError(f"{names} cannot be applied to {type(info).__name__}")
    return replace(info, **constraints)


def _merge_json(a: Any, b: Any) -> Any:
    if isinstance(a, dict) and isinstance(b, dict):
        merged = dict(a)
        for key, value in b.items():
            merged[key] = _merge_json(merged[key], value) if key in merged else value
        return merged
    if isinstance(a, list) and isinstance(b, list):
        return [*a, *b]
    return b


def _sort_literals(values: tuple[Any, ...]) -> tuple[Any, ...]:
    try:
        return tuple(sorted(values, key=_literal_sort_key))
    except TypeError:
        return values


def _literal_sort_key(value: Any) -> tuple[str, Any]:
    return type(value).__qualname__, value


def type_info(tp: Any) -> Any:
    return _Builder().build(tp)


def multi_type_info(tps: Iterable[Any]) -> tuple[Any, ...]:
    builder = _Builder()
    return tuple(builder.build(tp) for tp in tps)


def is_struct(obj: Any) -> bool:
    return _is_struct_type(type(obj))


def is_struct_type(tp: Any) -> bool:
    return _is_struct_type(tp)


def __dir__() -> list[str]:
    return list(__all__)
