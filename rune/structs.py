from typing import Any, get_origin

from rune.struct import (
    _RENAMED_FLAG,
    NODEFAULT,
    Struct,
    _DefaultFactory,
    _is_struct_type,
    resolve_generic_fields,
)

__all__ = ["FieldInfo", "fields"]


class FieldInfo(Struct):
    name: str
    encode_name: str
    type: Any
    default: Any = NODEFAULT
    default_factory: Any = NODEFAULT

    @property
    def required(self) -> bool:
        return self.default is NODEFAULT and self.default_factory is NODEFAULT


def fields(type_or_instance: Any) -> tuple[FieldInfo, ...]:
    origin = get_origin(type_or_instance)
    if origin is not None:
        cls = origin
    elif isinstance(type_or_instance, type):
        cls = type_or_instance
    else:
        cls = type(type_or_instance)
    if not _is_struct_type(cls):
        raise TypeError(f"fields() expects a Struct type or instance, got {type_or_instance!r}")
    raw_fields = (
        cls.__struct_fields__ if origin is None else resolve_generic_fields(type_or_instance)
    )
    rename = cls.__struct_rename__ if cls.__struct_flags__ & _RENAMED_FLAG else None
    infos = []
    for name, tp in raw_fields:
        default = getattr(cls, name, NODEFAULT)
        factory = NODEFAULT
        if type(default) is _DefaultFactory:
            default, factory = NODEFAULT, default.factory
        encode_name = rename(name) if rename is not None else name
        infos.append(FieldInfo(name, encode_name, tp, default, factory))
    return tuple(infos)
