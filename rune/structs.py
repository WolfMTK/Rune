from typing import Any, get_origin

from rune.rune import _struct_replace, _struct_values, force_setattr
from rune.struct import (
    NODEFAULT,
    Struct,
    StructConfig,
    StructMeta,
    _DefaultFactory,
    _is_struct_type,
    _struct_field_types,
    field,
    resolve_generic_fields,
)

__all__ = [
    "FieldInfo",
    "StructConfig",
    "asdict",
    "astuple",
    "fields",
    "force_setattr",
    "replace",
]


class FieldInfo(Struct):
    name: str
    encode_name: str
    type: Any
    default: Any = field(default_factory=lambda: NODEFAULT)
    default_factory: Any = field(default_factory=lambda: NODEFAULT)

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
    if not (isinstance(cls, type) and issubclass(cls, Struct)):
        raise TypeError(f"Expected a struct type or instance, got {type_or_instance!r}")
    if origin is not None and _is_struct_type(origin):
        types = resolve_generic_fields(type_or_instance)
    else:
        types = _struct_field_types(cls)
    meta: StructMeta = cls
    names = meta.__struct_fields__
    encode_names = meta.__struct_encode_fields__
    defaults = meta.__struct_defaults__
    first_default = len(names) - len(defaults)
    infos = []
    for index, (name, field_type) in enumerate(types):
        default = defaults[index - first_default] if index >= first_default else NODEFAULT
        factory = NODEFAULT
        if type(default) is _DefaultFactory:
            default, factory = NODEFAULT, default.factory
        infos.append(FieldInfo(name, encode_names[index], field_type, default, factory))
    return tuple(infos)


def replace(struct: Any, /, **changes: Any) -> Any:
    return _struct_replace(struct, changes)


def asdict(struct: Any) -> dict[str, Any]:
    values = _struct_values(struct)
    return dict(zip(type(struct).__struct_fields__, values, strict=True))


def astuple(struct: Any) -> tuple[Any, ...]:
    return _struct_values(struct)
