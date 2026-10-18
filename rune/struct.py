import inspect
from typing import Any, ClassVar, TypeVar, get_args, get_origin

Fields = tuple[tuple[str, Any], ...]
TypeMapping = dict[Any, Any]

_MISSING: Any = object()


class StructMeta(type):
    pass


class _LazyFields:
    def __get__(self, obj: Any, owner: type) -> Fields:
        fields = tuple(_fields(owner, {}).items())
        owner.__struct_fields__ = fields
        return fields


class Struct(metaclass=StructMeta):
    __struct_fields__: ClassVar[Fields] = ()
    __struct_array_like__: ClassVar[bool] = False
    __struct_tag__: ClassVar[str | int | None] = None
    __struct_tag_field__: ClassVar[str] = "type"

    def __init_subclass__(
        cls,
        tag: str | int | bool | None = None,
        tag_field: str = "type",
        array_like: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        cls.__struct_fields__ = _LazyFields()
        cls.__struct_array_like__ = array_like
        cls.__struct_tag_field__ = tag_field
        cls.__struct_tag__ = _resolve_tag(cls, tag)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        cls = type(self)
        if args:
            kwargs = _merge_positional_args(cls, args, kwargs)
        for name, _ in cls.__struct_fields__:
            value = kwargs.pop(name, _MISSING)
            if value is _MISSING:
                value = getattr(cls, name, _MISSING)
            if value is _MISSING:
                raise TypeError(f"Missing required field `{name}`")
            setattr(self, name, value)

        if kwargs:
            unknown = ", ".join(f"`{name}`" for name in kwargs)
            raise TypeError(f"Unexpected field(s): {unknown}")

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return all(
            getattr(self, name) == getattr(other, name) for name, _ in self.__struct_fields__
        )

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={getattr(self, name)!r}" for name, _ in self.__struct_fields__)
        return f"{type(self).__name__}({fields})"


def _resolve_tag(cls: type, tag: Any) -> str | int | None:
    if tag is True:
        return cls.__name__
    elif tag is None or tag is False:
        return None
    elif isinstance(tag, (str, int)):
        return tag
    raise TypeError(f"Struct tag must be a str, an int, True or False, got {tag!r}")


def _merge_positional_args(
    cls: type, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    fields = cls.__struct_fields__
    if len(args) > len(fields):
        raise TypeError(
            f"{cls.__name__}() takes at most {len(fields)} positional "
            f"argument(s) but {len(args)} were given"
        )
    for (name, _), value in zip(fields, args, strict=False):
        if name in kwargs:
            raise TypeError(f"Field `{name}` given both positionally and by name")
        kwargs[name] = value
    return kwargs


def _is_struct_type(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, Struct) and obj is not Struct


def _own_annotations(cls: type) -> dict[str, Any]:
    cached = cls.__dict__.get("__struct_annotations__")
    if cached is not None:
        return cached
    annotations = dict(inspect.get_annotations(cls, eval_str=True))
    cls.__struct_annotations__ = annotations
    return annotations


def _substitute(tp: Any, mapping: TypeMapping) -> Any:
    if not mapping or isinstance(tp, type):
        return tp
    elif isinstance(tp, TypeVar):
        return mapping.get(tp, tp)
    params = getattr(tp, "__parameters__", ())
    if any(param in mapping for param in params):
        return tp[tuple(mapping.get(param, param) for param in params)]
    return tp


def _base_mapping(base: Any, base_origin: type, mapping: TypeMapping) -> TypeMapping:
    params = getattr(base_origin, "__parameters__", ())
    return {
        param: _substitute(arg, mapping) for param, arg in zip(params, get_args(base), strict=False)
    }


def _fields(cls: type, mapping: TypeMapping) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for base in cls.__dict__.get("__orig_bases__", cls.__bases__):
        base_origin = get_origin(base) or base
        if _is_struct_type(base_origin):
            fields.update(_fields(base_origin, _base_mapping(base, base_origin, mapping)))
    for name, tp in _own_annotations(cls).items():
        fields[name] = _substitute(tp, mapping)
    return fields


def resolve_generic_fields(tp: Any) -> Fields:
    origin = get_origin(tp)
    if not _is_struct_type(origin):
        raise TypeError(f"{tp!r} is not a struct type")
    params = getattr(origin, "__parameters__", ())
    args = get_args(tp)
    if not params:
        raise TypeError(f"{origin.__qualname__} is not a generic type")
    elif len(args) != len(params):
        raise TypeError(
            f"{origin.__qualname__} expects {len(params)} type argument(s), got {len(args)}"
        )
    return tuple(_fields(origin, dict(zip(params, args, strict=True))).items())
