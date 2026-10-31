import inspect
import reprlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, get_args, get_origin, get_type_hints

_T = TypeVar("_T")

if TYPE_CHECKING:
    from typing_extensions import dataclass_transform
else:

    def dataclass_transform(**_options: Any) -> Callable[[_T], _T]:
        return lambda target: target


Fields = tuple[tuple[str, Any], ...]
TypeMapping = dict[Any, Any]

_MISSING: Any = object()
_COPIED_DEFAULT_TYPES = frozenset({list, dict, set})
_I64_MIN, _I64_MAX = -(2**63), 2**63 - 1


@dataclass_transform()
class StructMeta(type):
    pass


class _Lazy:
    def __init__(self, name: str, compute: Callable[[type], Any]) -> None:
        self.name = name
        self.compute = compute

    def __get__(self, obj: Any, owner: type) -> Any:
        value = self.compute(owner)
        setattr(owner, self.name, value)
        return value


class Struct(metaclass=StructMeta):
    __struct_fields__: ClassVar[Fields] = ()
    __match_args__: ClassVar[tuple[str, ...]] = ()
    __struct_array_like__: ClassVar[bool] = False
    __struct_tag__: ClassVar[str | int | None] = None
    __struct_tag_field__: ClassVar[str] = "type"

    def __init_subclass__(
        cls,
        tag: str | int | bool | None = None,
        tag_field: str | None = None,
        array_like: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        cls.__struct_fields__ = _Lazy("__struct_fields__", _resolve_fields)
        if "__match_args__" not in cls.__dict__:
            cls.__match_args__ = _Lazy("__match_args__", _field_names)
        if array_like is not None:
            cls.__struct_array_like__ = array_like
        if tag_field is not None:
            cls.__struct_tag_field__ = tag_field
        cls.__struct_tag__ = _resolve_tag(cls, tag)

    def __init__(self, /, *args: Any, **kwargs: Any) -> None:
        cls = type(self)
        if args:
            kwargs = _merge_positional_args(cls, args, kwargs)
        for name, _ in cls.__struct_fields__:
            value = kwargs.pop(name, _MISSING)
            if value is _MISSING:
                value = getattr(cls, name, _MISSING)
                if value is _MISSING:
                    raise TypeError(f"Missing required field `{name}`")
                if type(value) in _COPIED_DEFAULT_TYPES:
                    value = value.copy()
            setattr(self, name, value)

        if kwargs:
            unknown = ", ".join(f"`{name}`" for name in kwargs)
            raise TypeError(f"Unexpected field(s): {unknown}")
        post_init = getattr(cls, "__post_init__", None)
        if post_init is not None:
            post_init(self)

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if type(other) is not type(self):
            return NotImplemented
        return all(
            getattr(self, name) == getattr(other, name) for name, _ in self.__struct_fields__
        )

    @reprlib.recursive_repr()
    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={getattr(self, name)!r}" for name, _ in self.__struct_fields__)
        return f"{type(self).__name__}({fields})"


def resolve_generic_fields(tp: Any) -> Fields:
    origin = get_origin(tp)
    if not _is_struct_type(origin):
        raise TypeError(f"{tp!r} is not a struct type")
    return tuple(_fields(origin, _type_mapping(tp, origin)).items())


def resolve_namedtuple_fields(tp: Any) -> Fields:
    origin = get_origin(tp) or tp
    annotations = get_type_hints(origin)
    hints = {name: annotations.get(name, Any) for name in origin._fields}
    if origin is tp:
        return tuple(hints.items())
    mapping = _type_mapping(tp, origin)
    return tuple((name, _substitute(hint, mapping)) for name, hint in hints.items())


def _resolve_fields(cls: type) -> Fields:
    return tuple(_fields(cls, {}).items())


def _field_names(cls: type) -> tuple[str, ...]:
    return tuple(name for name, _ in cls.__struct_fields__)


def _is_class_var(tp: Any) -> bool:
    return tp is ClassVar or get_origin(tp) is ClassVar


def _is_struct_type(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, Struct) and obj is not Struct


def _own_annotations(cls: type) -> dict[str, Any]:
    cached = cls.__dict__.get("__struct_annotations__")
    if cached is not None:
        return cached
    hints = get_type_hints(cls, include_extras=True)
    annotations = {
        name: hints[name] for name in inspect.get_annotations(cls) if not _is_class_var(hints[name])
    }
    cls.__struct_annotations__ = annotations
    return annotations


def _substitute(tp: Any, mapping: TypeMapping) -> Any:
    if not mapping or isinstance(tp, type):
        return tp
    if isinstance(tp, TypeVar):
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
        if not _is_struct_type(base_origin):
            continue
        for name, tp in _fields(base_origin, _base_mapping(base, base_origin, mapping)).items():
            if name in fields and fields[name] != tp:
                raise TypeError(
                    f"{cls.__qualname__}: field `{name}` has different types in its base classes"
                )
            fields[name] = tp
    for name, tp in _own_annotations(cls).items():
        fields[name] = _substitute(tp, mapping)
    return fields


def _type_mapping(tp: Any, origin: type) -> TypeMapping:
    params = getattr(origin, "__parameters__", ())
    args = get_args(tp)
    if not params:
        raise TypeError(f"{origin.__qualname__} is not a generic type")
    if len(args) != len(params):
        raise TypeError(
            f"{origin.__qualname__} expects {len(params)} type argument(s), got {len(args)}"
        )
    return dict(zip(params, args, strict=True))


def _resolve_tag(cls: type, tag: Any) -> str | int | None:
    if tag is True:
        return cls.__name__
    if tag is None or tag is False:
        return None
    if isinstance(tag, int) and not _I64_MIN <= tag <= _I64_MAX:
        raise NotImplementedError(
            "Integer tag values outside the 64-bit signed integer range are not supported"
        )
    if isinstance(tag, (str, int)):
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
