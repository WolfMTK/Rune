import inspect
import reprlib
import sys
from collections.abc import Callable, Iterable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    TypeGuard,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from rune.rune import (
    _struct_copy,
    _struct_finalize,
    _struct_replace,
    _struct_values,
    _StructMixin,
    force_setattr,
)

_T = TypeVar("_T")

if TYPE_CHECKING:
    from typing_extensions import dataclass_transform
else:

    def dataclass_transform(**_options: Any) -> Callable[[_T], _T]:
        return lambda target: target


Fields = tuple[tuple[str, Any], ...]
TypeMapping = dict[Any, Any]
FieldSpec = str | tuple[str, Any] | tuple[str, Any, Any]

_MISSING: Any = object()
_I64_MIN, _I64_MAX = -(2**63), 2**63 - 1
_HASH_SLOT = "__rune_cached_hash__"
_FORBIDDEN_FIELD_NAMES = frozenset({"__weakref__", "__dict__", _HASH_SLOT})
_EMPTY_DEFAULT_TYPES = (dict, list, set, bytearray)
_FLAG_OPTIONS = (
    "frozen",
    "eq",
    "order",
    "array_like",
    "gc",
    "omit_defaults",
    "repr_omit_defaults",
    "forbid_unknown_fields",
)
_FLAG_DEFAULTS = {name: name in ("eq", "gc") for name in _FLAG_OPTIONS}
_INVALID_RENAMED_CHARACTERS = frozenset(chr(code) for code in range(0x20)) | {"\\", '"'}


class _NoDefault:
    __slots__ = ()
    _instance: ClassVar["_NoDefault | None"] = None

    def __new__(cls) -> "_NoDefault":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NODEFAULT"

    def __reduce__(self) -> str:
        return "NODEFAULT"


NODEFAULT: Any = _NoDefault()


class _DefaultFactory:
    __slots__ = ("factory",)

    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory = factory

    def __repr__(self) -> str:
        return f"Factory({self.factory!r})"


class Field:
    __slots__ = ("default", "default_factory", "name")

    def __init__(self, default: Any, default_factory: Any, name: str | None) -> None:
        self.default = default
        self.default_factory = default_factory
        self.name = name

    def __repr__(self) -> str:
        return (
            f"Field(default={self.default!r}, default_factory={self.default_factory!r}, "
            f"name={self.name!r})"
        )


@overload
def field(*, default: _T, name: str | None = None) -> _T: ...


@overload
def field(*, default_factory: Callable[[], _T], name: str | None = None) -> _T: ...


@overload
def field(*, name: str | None = None) -> Any: ...


def field(
    *, default: Any = NODEFAULT, default_factory: Any = NODEFAULT, name: str | None = None
) -> Any:
    if default is not NODEFAULT and default_factory is not NODEFAULT:
        raise TypeError("Cannot set both `default` and `default_factory`")
    if default_factory is not NODEFAULT and not callable(default_factory):
        raise TypeError("default_factory must be callable")
    if name is not None and not isinstance(name, str):
        raise TypeError("name must be a str or None")
    return Field(default, default_factory, name)


class StructConfig:
    __slots__ = (
        "array_like",
        "cache_hash",
        "dict",
        "eq",
        "forbid_unknown_fields",
        "frozen",
        "gc",
        "omit_defaults",
        "order",
        "repr_omit_defaults",
        "tag",
        "tag_field",
        "weakref",
    )

    array_like: bool
    cache_hash: bool
    dict: bool
    eq: bool
    forbid_unknown_fields: bool
    frozen: bool
    gc: bool
    omit_defaults: bool
    order: bool
    repr_omit_defaults: bool
    tag: str | int | None
    tag_field: str | None
    weakref: bool

    def __init__(self, **values: Any) -> None:
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("StructConfig is immutable")

    def __repr__(self) -> str:
        values = ", ".join(f"{name}={getattr(self, name)!r}" for name in self.__slots__)
        return f"StructConfig({values})"


class _Options:
    __slots__ = ("cache_hash", "flags", "rename", "tag", "tag_field")

    def __init__(self) -> None:
        self.flags: dict[str, bool | None] = dict.fromkeys(_FLAG_OPTIONS)
        self.tag: Any = None
        self.tag_field: Any = None
        self.rename: Any = None
        self.cache_hash = False


def _camel(name: str, capitalize_first: bool) -> str:
    parts = name.split("_")
    if len(parts) == 1 and not capitalize_first:
        return name
    out = []
    first = True
    for part in parts:
        if first and not part:
            out.append("_")
            continue
        out.append(part.title() if not first or capitalize_first else part)
        first = False
    return "".join(out)


_RENAME_RULES: dict[str, Callable[[str], str]] = {
    "lower": str.lower,
    "upper": str.upper,
    "camel": lambda name: _camel(name, False),
    "pascal": lambda name: _camel(name, True),
    "kebab": lambda name: name.strip("_").replace("_", "-"),
}


def _apply_rename(rename: Any, name: str) -> str:
    if isinstance(rename, str):
        rule = _RENAME_RULES.get(rename)
        if rule is None:
            raise ValueError(f"rename={rename!r} is unsupported")
        return rule(name)
    if callable(rename):
        renamed = rename(name)
        what = "calling `rename`"
    elif isinstance(rename, Mapping):
        try:
            renamed = rename[name]
        except KeyError:
            return name
        what = "`rename[field]`"
    else:
        raise TypeError("`rename` must be a str, callable, or mapping")
    if renamed is None:
        return name
    if type(renamed) is not str:
        raise TypeError(
            f"Expected {what} to return a `str` or `None`, got `{type(renamed).__name__}`"
        )
    return renamed


def _simple_qualname(cls: type) -> str:
    return cls.__qualname__.rsplit(".<locals>.", 1)[-1]


def _namespace_annotations(namespace: dict[str, Any]) -> dict[str, Any]:
    annotations = namespace.get("__annotations__")
    if annotations is not None:
        if not isinstance(annotations, dict):
            raise TypeError("__annotations__ must be a dict")
        return annotations
    if sys.version_info >= (3, 14):
        import annotationlib

        annotate = annotationlib.get_annotate_from_class_namespace(namespace)
        if annotate is not None:
            return annotationlib.call_annotate_function(annotate, annotationlib.Format.FORWARDREF)
    return {}


def _module_namespace(namespace: dict[str, Any]) -> dict[str, Any] | None:
    module = sys.modules.get(namespace.get("__module__", ""))
    return None if module is None else vars(module)


def _is_class_var(annotation: Any, namespace: dict[str, Any]) -> bool:
    if annotation is ClassVar:
        return True
    if type(annotation) is str:
        for prefix in ("ClassVar", "typing.ClassVar"):
            if annotation.startswith(prefix) and annotation[len(prefix) : len(prefix) + 1] in (
                "",
                "[",
            ):
                module_namespace = _module_namespace(namespace)
                if module_namespace is None:
                    return False
                if prefix == "ClassVar":
                    return module_namespace.get("ClassVar") is ClassVar
                typing_module = module_namespace.get("typing")
                if typing_module is None:
                    return False
                return typing_module.ClassVar is ClassVar
        return False
    return get_origin(annotation) is ClassVar


def _check_default(name: str, value: Any) -> Any:
    if isinstance(value, Field):
        if value.default is not NODEFAULT:
            value = value.default
        elif value.default_factory is tuple:
            return ()
        elif value.default_factory is frozenset:
            return frozenset()
        elif value.default_factory is not NODEFAULT:
            return _DefaultFactory(value.default_factory)
        else:
            return NODEFAULT
    if type(value) in _EMPTY_DEFAULT_TYPES:
        if value:
            raise TypeError(
                f"Using a non-empty mutable collection ({value!r}) as a default value is "
                "unsafe. Instead configure a `default_factory` for this field."
            )
        return _DefaultFactory(type(value))
    cls = type(value)
    if isinstance(cls, StructMeta) and not cls.__struct_config__.frozen:
        raise TypeError(
            f"Using a mutable struct object ({value!r}) as a default value is unsafe. "
            "Either configure a `default_factory` for this field, or set "
            f"`frozen=True` on `{cls.__name__}`"
        )
    return value


class _Builder:
    def __init__(self, namespace: dict[str, Any]) -> None:
        self.namespace = namespace
        self.options = _Options()
        self.defaults: dict[str, Any] = {}
        self.keyword_only: set[str] = set()
        self.renamed: dict[str, str] = {}
        self.slots: list[str] = []
        self.has_weakref = False
        self.has_dict = False
        self.has_hash_slot = False
        self.has_non_slots_bases = False

    def collect_base(self, base: Any) -> None:
        if not isinstance(base, type):
            raise TypeError("All base classes must be types")
        if base is _StructMixin:
            return
        if base.__weakrefoffset__:
            self.has_weakref = True
        if base.__dictoffset__:
            self.has_dict = True
        if not isinstance(base, StructMeta):
            if base.__dictoffset__:
                self.has_non_slots_bases = True
            for attribute in ("__init__", "__new__"):
                if attribute in base.__dict__:
                    raise TypeError(f"Struct base classes cannot define {attribute}")
            return
        options = base.__dict__["__rune_options__"]
        if options.cache_hash:
            self.has_hash_slot = True
            self.options.cache_hash = True
        if options.tag_field is not None:
            self.options.tag_field = options.tag_field
        if options.tag is not None:
            self.options.tag = options.tag
        if options.rename is not None:
            self.options.rename = options.rename
        for name, value in options.flags.items():
            if value is not None:
                self.options.flags[name] = value
        fields = base.__struct_fields__
        encode_fields = base.__struct_encode_fields__
        defaults = base.__struct_defaults__
        first_default = len(fields) - len(defaults)
        positional = base.__dict__["__rune_positional__"]
        for index, (name, encode_name) in enumerate(zip(fields, encode_fields, strict=True)):
            default = defaults[index - first_default] if index >= first_default else NODEFAULT
            self.defaults[name] = default
            if index >= positional:
                self.keyword_only.add(name)
            else:
                self.keyword_only.discard(name)
            if name != encode_name:
                self.renamed[name] = encode_name

    def collect_fields(self, kw_only: bool) -> None:
        namespace = self.namespace
        for name, annotation in _namespace_annotations(namespace).items():
            if type(name) is not str:
                raise TypeError("__annotations__ keys must be strings")
            if name in _FORBIDDEN_FIELD_NAMES:
                raise TypeError(f"Cannot have a struct field named {name!r}")
            if _is_class_var(annotation, namespace):
                continue
            if name not in self.defaults:
                self.slots.append(name)
            if kw_only:
                self.keyword_only.add(name)
            else:
                self.keyword_only.discard(name)
            value = namespace.pop(name, _MISSING)
            self.process_rename(name, value)
            self.defaults[name] = NODEFAULT if value is _MISSING else _check_default(name, value)

    def process_rename(self, name: str, value: Any) -> None:
        if isinstance(value, Field) and value.name is not None:
            if value.name == name:
                self.renamed.pop(name, None)
            else:
                self.renamed[name] = value.name
            return
        rename = self.options.rename
        if rename is None:
            return
        renamed = _apply_rename(rename, name)
        if renamed != name:
            self.renamed[name] = renamed
        else:
            self.renamed.pop(name, None)

    def construct_fields(self) -> tuple[tuple[str, ...], tuple[Any, ...], int]:
        fields: list[str] = []
        defaults: list[Any] = []
        for name, default in self.defaults.items():
            if name in self.keyword_only:
                continue
            fields.append(name)
            if default is NODEFAULT:
                if defaults:
                    raise TypeError(
                        f"Required field '{name}' cannot follow optional fields. Either "
                        "reorder the struct fields, or set `kw_only=True` in the struct "
                        "definition."
                    )
            else:
                defaults.append(default)
        positional = len(fields)
        for name, default in self.defaults.items():
            if name not in self.keyword_only:
                continue
            fields.append(name)
            if defaults or default is not NODEFAULT:
                defaults.append(default)
        return tuple(fields), tuple(defaults), positional

    def construct_slots(self, weakref: bool | None, dict_: bool | None, cache_hash: bool) -> None:
        slots = self.slots
        if weakref and not self.has_weakref:
            slots.append("__weakref__")
        elif weakref is False and self.has_weakref:
            raise ValueError("Cannot set `weakref=False` if base class already has `weakref=True`")
        if dict_ and not self.has_dict:
            slots.append("__dict__")
        elif dict_ is False and self.has_dict:
            raise ValueError("Cannot set `dict=False` if base class already has `dict=True`")
        if cache_hash and not self.has_hash_slot:
            slots.append(_HASH_SLOT)
        self.namespace["__slots__"] = tuple(sorted(slots))

    def construct_encode_fields(self, fields: tuple[str, ...]) -> tuple[str, ...]:
        if not self.renamed:
            return fields
        encode_fields = tuple(self.renamed.get(name, name) for name in fields)
        if len(set(encode_fields)) != len(encode_fields):
            raise ValueError("Multiple fields rename to the same name, field names must be unique")
        for name in encode_fields:
            if _INVALID_RENAMED_CHARACTERS.intersection(name):
                raise ValueError(
                    "Renamed field names must not contain '\\', '\"', or control characters "
                    f"('\\u0000' to '\\u001F') - '{name}' is invalid"
                )
        return encode_fields


def _resolve_tag(cls: type, options: _Options, encode_fields: tuple[str, ...]) -> tuple[Any, Any]:
    tag = options.tag
    if tag is False or (tag is None and options.tag_field is None):
        return None, None
    if tag is None or tag is True:
        value = _simple_qualname(cls)
    else:
        value = tag(_simple_qualname(cls)) if callable(tag) else tag
        if type(value) is int:
            if not _I64_MIN <= value <= _I64_MAX:
                raise ValueError("Integer `tag` values must be within [-2**63, 2**63 - 1]")
        elif type(value) is not str:
            raise TypeError("`tag` must be a `str` or an `int`")
    tag_field = options.tag_field
    if tag_field is None:
        tag_field = "type"
    elif type(tag_field) is not str:
        raise TypeError("`tag_field` must be a `str`")
    if tag_field in encode_fields:
        raise ValueError(f"`tag_field={tag_field!r}` conflicts with an existing field of that name")
    return value, tag_field


def _own_struct_attribute(cls: type, name: str, public_name: str) -> Any:
    try:
        return cls.__dict__[name]
    except KeyError:
        raise AttributeError(
            f"type object {cls.__name__!r} has no attribute {public_name!r}"
        ) from None


_root: Any = None


_ROOT_CONFIG = StructConfig(
    tag=None,
    tag_field=None,
    weakref=False,
    dict=False,
    cache_hash=False,
    **_FLAG_DEFAULTS,
)


class _StructMetaMeta(type):
    @property
    def __text_signature__(cls) -> str:
        return "(name, bases, namespace, /, **kwargs)"


@dataclass_transform(field_specifiers=(field,))
class StructMeta(type, metaclass=_StructMetaMeta):
    __rune_config__ = _ROOT_CONFIG
    __rune_positional__ = 0

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        *,
        tag_field: str | None = None,
        tag: str | int | bool | Callable[[str], str | int] | None = None,
        rename: Any = _MISSING,
        omit_defaults: bool | None = None,
        forbid_unknown_fields: bool | None = None,
        frozen: bool | None = None,
        eq: bool | None = None,
        order: bool | None = None,
        kw_only: bool = False,
        repr_omit_defaults: bool | None = None,
        array_like: bool | None = None,
        gc: bool | None = None,
        weakref: bool | None = None,
        dict: bool | None = None,
        cache_hash: bool | None = None,
    ) -> Any:
        if _root is None:
            return mcls._new_root(name, bases, namespace)
        for attribute in ("__init__", "__new__", "__slots__"):
            if attribute in namespace:
                raise TypeError(f"Struct types cannot define {attribute}")
        if not any(isinstance(base, StructMeta) for base in bases):
            bases = (*(base for base in bases if base is not object), _root)
        builder = _Builder(_copy_namespace(namespace))
        for base in reversed(bases):
            builder.collect_base(base)
        options = builder.options
        if tag is not None:
            options.tag = tag
        if tag_field is not None:
            options.tag_field = tag_field
        if rename is not _MISSING:
            options.rename = rename
        arguments = {
            "frozen": frozen,
            "eq": eq,
            "order": order,
            "array_like": array_like,
            "gc": gc,
            "omit_defaults": omit_defaults,
            "repr_omit_defaults": repr_omit_defaults,
            "forbid_unknown_fields": forbid_unknown_fields,
        }
        for option, value in arguments.items():
            if value is not None:
                options.flags[option] = bool(value)
        flags = {
            key: _FLAG_DEFAULTS[key] if value is None else value
            for key, value in options.flags.items()
        }
        if not flags["eq"] and flags["order"]:
            raise ValueError("Cannot set eq=False and order=True")
        wants_hash_cache = bool(cache_hash) if cache_hash is not None else builder.has_hash_slot
        if cache_hash and not flags["frozen"]:
            raise ValueError("Cannot set cache_hash=True without frozen=True")
        if cache_hash is False and builder.has_hash_slot:
            raise ValueError(
                "Cannot set `cache_hash=False` if base class already has `cache_hash=True`"
            )
        if not flags["gc"]:
            if builder.has_non_slots_bases:
                raise ValueError(
                    "Cannot set gc=False when inheriting from non-struct types with a __dict__"
                )
            if dict or builder.has_dict:
                raise ValueError("Cannot set gc=False and dict=True")
            if weakref or builder.has_weakref:
                raise ValueError(
                    "Cannot set gc=False and weakref=True at the same time, due to possible "
                    "memory corruption"
                )
        builder.collect_fields(bool(kw_only))
        fields, defaults, positional = builder.construct_fields()
        builder.construct_slots(weakref, dict, bool(cache_hash))
        encode_fields = builder.construct_encode_fields(fields)
        cls = super().__new__(mcls, name, bases, builder.namespace)
        tag_value, resolved_tag_field = _resolve_tag(cls, options, encode_fields)
        options.cache_hash = wants_hash_cache
        mcls._finalize(
            cls,
            fields,
            encode_fields,
            defaults,
            positional,
            options,
            StructConfig(
                tag=tag_value,
                tag_field=resolved_tag_field,
                weakref=cls.__weakrefoffset__ != 0,
                dict=cls.__dictoffset__ != 0,
                cache_hash=wants_hash_cache,
                **flags,
            ),
        )
        return cls

    @staticmethod
    def _new_root(name: str, bases: tuple[type, ...], namespace: dict[str, Any]) -> Any:
        cls = type.__new__(StructMeta, name, bases, namespace)
        StructMeta._finalize(
            cls,
            (),
            (),
            (),
            0,
            _Options(),
            _ROOT_CONFIG,
        )
        return cls

    @staticmethod
    def _finalize(
        cls: type,
        fields: tuple[str, ...],
        encode_fields: tuple[str, ...],
        defaults: tuple[Any, ...],
        positional: int,
        options: _Options,
        config: StructConfig,
    ) -> None:
        type.__setattr__(cls, "__rune_options__", options)
        type.__setattr__(cls, "__rune_config__", config)
        type.__setattr__(cls, "__rune_positional__", positional)
        _struct_finalize(cls, fields, encode_fields, defaults, len(fields) - positional, config)

    @property
    def __struct_fields__(cls) -> tuple[str, ...]:
        return _own_struct_attribute(cls, "__rune_struct__", "__struct_fields__").fields

    @property
    def __struct_encode_fields__(cls) -> tuple[str, ...]:
        info = _own_struct_attribute(cls, "__rune_struct__", "__struct_encode_fields__")
        return info.encode_fields

    @property
    def __struct_defaults__(cls) -> tuple[Any, ...]:
        return _own_struct_attribute(cls, "__rune_struct__", "__struct_defaults__").defaults

    @property
    def __struct_config__(cls) -> StructConfig:
        return _own_struct_attribute(cls, "__rune_config__", "__struct_config__")

    @property
    def __match_args__(cls) -> tuple[str, ...]:
        info = _own_struct_attribute(cls, "__rune_struct__", "__match_args__")
        return info.fields[: cls.__dict__["__rune_positional__"]]

    @property
    def __signature__(cls) -> inspect.Signature:
        fields = cls.__struct_fields__
        defaults = cls.__struct_defaults__
        positional = cls.__dict__["__rune_positional__"]
        types = dict(_struct_field_types(cls))
        first_default = len(fields) - len(defaults)
        parameters = []
        for index, name in enumerate(fields):
            default = defaults[index - first_default] if index >= first_default else NODEFAULT
            if type(default) is _DefaultFactory:
                default = default.factory()
            kind = (
                inspect.Parameter.POSITIONAL_OR_KEYWORD
                if index < positional
                else inspect.Parameter.KEYWORD_ONLY
            )
            parameters.append(
                inspect.Parameter(
                    name,
                    kind,
                    default=inspect.Parameter.empty if default is NODEFAULT else default,
                    annotation=types.get(name, inspect.Parameter.empty),
                )
            )
        return inspect.Signature(parameters)


def _copy_namespace(namespace: Mapping[str, Any]) -> dict[str, Any]:
    return dict(namespace)


def _is_default(value: Any, default: Any) -> bool:
    if value is default:
        return True
    if type(default) is _DefaultFactory:
        factory = default.factory
        return factory in _EMPTY_DEFAULT_TYPES and type(value) is factory and not value
    return False


def _rebuild(cls: type, kwargs: dict[str, Any]) -> Any:
    return cls(**kwargs)


class Struct(_StructMixin, metaclass=StructMeta):
    __slots__ = ()
    __module__ = "rune"

    @property
    def __struct_fields__(self) -> tuple[str, ...]:
        cls: StructMeta = type(self)
        return cls.__struct_fields__

    @property
    def __struct_encode_fields__(self) -> tuple[str, ...]:
        cls: StructMeta = type(self)
        return cls.__struct_encode_fields__

    @property
    def __struct_defaults__(self) -> tuple[Any, ...]:
        cls: StructMeta = type(self)
        return cls.__struct_defaults__

    @property
    def __struct_config__(self) -> StructConfig:
        cls: StructMeta = type(self)
        return cls.__struct_config__

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self) or not type(self).__rune_config__.eq:
            return NotImplemented
        if other is self:
            return True
        return _struct_values(self) == _struct_values(other)

    def __lt__(self, other: object) -> bool:
        if type(other) is not type(self) or not type(self).__rune_config__.order:
            return NotImplemented
        return _struct_values(self) < _struct_values(other)

    def __le__(self, other: object) -> bool:
        if type(other) is not type(self) or not type(self).__rune_config__.order:
            return NotImplemented
        return _struct_values(self) <= _struct_values(other)

    def __gt__(self, other: object) -> bool:
        if type(other) is not type(self) or not type(self).__rune_config__.order:
            return NotImplemented
        return _struct_values(self) > _struct_values(other)

    def __ge__(self, other: object) -> bool:
        if type(other) is not type(self) or not type(self).__rune_config__.order:
            return NotImplemented
        return _struct_values(self) >= _struct_values(other)

    def __hash__(self) -> int:
        config = type(self).__rune_config__
        if not config.eq:
            return object.__hash__(self)
        if not config.frozen:
            raise TypeError(f"unhashable type: '{type(self).__name__}'")
        if config.cache_hash:
            try:
                return object.__getattribute__(self, _HASH_SLOT)
            except AttributeError:
                pass
        value = hash((type(self), *_struct_values(self)))
        if config.cache_hash:
            force_setattr(self, _HASH_SLOT, value)
        return value

    def __rich_repr__(self) -> list[tuple[str, Any]]:
        cls: StructMeta = type(self)
        fields = cls.__struct_fields__
        values = _struct_values(self)
        if not cls.__rune_config__.repr_omit_defaults:
            return list(zip(fields, values, strict=True))
        defaults = cls.__struct_defaults__
        first_default = len(fields) - len(defaults)
        return [
            (name, value)
            for index, (name, value) in enumerate(zip(fields, values, strict=True))
            if index < first_default or not _is_default(value, defaults[index - first_default])
        ]

    @reprlib.recursive_repr()
    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={value!r}" for name, value in self.__rich_repr__())
        return f"{type(self).__name__}({fields})"

    def __copy__(self) -> Any:
        return _struct_copy(self)

    def __replace__(self, **changes: Any) -> Any:
        return _struct_replace(self, changes)

    def __reduce__(self) -> tuple[Any, ...]:
        cls: StructMeta = type(self)
        values = _struct_values(self)
        if cls.__dict__["__rune_positional__"] == len(values):
            return cls, values
        return _rebuild, (cls, dict(zip(cls.__struct_fields__, values, strict=True)))


_root = Struct


def defstruct(
    name: str,
    fields: Iterable[FieldSpec] = _MISSING,
    *,
    bases: tuple[type, ...] | None = None,
    module: str | None = None,
    namespace: dict[str, Any] | None = None,
    **options: Any,
) -> type[Struct]:
    if not isinstance(name, str):
        raise TypeError(f"defstruct() argument 'name' must be str, not {type(name).__name__}")
    if fields is _MISSING:
        raise TypeError("defstruct() missing required argument 'fields'")
    if bases is not None and not isinstance(bases, tuple):
        raise TypeError("`bases` must be a tuple or None")
    if module is not None and not isinstance(module, str):
        raise TypeError("`module` must be a str or None")
    if namespace is not None and not isinstance(namespace, dict):
        raise TypeError("`namespace` must be a dict or None")
    try:
        specs = iter(fields)
    except TypeError:
        raise TypeError("`fields` must be an iterable") from None
    body = {} if namespace is None else dict(namespace)
    annotations: dict[str, Any] = {}
    for spec in specs:
        if isinstance(spec, str):
            field_name, field_type, default = spec, Any, _MISSING
        elif isinstance(spec, tuple) and len(spec) in (2, 3) and isinstance(spec[0], str):
            field_name, field_type, default = (*spec, _MISSING)[:3]
        else:
            raise TypeError(
                "items in `fields` must be one of `str`, `tuple[str, type]`, or "
                "`tuple[str, type, Any]`"
            )
        annotations[field_name] = field_type
        if default is not _MISSING:
            body[field_name] = default
    body["__annotations__"] = annotations
    if module is None:
        module = sys._getframe(1).f_globals.get("__name__")
    if module is not None:
        body["__module__"] = module
    return StructMeta(name, (_root,) if bases is None else bases, body, **options)


def _struct_field_types(cls: StructMeta) -> Fields:
    cached = cls.__dict__.get("__rune_field_types__")
    if cached is not None:
        return cached
    types = _fields(cls, {})
    resolved = tuple((name, types[name]) for name in cls.__struct_fields__)
    type.__setattr__(cls, "__rune_field_types__", resolved)
    return resolved


def resolve_generic_fields(tp: Any) -> Fields:
    origin = get_origin(tp)
    if not _is_struct_type(origin):
        raise TypeError(f"{tp!r} is not a struct type")
    types = _fields(origin, _type_mapping(tp, origin))
    return tuple((name, types[name]) for name in origin.__struct_fields__)


def resolve_type_alias(tp: Any) -> Any:
    alias = get_origin(tp) or tp
    if alias is tp:
        return alias.__value__
    params = alias.__type_params__
    args = get_args(tp)
    if len(args) != len(params):
        raise TypeError(f"{alias.__name__} expects {len(params)} type argument(s), got {len(args)}")
    return _substitute(alias.__value__, dict(zip(params, args, strict=True)))


def resolve_namedtuple_fields(tp: Any) -> Fields:
    origin = get_origin(tp) or tp
    annotations = get_type_hints(origin)
    hints = {name: annotations.get(name, Any) for name in origin._fields}
    if origin is tp:
        return tuple(hints.items())
    mapping = _type_mapping(tp, origin)
    return tuple((name, _substitute(hint, mapping)) for name, hint in hints.items())


def _is_struct_type(obj: Any) -> TypeGuard[StructMeta]:
    return isinstance(obj, StructMeta) and obj is not Struct


def _own_annotations(cls: StructMeta) -> dict[str, Any]:
    cached = cls.__dict__.get("__struct_annotations__")
    if cached is not None:
        return cached
    try:
        hints = get_type_hints(cls, include_extras=True)
    except NameError:
        hints = get_type_hints(cls, localns=_self_namespace(cls), include_extras=True)
    fields = set(cls.__struct_fields__)
    annotations = {
        name: hints[name]
        for name in inspect.get_annotations(cls)
        if name in fields and name in hints
    }
    type.__setattr__(cls, "__struct_annotations__", annotations)
    return annotations


def _self_namespace(cls: type) -> dict[str, Any]:
    module = sys.modules.get(cls.__module__)
    return {cls.__name__: cls, **vars(cls), **getattr(module, "__dict__", {})}


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


def _fields(cls: StructMeta, mapping: TypeMapping) -> dict[str, Any]:
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
