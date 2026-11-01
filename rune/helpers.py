import dataclasses
import inspect
import sys
import typing
from typing import Any, ClassVar, get_args, get_origin, get_type_hints

from rune.struct import _base_mapping, _DefaultFactory, _substitute, _type_mapping

_QUALIFIERS = (typing.Required, typing.NotRequired)


def is_typeddict(tp: Any) -> bool:
    if typing.is_typeddict(tp):
        return True
    typing_extensions = sys.modules.get("typing_extensions")
    return typing_extensions is not None and typing_extensions.is_typeddict(tp)


def plain_class_fields(tp: Any, decoding: bool = True) -> dict[str, tuple[Any, bool, Any]]:
    origin = get_origin(tp) or tp
    mapping = {} if origin is tp else _type_mapping(tp, origin)
    types = _fields(origin, mapping, decoding)
    info = _field_info(origin, decoding)
    if decoding and is_typeddict(origin):
        missing = [key for key, (required, _, _) in info.items() if required and key not in types]
        if missing:
            names = ", ".join(f"`{key}`" for key in missing)
            raise RuntimeError(f"{origin.__qualname__} requires keys {names} that are not declared")
    fields = {}
    for name, field_ty in types.items():
        required, decodable, default = info.get(name, (False, True, None))
        if decoding and not decodable:
            continue
        fields[name] = (field_ty, required, default)
    return fields


def _is_plain(obj: Any) -> bool:
    return isinstance(obj, type) and (
            is_typeddict(obj) or dataclasses.is_dataclass(obj) or hasattr(obj, "__attrs_attrs__")
    )


def _fields(cls: type, mapping: dict[Any, Any], decoding: bool) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for base in cls.__dict__.get("__orig_bases__", cls.__bases__):
        base_origin = get_origin(base) or base
        if not _is_plain(base_origin):
            continue
        for name, field_ty in _fields(
                base_origin, _base_mapping(base, base_origin, mapping), decoding
        ).items():
            if name in fields and fields[name] != field_ty:
                raise TypeError(
                    f"{cls.__qualname__}: field `{name}` has different types in its base classes"
                )
            fields[name] = field_ty
    for name, field_ty in _own_fields(cls, fields, decoding).items():
        fields[name] = _substitute(field_ty, mapping)
    return fields


def _own_fields(cls: type, inherited: dict[str, Any], decoding: bool) -> dict[str, Any]:
    if not _declares_fields(cls):
        return {}
    hints = _resolved_hints(cls)
    if "__attrs_attrs__" in cls.__dict__:
        return _attrs_own_fields(cls, hints)
    typed_dict = is_typeddict(cls)
    annotations = inspect.get_annotations(cls)
    fields = {}
    for name, annotation in annotations.items():
        if typed_dict and name in inherited:
            continue
        hint = _strip_qualifiers(hints.get(name, annotation))
        if _is_class_var(hint):
            continue
        if _is_init_var(hint):
            if decoding:
                raise TypeError("`InitVar` fields are not supported")
            continue
        fields[name] = hint
    return fields


def _resolved_hints(cls: type) -> dict[str, Any]:
    type_params = {param.__name__: param for param in getattr(cls, "__type_params__", ())}
    try:
        return get_type_hints(cls, localns=type_params, include_extras=True)
    except NameError:
        return inspect.get_annotations(cls)


def _attrs_own_fields(cls: type, hints: dict[str, Any]) -> dict[str, Any]:
    return {
        attribute.name: hints.get(attribute.name, attribute.type or Any)
        for attribute in cls.__attrs_attrs__
        if not attribute.inherited and not attribute.name.startswith("_")
    }


def _declares_fields(cls: type) -> bool:
    own = cls.__dict__
    return "__dataclass_fields__" in own or "__attrs_attrs__" in own or "__required_keys__" in own


def _strip_qualifiers(hint: Any) -> Any:
    while get_origin(hint) in _QUALIFIERS:
        hint = get_args(hint)[0]
    return hint


def _is_class_var(hint: Any) -> bool:
    return hint is ClassVar or get_origin(hint) is ClassVar


def _is_init_var(hint: Any) -> bool:
    return hint is dataclasses.InitVar or isinstance(hint, dataclasses.InitVar)


def _field_info(origin: type, decoding: bool) -> dict[str, tuple[bool, bool, Any]]:
    if is_typeddict(origin):
        required = _typed_dict_required(origin)
        return {
            name: (name in required, True, None)
            for name in (*origin.__required_keys__, *origin.__optional_keys__)
        }
    if dataclasses.is_dataclass(origin):
        return _dataclass_info(origin)
    if hasattr(origin, "__attrs_attrs__"):
        return _attrs_info(origin, decoding)
    return {}


def _typed_dict_required(origin: type) -> set[str]:
    required = set(origin.__required_keys__)
    hints = _resolved_hints(origin)
    for name in (*origin.__required_keys__, *origin.__optional_keys__):
        qualifier = get_origin(hints.get(name))
        if qualifier is typing.NotRequired:
            required.discard(name)
        elif qualifier is typing.Required:
            required.add(name)
    return required


def _dataclass_info(origin: type) -> dict[str, tuple[bool, bool, Any]]:
    if not isinstance(origin.__dataclass_fields__, dict):
        raise RuntimeError(f"{origin.__qualname__}.__dataclass_fields__ is not a dict")
    info = {}
    for field in dataclasses.fields(origin):
        if field.default_factory is not dataclasses.MISSING:
            default = _DefaultFactory(field.default_factory)
        else:
            default = field.default
        has_default = (
                field.default is not dataclasses.MISSING
                or field.default_factory is not dataclasses.MISSING
        )
        info[field.name] = (not has_default, field.init, default)
    return info


def _attrs_info(origin: type, decoding: bool) -> dict[str, tuple[bool, bool, Any]]:
    from attrs import NOTHING, Factory

    info = {}
    for attribute in origin.__attrs_attrs__:
        default = attribute.default
        if isinstance(default, Factory):
            if default.takes_self and decoding:
                raise NotImplementedError(
                    f"`takes_self` factories are not supported, got `{attribute.name}` "
                    f"of {origin.__qualname__}"
                )
            default = _DefaultFactory(default.factory)
        info[attribute.name] = (attribute.default is NOTHING, attribute.init, default)
    return info
