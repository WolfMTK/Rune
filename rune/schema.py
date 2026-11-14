import re
import textwrap
from collections.abc import Callable, Iterable
from types import ModuleType
from typing import Any

from rune._conversion import to_builtins
from rune.struct import NODEFAULT

__all__ = ["schema", "schema_components"]

_REF_TEMPLATE = "#/$defs/{name}"
_FACTORY_DEFAULTS = (list, dict, set, bytearray)

SchemaHook = Callable[[type], dict[str, Any]]


def schema(
    type: Any,
    *,
    schema_hook: SchemaHook | None = None,
    ref_template: str = _REF_TEMPLATE,
) -> dict[str, Any]:
    (out,), components = schema_components(
        (type,), schema_hook=schema_hook, ref_template=ref_template
    )
    if components:
        out["$defs"] = components
    return out


def schema_components(
    types: Iterable[Any],
    *,
    schema_hook: SchemaHook | None = None,
    ref_template: str = _REF_TEMPLATE,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    from rune import inspect

    infos = inspect.multi_type_info(types)
    component_types = _collect_component_types(inspect, infos)
    names = _build_name_map(component_types)
    generator = _SchemaGenerator(inspect, names, schema_hook, ref_template)
    schemas = tuple(generator.to_schema(info) for info in infos)
    components = {
        names[cls]: generator.to_schema(info, check_ref=False)
        for cls, info in component_types.items()
    }
    return schemas, components


def _collect_component_types(mi: ModuleType, infos: Iterable[Any]) -> dict[Any, Any]:
    records = (mi.StructType, mi.TypedDictType, mi.DataclassType, mi.NamedTupleType)
    collections = (mi.ListType, mi.SetType, mi.FrozenSetType, mi.VarTupleType)
    components: dict[Any, Any] = {}

    def collect(info: Any) -> None:
        if isinstance(info, records):
            if info.cls not in components:
                components[info.cls] = info
                for field in info.fields:
                    collect(field.type)
        elif isinstance(info, mi.EnumType):
            components[info.cls] = info
        elif isinstance(info, mi.Metadata):
            collect(info.type)
        elif isinstance(info, collections):
            collect(info.item_type)
        elif isinstance(info, mi.TupleType):
            for item in info.item_types:
                collect(item)
        elif isinstance(info, (mi.DictType, mi.FrozenDictType)):
            collect(info.key_type)
            collect(info.value_type)
        elif isinstance(info, mi.UnionType):
            for member in info.types:
                collect(member)

    for info in infos:
        collect(info)
    return components


def _type_repr(obj: Any) -> str:
    return obj.__name__ if isinstance(obj, type) else repr(obj)


def _class_name(cls: Any) -> str:
    if hasattr(cls, "__origin__"):
        arguments = ", ".join(_type_repr(argument) for argument in cls.__args__)
        return f"{cls.__origin__.__name__}[{arguments}]"
    return cls.__name__


def _normalize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9.\-_]", "_", name)


def _full_name(cls: Any) -> str:
    return _normalize(f"{cls.__module__}.{cls.__qualname__}")


def _build_name_map(component_types: dict[Any, Any]) -> dict[Any, str]:
    conflicts: set[str] = set()
    names: dict[str, Any] = {}
    for cls in component_types:
        name = _normalize(_class_name(cls))
        if name in names:
            conflicts.add(name)
            previous = names.pop(name)
            names[_full_name(previous)] = previous
        if name in conflicts:
            names[_full_name(cls)] = cls
        else:
            names[name] = cls
    return {cls: name for name, cls in names.items()}


class _SchemaGenerator:
    def __init__(
        self,
        mi: ModuleType,
        names: dict[Any, str],
        schema_hook: SchemaHook | None,
        ref_template: str,
    ) -> None:
        self.mi = mi
        self.names = names
        self.schema_hook = schema_hook
        self.ref_template = ref_template

    def ref(self, cls: Any) -> str:
        return self.ref_template.format(name=self.names[cls])

    def doc(self, info: Any) -> str:
        mi = self.mi
        cls = getattr(info.cls, "__origin__", info.cls)
        doc = getattr(cls, "__doc__", "")
        if not doc:
            return ""
        doc = textwrap.dedent(doc).strip("\r\n")
        if isinstance(info, mi.EnumType):
            if doc == "An enumeration.":
                return ""
        elif (
            isinstance(info, (mi.NamedTupleType, mi.DataclassType))
            and doc.startswith(f"{cls.__name__}(")
            and doc.endswith(")")
        ):
            return ""
        return doc

    def describe(self, schema: dict[str, Any], info: Any) -> None:
        schema.setdefault("title", _class_name(info.cls))
        if doc := self.doc(info):
            schema.setdefault("description", doc)

    def to_schema(self, info: Any, check_ref: bool = True) -> dict[str, Any]:
        mi = self.mi
        schema: dict[str, Any] = {}
        while isinstance(info, mi.Metadata):
            schema = mi._merge_json(schema, info.extra_json_schema or {})
            info = info.type

        if check_ref and hasattr(info, "cls") and info.cls in self.names:
            schema["$ref"] = self.ref(info.cls)
            return schema

        if isinstance(info, (mi.AnyType, mi.RawType)):
            pass
        elif isinstance(info, mi.NoneType):
            schema["type"] = "null"
        elif isinstance(info, mi.BoolType):
            schema["type"] = "boolean"
        elif isinstance(info, (mi.IntType, mi.FloatType)):
            self.numeric(schema, info)
        elif isinstance(info, mi.StrType):
            schema["type"] = "string"
            _set(schema, "maxLength", info.max_length)
            _set(schema, "minLength", info.min_length)
            _set(schema, "pattern", info.pattern)
        elif isinstance(info, (mi.BytesType, mi.ByteArrayType, mi.MemoryViewType)):
            schema["type"] = "string"
            schema["contentEncoding"] = "base64"
            _set(schema, "maxLength", _base64_length(info.max_length))
            _set(schema, "minLength", _base64_length(info.min_length))
        elif isinstance(info, mi.DateTimeType):
            schema["type"] = "string"
            if info.tz is True:
                schema["format"] = "date-time"
        elif isinstance(info, mi.TimeType):
            schema["type"] = "string"
            if info.tz is True:
                schema["format"] = "time"
            elif info.tz is False:
                schema["format"] = "partial-time"
        elif isinstance(info, mi.DateType):
            schema.update(type="string", format="date")
        elif isinstance(info, mi.TimeDeltaType):
            schema.update(type="string", format="duration")
        elif isinstance(info, mi.UUIDType):
            schema.update(type="string", format="uuid")
        elif isinstance(info, mi.DecimalType):
            schema.update(type="string", format="decimal")
        elif isinstance(info, (mi.ListType, mi.SetType, mi.FrozenSetType, mi.VarTupleType)):
            self.collection(schema, info)
        elif isinstance(info, mi.TupleType):
            schema["type"] = "array"
            schema["minItems"] = schema["maxItems"] = len(info.item_types)
            if info.item_types:
                schema["prefixItems"] = [self.to_schema(item) for item in info.item_types]
                schema["items"] = False
        elif isinstance(info, (mi.DictType, mi.FrozenDictType)):
            self.mapping(schema, info)
        elif isinstance(info, mi.UnionType):
            self.union(schema, info)
        elif isinstance(info, mi.LiteralType):
            schema["enum"] = list(mi._sort_literals(info.values))
        elif isinstance(info, mi.EnumType):
            self.describe(schema, info)
            members = {id(member): member for member in info.cls.__members__.values()}
            schema["enum"] = sorted(member.value for member in members.values())
        elif isinstance(info, mi.StructType):
            self.struct(schema, info)
        elif isinstance(info, (mi.TypedDictType, mi.DataclassType, mi.NamedTupleType)):
            self.record(schema, info)
        elif isinstance(info, mi.ExtType):
            raise TypeError("json-schema doesn't support msgpack Ext types")
        elif isinstance(info, mi.CustomType):
            schema = self.custom(schema, info)
        else:
            raise TypeError(f"json-schema doesn't support type {info!r}")
        return schema

    def numeric(self, schema: dict[str, Any], info: Any) -> None:
        schema["type"] = "integer" if isinstance(info, self.mi.IntType) else "number"
        _set(schema, "minimum", info.ge)
        _set(schema, "exclusiveMinimum", info.gt)
        _set(schema, "maximum", info.le)
        _set(schema, "exclusiveMaximum", info.lt)
        _set(schema, "multipleOf", info.multiple_of)

    def collection(self, schema: dict[str, Any], info: Any) -> None:
        mi = self.mi
        schema["type"] = "array"
        if not isinstance(info.item_type, mi.AnyType):
            schema["items"] = self.to_schema(info.item_type)
        _set(schema, "maxItems", info.max_length)
        _set(schema, "minItems", info.min_length)
        if isinstance(info, (mi.SetType, mi.FrozenSetType)):
            schema["uniqueItems"] = True

    def mapping(self, schema: dict[str, Any], info: Any) -> None:
        mi = self.mi
        schema["type"] = "object"
        key = info.key_type
        while isinstance(key, mi.Metadata):
            key = key.type
        if _has_text_keys(mi, key):
            property_names = self.to_schema(info.key_type)
            property_names.pop("type", None)
            if property_names:
                schema["propertyNames"] = property_names
        if not isinstance(info.value_type, mi.AnyType):
            schema["additionalProperties"] = self.to_schema(info.value_type)
        _set(schema, "maxProperties", info.max_length)
        _set(schema, "minProperties", info.min_length)

    def union(self, schema: dict[str, Any], info: Any) -> None:
        mi = self.mi
        structs: dict[Any, Any] = {}
        others: list[Any] = []
        none_member = mi.NoneType() if info.includes_none else None
        tag_field = None
        for member in info.types:
            real = member
            while isinstance(real, mi.Metadata):
                real = real.type
            if isinstance(real, mi.StructType) and not real.array_like:
                tag_field = real.tag_field
                structs[real.tag] = real
            elif isinstance(real, mi.NoneType):
                none_member = member
            else:
                others.append(member)
        options = [self.to_schema(member) for member in others]
        if len(structs) >= 2:
            tagged = {
                "anyOf": [self.to_schema(member) for member in structs.values()],
                "discriminator": {
                    "propertyName": tag_field,
                    "mapping": {tag: self.ref(member.cls) for tag, member in structs.items()},
                },
            }
            if not options and none_member is None:
                schema.update(tagged)
                return
            options.append(tagged)
        elif structs:
            options.append(self.to_schema(next(iter(structs.values()))))
        if none_member is not None:
            options.append(self.to_schema(none_member))
        schema["anyOf"] = options

    def field_schemas(self, info: Any) -> tuple[list[str], list[dict[str, Any]], list[str]]:
        names: list[str] = []
        schemas: list[dict[str, Any]] = []
        required: list[str] = []
        for field in info.fields:
            field_schema = self.to_schema(field.type)
            if field.required:
                required.append(field.encode_name)
            elif field.default is not NODEFAULT:
                field_schema["default"] = to_builtins(field.default, str_keys=True)
            elif field.default_factory in _FACTORY_DEFAULTS:
                field_schema["default"] = to_builtins(field.default_factory(), str_keys=True)
            names.append(field.encode_name)
            schemas.append(field_schema)
        return names, schemas, required

    def struct(self, schema: dict[str, Any], info: Any) -> None:
        self.describe(schema, info)
        names, fields, required = self.field_schemas(info)
        if info.tag_field is not None:
            names.insert(0, info.tag_field)
            fields.insert(0, {"enum": [info.tag]})
            required.insert(0, info.tag_field)
        if info.array_like:
            trailing_defaults = 0
            for field in reversed(info.fields):
                if field.required:
                    break
                trailing_defaults += 1
            schema["type"] = "array"
            schema["prefixItems"] = fields
            schema["minItems"] = len(fields) - trailing_defaults
            if info.forbid_unknown_fields:
                schema["maxItems"] = len(fields)
        else:
            schema["type"] = "object"
            schema["properties"] = dict(zip(names, fields, strict=True))
            schema["required"] = required
            if info.forbid_unknown_fields:
                schema["additionalProperties"] = False

    def record(self, schema: dict[str, Any], info: Any) -> None:
        self.describe(schema, info)
        names, fields, required = self.field_schemas(info)
        if isinstance(info, self.mi.NamedTupleType):
            schema["type"] = "array"
            schema["prefixItems"] = fields
            schema["minItems"] = len(required)
            schema["maxItems"] = len(fields)
        else:
            schema["type"] = "object"
            schema["properties"] = dict(zip(names, fields, strict=True))
            schema["required"] = required

    def custom(self, schema: dict[str, Any], info: Any) -> dict[str, Any]:
        if self.schema_hook is not None:
            try:
                schema = self.mi._merge_json(self.schema_hook(info.cls), schema)
            except NotImplementedError:
                pass
        if not schema:
            raise TypeError(
                "Generating JSON schema for custom types requires either:\n"
                "- specifying a `schema_hook`\n"
                "- annotating the type with `Meta(extra_json_schema=...)`\n"
                "\n"
                f"type {info.cls!r} is not supported"
            )
        return schema


def _set(schema: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        schema[key] = value


def _base64_length(length: int | None) -> int | None:
    return None if length is None else 4 * ((length + 2) // 3)


def _has_text_keys(mi: ModuleType, key: Any) -> bool:
    if isinstance(key, mi.StrType):
        return True
    if isinstance(key, mi.LiteralType):
        return all(isinstance(value, str) for value in key.values)
    if isinstance(key, mi.EnumType):
        return all(isinstance(member.value, str) for member in key.cls.__members__.values())
    return False
