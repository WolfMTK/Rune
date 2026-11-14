import datetime
import decimal
import enum
import json
import sys
import typing
import uuid
from base64 import b64encode
from collections import namedtuple
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    TypedDict,
    TypeVar,
    Union,
)

import pytest

import rune
from rune import Meta

from .utils import py315_or_later_only, temp_module

if sys.version_info >= (3, 15):
    # This is needed for `ruff` to recognize `frozendict` name
    # and to not raise `F821`:
    from builtins import frozendict

T = TypeVar("T")


def test_any():
    assert rune.json.schema(Any) == {}


def test_raw():
    assert rune.json.schema(rune.Raw) == {}


def test_msgpack_ext():
    with pytest.raises(TypeError):
        assert rune.json.schema(rune.msgpack.Ext)


def test_custom():
    with pytest.raises(TypeError, match="Generating JSON schema for custom types"):
        assert rune.json.schema(complex)

    schema = {"type": "string", "format": "complex"}

    assert rune.json.schema(Annotated[complex, Meta(extra_json_schema=schema)]) == schema


def test_custom_schema_hook():
    schema = {"type": "string", "format": "complex"}

    def schema_hook(cls):
        if cls is complex:
            return schema
        raise NotImplementedError

    assert rune.json.schema(complex, schema_hook=schema_hook) == schema
    assert rune.json.schema(
        Annotated[complex, Meta(extra_json_schema={"title": "A complex field"})],
        schema_hook=schema_hook,
    ) == {**schema, "title": "A complex field"}

    with pytest.raises(TypeError, match="Generating JSON schema for custom types"):
        rune.json.schema(slice, schema_hook=schema_hook)


def test_none():
    assert rune.json.schema(None) == {"type": "null"}


def test_bool():
    assert rune.json.schema(bool) == {"type": "boolean"}


def test_int():
    assert rune.json.schema(int) == {"type": "integer"}


def test_float():
    assert rune.json.schema(float) == {"type": "number"}


def test_string():
    assert rune.json.schema(str) == {"type": "string"}


@pytest.mark.parametrize("typ", [bytes, bytearray, memoryview])
def test_binary(typ):
    assert rune.json.schema(typ) == {
        "type": "string",
        "contentEncoding": "base64",
    }


@pytest.mark.parametrize(
    "annot, extra",
    [
        (None, {}),
        (Meta(tz=None), {}),
        (Meta(tz=True), {"format": "date-time"}),
        (Meta(tz=False), {}),
    ],
)
def test_datetime(annot, extra):
    typ = datetime.datetime
    if annot is not None:
        typ = Annotated[typ, annot]
    assert rune.json.schema(typ) == {"type": "string", **extra}


@pytest.mark.parametrize(
    "annot, extra",
    [
        (None, {}),
        (Meta(tz=None), {}),
        (Meta(tz=True), {"format": "time"}),
        (Meta(tz=False), {"format": "partial-time"}),
    ],
)
def test_time(annot, extra):
    typ = datetime.time
    if annot is not None:
        typ = Annotated[typ, annot]
    assert rune.json.schema(typ) == {"type": "string", **extra}


def test_date():
    assert rune.json.schema(datetime.date) == {
        "type": "string",
        "format": "date",
    }


def test_timedelta():
    assert rune.json.schema(datetime.timedelta) == {
        "type": "string",
        "format": "duration",
    }


def test_uuid():
    assert rune.json.schema(uuid.UUID) == {
        "type": "string",
        "format": "uuid",
    }


def test_decimal():
    assert rune.json.schema(decimal.Decimal) == {
        "type": "string",
        "format": "decimal",
    }


def test_newtype():
    UserId = NewType("UserId", str)
    assert rune.json.schema(UserId) == {"type": "string"}
    assert rune.json.schema(Annotated[UserId, Meta(max_length=10)]) == {
        "type": "string",
        "maxLength": 10,
    }


@pytest.mark.parametrize("typ", [list, tuple, list, tuple])
def test_sequence_any(typ):
    assert rune.json.schema(typ) == {"type": "array"}


@pytest.mark.parametrize("cls", [list, tuple, list, tuple])
def test_sequence_typed(cls):
    args = (int, ...) if cls in (tuple, tuple) else int
    typ = cls[args]
    assert rune.json.schema(typ) == {"type": "array", "items": {"type": "integer"}}


@pytest.mark.parametrize("typ", [set, frozenset, set, frozenset])
def test_set_any(typ):
    assert rune.json.schema(typ) == {"type": "array", "uniqueItems": True}


@pytest.mark.parametrize("cls", [set, frozenset, set, frozenset])
def test_set_typed(cls):
    typ = cls[int]
    assert rune.json.schema(typ) == {
        "type": "array",
        "uniqueItems": True,
        "items": {"type": "integer"},
    }


@pytest.mark.parametrize("cls", [tuple, tuple])
def test_tuple(cls):
    typ = cls[int, float, str]
    assert rune.json.schema(typ) == {
        "type": "array",
        "minItems": 3,
        "maxItems": 3,
        "items": False,
        "prefixItems": [
            {"type": "integer"},
            {"type": "number"},
            {"type": "string"},
        ],
    }


@pytest.mark.parametrize("cls", [tuple, tuple])
def test_empty_tuple(cls):
    typ = cls[()]
    assert rune.json.schema(typ) == {
        "type": "array",
        "minItems": 0,
        "maxItems": 0,
    }


@pytest.mark.parametrize("typ", [dict, dict])
def test_dict_any(typ):
    assert rune.json.schema(typ) == {"type": "object"}


@pytest.mark.parametrize("cls", [dict, dict])
def test_dict_typed(cls):
    typ = cls[str, int]
    assert rune.json.schema(typ) == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }


@py315_or_later_only
def test_frozendict_any():
    assert rune.json.schema(frozendict) == {"type": "object"}


@py315_or_later_only
def test_frozendict_typed():
    typ = frozendict[str, bool]
    assert rune.json.schema(typ) == {
        "type": "object",
        "additionalProperties": {"type": "boolean"},
    }


def test_abstract_sequence():
    # Only testing one here, the main tests are in `test_inspect`
    typ = typing.Sequence[int]
    assert rune.json.schema(typ) == {"type": "array", "items": {"type": "integer"}}


def test_abstract_mapping():
    # Only testing one here, the main tests are in `test_inspect`
    typ = typing.MutableMapping[str, int]
    assert rune.json.schema(typ) == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }


def test_int_enum():
    class Example(enum.IntEnum):
        C = 1
        B = 3
        A = 2

    assert rune.json.schema(Example) == {
        "$ref": "#/$defs/Example",
        "$defs": {"Example": {"title": "Example", "enum": [1, 2, 3]}},
    }


def test_enum():
    class Example(enum.Enum):
        """A docstring"""

        C = "x"
        B = "z"
        A = "y"
        ALIAS = "x"

    assert rune.json.schema(Example) == {
        "$ref": "#/$defs/Example",
        "$defs": {
            "Example": {
                "title": "Example",
                "description": "A docstring",
                "enum": ["x", "y", "z"],
            }
        },
    }


@pytest.mark.parametrize("enum_type", [enum.Enum, enum.IntEnum, enum.Flag, enum.IntFlag])
def test_enum_zero_composite_and_alias_members(enum_type):
    class Example(enum_type):
        NONE = 0
        READ = 1
        WRITE = 2
        BOTH = 3
        BOTH_ALIAS = 3

    assert rune.json.schema(Example) == {
        "$ref": "#/$defs/Example",
        "$defs": {"Example": {"title": "Example", "enum": [0, 1, 2, 3]}},
    }


@pytest.mark.parametrize("enum_type", [enum.Flag, enum.IntFlag])
@pytest.mark.parametrize("value", [0, 3])
def test_enum_single_flag_member(enum_type, value):
    Example = enum_type("Example", {"MEMBER": value})

    assert rune.json.schema(Example) == {
        "$ref": "#/$defs/Example",
        "$defs": {"Example": {"title": "Example", "enum": [value]}},
    }


@pytest.mark.parametrize("value_type,values", [(int, [1, 2]), (str, ["a", "b"])])
def test_enum_unhashable_scalar_values(value_type, values):
    class Value(value_type):
        __hash__ = None

    class Example(enum.Enum):
        A = values[0]
        B = Value(values[1])
        ALIAS = B

    assert rune.json.schema(Example) == {
        "$ref": "#/$defs/Example",
        "$defs": {"Example": {"title": "Example", "enum": values}},
    }


@pytest.mark.parametrize("enum_type", [enum.Enum, enum.IntEnum, enum.Flag, enum.IntFlag])
@pytest.mark.parametrize("value", [10_000, 1 << 60])
def test_enum_large_value_aliases(enum_type, value):
    class Example(enum_type):
        # Construct equal values separately instead of reusing an integer constant.
        A = int(str(value))
        ALIAS = int(str(value))
        ALIAS2 = value

    assert rune.json.schema(Example) == {
        "$ref": "#/$defs/Example",
        "$defs": {"Example": {"title": "Example", "enum": [value]}},
    }


@pytest.mark.parametrize("enum_type", [enum.Flag, enum.IntFlag])
@pytest.mark.parametrize("shift", [0, 60])
def test_enum_flag_expressions(enum_type, shift):
    class Example(enum_type):
        A = 1 << shift
        B = 2 << shift
        C = A | B
        D = ~A
        C_ALIAS = A | B
        D_ALIAS = ~A

    # Python 3.15 normalizes negative flag declarations during class creation.
    values = sorted({Example.D.value, 1 << shift, 2 << shift, 3 << shift})
    assert rune.json.schema(Example) == {
        "$ref": "#/$defs/Example",
        "$defs": {"Example": {"title": "Example", "enum": values}},
    }


def test_enum_unhashable_members():
    class Example(enum.Enum):
        __hash__ = None
        A = 1
        B = 2
        ALIAS = B

    assert rune.json.schema(Example) == {
        "$ref": "#/$defs/Example",
        "$defs": {"Example": {"title": "Example", "enum": [1, 2]}},
    }


def test_int_literal():
    assert rune.json.schema(Literal[3, 1, 2]) == {"enum": [1, 2, 3]}


def test_str_literal():
    assert rune.json.schema(Literal["c", "a", "b"]) == {"enum": ["a", "b", "c"]}


def test_mixed_literal():
    # A Literal may mix value types; building its schema must not crash trying
    # to sort values of incomparable types (gh#1018).
    assert rune.json.schema(Literal[1, None]) == {"enum": [None, 1]}
    assert rune.json.schema(Literal[True, "yes"]) == {"enum": [True, "yes"]}


def test_struct_object():
    class Point(rune.Struct, forbid_unknown_fields=True):
        x: int
        y: int

    class Polygon(rune.Struct):
        """An example docstring"""

        vertices: list[Point]
        name: str | None = None
        metadata: dict[str, str] = {}  # noqa

    assert rune.json.schema(Polygon) == {
        "$ref": "#/$defs/Polygon",
        "$defs": {
            "Polygon": {
                "title": "Polygon",
                "description": "An example docstring",
                "type": "object",
                "properties": {
                    "vertices": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Point"},
                    },
                    "name": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                    },
                    "metadata": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "default": {},
                    },
                },
                "required": ["vertices"],
            },
            "Point": {
                "title": "Point",
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            },
        },
    }


@pytest.mark.parametrize(
    "factory, default",
    [
        (list, []),
        (dict, {}),
        (set, []),
        (bytearray, ""),
    ],
)
def test_struct_default_factory_default(factory, default):
    class Example(rune.Struct):
        x: factory = rune.field(default_factory=factory)

    schema = rune.json.schema(Example)["$defs"]["Example"]
    assert schema["properties"]["x"]["default"] == default
    json.dumps(schema)


@pytest.mark.parametrize("forbid_unknown_fields", [False, True])
def test_struct_array_like(forbid_unknown_fields):
    class Example(rune.Struct, array_like=True, forbid_unknown_fields=forbid_unknown_fields):
        """An example docstring"""

        a: int
        b: str
        c: list[int] = []  # noqa
        d: dict[str, int] = {}  # noqa

    sol = {
        "$ref": "#/$defs/Example",
        "$defs": {
            "Example": {
                "title": "Example",
                "description": "An example docstring",
                "type": "array",
                "prefixItems": [
                    {"type": "integer"},
                    {"type": "string"},
                    {"type": "array", "items": {"type": "integer"}, "default": []},
                    {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                        "default": {},
                    },
                ],
                "minItems": 2,
            }
        },
    }
    if forbid_unknown_fields:
        sol["$defs"]["Example"]["maxItems"] = 4
    assert rune.json.schema(Example) == sol


@pytest.mark.parametrize(
    "tag, min_items, payload",
    [(False, 0, b"[]"), (True, 1, b'["Example"]')],
)
def test_struct_array_like_all_fields_optional(tag, min_items, payload):
    class Example(rune.Struct, array_like=True, tag=tag):
        a: int = 1
        b: list[int] = rune.field(default_factory=list)

    schema = rune.json.schema(Example)["$defs"]["Example"]
    assert rune.json.decode(payload, type=Example) == Example()
    assert schema["minItems"] == min_items


def test_struct_no_fields():
    class Example(rune.Struct):
        pass

    assert rune.json.schema(Example) == {
        "$ref": "#/$defs/Example",
        "$defs": {
            "Example": {
                "title": "Example",
                "type": "object",
                "properties": {},
                "required": [],
            }
        },
    }


def test_struct_object_tagged():
    class Point(rune.Struct, tag=True):
        x: int
        y: int

    assert rune.json.schema(Point) == {
        "$ref": "#/$defs/Point",
        "$defs": {
            "Point": {
                "title": "Point",
                "type": "object",
                "properties": {
                    "type": {"enum": ["Point"]},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["type", "x", "y"],
            }
        },
    }


def test_struct_array_tagged():
    class Point(rune.Struct, tag=True, array_like=True):
        x: int
        y: int

    assert rune.json.schema(Point) == {
        "$ref": "#/$defs/Point",
        "$defs": {
            "Point": {
                "title": "Point",
                "type": "array",
                "prefixItems": [
                    {"enum": ["Point"]},
                    {"type": "integer"},
                    {"type": "integer"},
                ],
                "minItems": 3,
            }
        },
    }


def test_struct_keyword_only():
    class Base(rune.Struct, kw_only=True):
        x: int = 1
        y: int
        z: int = 2

    class Test(Base):
        a: int
        b: int = 0

    assert rune.json.schema(Test) == {
        "$ref": "#/$defs/Test",
        "$defs": {
            "Test": {
                "title": "Test",
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer", "default": 0},
                    "x": {"type": "integer", "default": 1},
                    "y": {"type": "integer"},
                    "z": {"type": "integer", "default": 2},
                },
                "required": ["a", "y"],
            }
        },
    }


def test_struct_array_keyword_only():
    class Base(rune.Struct, kw_only=True, array_like=True):
        x: int = 1
        y: int
        z: int = 2

    class Test(Base):
        a: int
        b: int = 0

    assert rune.json.schema(Test) == {
        "$ref": "#/$defs/Test",
        "$defs": {
            "Test": {
                "title": "Test",
                "type": "array",
                "prefixItems": [
                    {"type": "integer"},
                    {"type": "integer", "default": 0},
                    {"type": "integer", "default": 1},
                    {"type": "integer"},
                    {"type": "integer", "default": 2},
                ],
                "minItems": 4,
            }
        },
    }


def test_typing_namedtuple():
    class Example(NamedTuple):
        """An example docstring"""

        a: str
        b: bool
        c: int = 0

    assert rune.json.schema(Example) == {
        "$ref": "#/$defs/Example",
        "$defs": {
            "Example": {
                "title": "Example",
                "description": "An example docstring",
                "type": "array",
                "prefixItems": [
                    {"type": "string"},
                    {"type": "boolean"},
                    {"type": "integer", "default": 0},
                ],
                "minItems": 2,
                "maxItems": 3,
            }
        },
    }


def test_collections_namedtuple():
    Example = namedtuple("Example", ["a", "b", "c"], defaults=(0,))

    assert rune.json.schema(Example) == {
        "$ref": "#/$defs/Example",
        "$defs": {
            "Example": {
                "title": "Example",
                "type": "array",
                "prefixItems": [{}, {}, {"default": 0}],
                "minItems": 2,
                "maxItems": 3,
            }
        },
    }


def test_generic_namedtuple():
    NamedTuple = pytest.importorskip("typing_extensions").NamedTuple

    class Ex(NamedTuple, Generic[T]):
        """An example docstring"""

        x: T
        y: list[T]

    assert rune.json.schema(Ex) == {
        "$ref": "#/$defs/Ex",
        "$defs": {
            "Ex": {
                "title": "Ex",
                "description": "An example docstring",
                "type": "array",
                "prefixItems": [{}, {"type": "array"}],
                "minItems": 2,
                "maxItems": 2,
            },
        },
    }

    assert rune.json.schema(Ex[int]) == {
        "$ref": "#/$defs/Ex_int_",
        "$defs": {
            "Ex_int_": {
                "title": "Ex[int]",
                "description": "An example docstring",
                "type": "array",
                "prefixItems": [
                    {"type": "integer"},
                    {"type": "array", "items": {"type": "integer"}},
                ],
                "minItems": 2,
                "maxItems": 2,
            },
        },
    }


@pytest.mark.parametrize("use_typing_extensions", [False, True])
def test_typeddict(use_typing_extensions):
    if use_typing_extensions:
        tex = pytest.importorskip("typing_extensions")
        cls = tex.TypedDict
    else:
        cls = TypedDict

    class Example(cls):
        """An example docstring"""

        a: str
        b: bool
        c: int

    assert rune.json.schema(Example) == {
        "$ref": "#/$defs/Example",
        "$defs": {
            "Example": {
                "title": "Example",
                "description": "An example docstring",
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "boolean"},
                    "c": {"type": "integer"},
                },
                "required": ["a", "b", "c"],
            }
        },
    }


@pytest.mark.parametrize("use_typing_extensions", [False, True])
def test_typeddict_optional(use_typing_extensions):
    if use_typing_extensions:
        tex = pytest.importorskip("typing_extensions")
        cls = tex.TypedDict
    else:
        cls = TypedDict

    class Base(cls):
        a: str
        b: bool

    class Example(Base, total=False):
        """An example docstring"""

        c: int

    assert rune.json.schema(Example) == {
        "$ref": "#/$defs/Example",
        "$defs": {
            "Example": {
                "title": "Example",
                "description": "An example docstring",
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "boolean"},
                    "c": {"type": "integer"},
                },
                "required": ["a", "b"],
            }
        },
    }


def test_generic_typeddict():
    TypedDict = pytest.importorskip("typing_extensions").TypedDict

    class Ex(TypedDict, Generic[T]):
        """An example docstring"""

        x: T
        y: list[T]

    assert rune.json.schema(Ex) == {
        "$ref": "#/$defs/Ex",
        "$defs": {
            "Ex": {
                "title": "Ex",
                "description": "An example docstring",
                "type": "object",
                "properties": {
                    "x": {},
                    "y": {"type": "array"},
                },
                "required": ["x", "y"],
            },
        },
    }

    assert rune.json.schema(Ex[int]) == {
        "$ref": "#/$defs/Ex_int_",
        "$defs": {
            "Ex_int_": {
                "title": "Ex[int]",
                "description": "An example docstring",
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["x", "y"],
            },
        },
    }


@pytest.mark.parametrize("module", ["dataclasses", "attrs"])
def test_dataclass_or_attrs(module):
    m = pytest.importorskip(module)
    if module == "attrs":
        decorator = m.define
        factory_default = m.field(factory=dict)
    else:
        decorator = m.dataclass
        factory_default = m.field(default_factory=dict)

    @decorator
    class Point:
        x: int
        y: int

    @decorator
    class Polygon:
        """An example docstring"""

        vertices: list[Point]
        name: str | None = None
        metadata: dict[str, str] = factory_default

    assert rune.json.schema(Polygon) == {
        "$ref": "#/$defs/Polygon",
        "$defs": {
            "Polygon": {
                "title": "Polygon",
                "description": "An example docstring",
                "type": "object",
                "properties": {
                    "vertices": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Point"},
                    },
                    "name": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                    },
                    "metadata": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "default": {},
                    },
                },
                "required": ["vertices"],
            },
            "Point": {
                "title": "Point",
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            },
        },
    }


@pytest.mark.parametrize("module", ["dataclasses", "attrs"])
def test_generic_dataclass_or_attrs(module):
    m = pytest.importorskip(module)
    decorator = m.define if module == "attrs" else m.dataclass

    @decorator
    class Ex(Generic[T]):
        """An example docstring"""

        x: T
        y: list[T]

    assert rune.json.schema(Ex) == {
        "$ref": "#/$defs/Ex",
        "$defs": {
            "Ex": {
                "title": "Ex",
                "description": "An example docstring",
                "type": "object",
                "properties": {
                    "x": {},
                    "y": {"type": "array"},
                },
                "required": ["x", "y"],
            },
        },
    }

    assert rune.json.schema(Ex[int]) == {
        "$ref": "#/$defs/Ex_int_",
        "$defs": {
            "Ex_int_": {
                "title": "Ex[int]",
                "description": "An example docstring",
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["x", "y"],
            },
        },
    }


def test_optional_struct_null_last():
    class Example(rune.Struct):
        x: int

    assert rune.json.schema(Optional[Example]) == {  # noqa
        "anyOf": [{"$ref": "#/$defs/Example"}, {"type": "null"}],
        "$defs": {
            "Example": {
                "title": "Example",
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            }
        },
    }


def test_optional_union_null_last():
    assert rune.json.schema(Union[int, None, str]) == {  # noqa
        "anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}],
    }


@pytest.mark.parametrize("use_union_operator", [False, True])
def test_union(use_union_operator):
    class Example(rune.Struct):
        x: int
        y: int

    if use_union_operator:
        typ = int | str | Example
    else:
        typ = Union[int, str, Example]  # noqa

    assert rune.json.schema(typ) == {
        "anyOf": [
            {"type": "integer"},
            {"type": "string"},
            {"$ref": "#/$defs/Example"},
        ],
        "$defs": {
            "Example": {
                "title": "Example",
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
            }
        },
    }


def test_struct_tagged_union():
    class Point(rune.Struct, tag=True):
        x: int
        y: int

    class Point3D(Point):
        z: int

    assert rune.json.schema(Point | Point3D) == {
        "anyOf": [{"$ref": "#/$defs/Point"}, {"$ref": "#/$defs/Point3D"}],
        "discriminator": {
            "mapping": {"Point": "#/$defs/Point", "Point3D": "#/$defs/Point3D"},
            "propertyName": "type",
        },
        "$defs": {
            "Point": {
                "properties": {
                    "type": {"enum": ["Point"]},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["type", "x", "y"],
                "title": "Point",
                "type": "object",
            },
            "Point3D": {
                "properties": {
                    "type": {"enum": ["Point3D"]},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "z": {"type": "integer"},
                },
                "required": ["type", "x", "y", "z"],
                "title": "Point3D",
                "type": "object",
            },
        },
    }
    assert rune.json.schema(Point | Point3D) == rune.json.schema(Union[Point, Point3D])  # noqa


def test_struct_tagged_union_with_none():
    class Point(rune.Struct, tag=True):
        x: int
        y: int

    class Point3D(Point):
        z: int

    schema = rune.json.schema(Union[Point, Point3D, None])  # noqa
    assert schema["anyOf"] == [
        {
            "anyOf": [{"$ref": "#/$defs/Point"}, {"$ref": "#/$defs/Point3D"}],
            "discriminator": {
                "mapping": {"Point": "#/$defs/Point", "Point3D": "#/$defs/Point3D"},
                "propertyName": "type",
            },
        },
        {"type": "null"},
    ]
    assert "discriminator" not in schema


def test_struct_tagged_union_mixed_types():
    class Point(rune.Struct, tag=True):
        x: int
        y: int

    class Point3D(Point):
        z: int

    assert rune.json.schema(Point | Point3D | int | float) == {
        "anyOf": [
            {"type": "integer"},
            {"type": "number"},
            {
                "anyOf": [{"$ref": "#/$defs/Point"}, {"$ref": "#/$defs/Point3D"}],
                "discriminator": {
                    "mapping": {"Point": "#/$defs/Point", "Point3D": "#/$defs/Point3D"},
                    "propertyName": "type",
                },
            },
        ],
        "$defs": {
            "Point": {
                "properties": {
                    "type": {"enum": ["Point"]},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["type", "x", "y"],
                "title": "Point",
                "type": "object",
            },
            "Point3D": {
                "properties": {
                    "type": {"enum": ["Point3D"]},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "z": {"type": "integer"},
                },
                "required": ["type", "x", "y", "z"],
                "title": "Point3D",
                "type": "object",
            },
        },
    }


def test_struct_tagged_union_with_none_and_other():
    class Point(rune.Struct, tag=True):
        x: int
        y: int

    class Point3D(Point):
        z: int

    schema = rune.json.schema(Union[Point, Point3D, int, None])  # noqa
    assert schema["anyOf"] == [
        {"type": "integer"},
        {
            "anyOf": [{"$ref": "#/$defs/Point"}, {"$ref": "#/$defs/Point3D"}],
            "discriminator": {
                "mapping": {"Point": "#/$defs/Point", "Point3D": "#/$defs/Point3D"},
                "propertyName": "type",
            },
        },
        {"type": "null"},
    ]
    assert "discriminator" not in schema


def test_optional_union_null_member_metadata_preserved():
    NullWithMeta = Annotated[None, rune.Meta(description="explicitly unset")]

    assert rune.json.schema(Union[int, NullWithMeta]) == {  # noqa
        "anyOf": [
            {"type": "integer"},
            {"type": "null", "description": "explicitly unset"},
        ],
    }


def test_struct_array_union():
    class Point(rune.Struct, array_like=True, tag=True):
        x: int
        y: int

    class Point3D(Point):
        z: int

    assert rune.json.schema(Point | Point3D) == {
        "anyOf": [{"$ref": "#/$defs/Point"}, {"$ref": "#/$defs/Point3D"}],
        "$defs": {
            "Point": {
                "minItems": 3,
                "prefixItems": [
                    {"enum": ["Point"]},
                    {"type": "integer"},
                    {"type": "integer"},
                ],
                "title": "Point",
                "type": "array",
            },
            "Point3D": {
                "minItems": 4,
                "prefixItems": [
                    {"enum": ["Point3D"]},
                    {"type": "integer"},
                    {"type": "integer"},
                    {"type": "integer"},
                ],
                "title": "Point3D",
                "type": "array",
            },
        },
    }


def test_struct_unset_fields():
    class Ex(rune.Struct):
        x: int | rune.UnsetType = rune.UNSET

    assert rune.json.schema(Ex) == {
        "$ref": "#/$defs/Ex",
        "$defs": {
            "Ex": {
                "properties": {"x": {"type": "integer"}},
                "required": [],
                "title": "Ex",
                "type": "object",
            }
        },
    }


def test_generic_struct():
    class Ex(rune.Struct, Generic[T]):
        """An example docstring"""

        x: T
        y: list[T]

    assert rune.json.schema(Ex) == {
        "$ref": "#/$defs/Ex",
        "$defs": {
            "Ex": {
                "title": "Ex",
                "description": "An example docstring",
                "type": "object",
                "properties": {
                    "x": {},
                    "y": {"type": "array"},
                },
                "required": ["x", "y"],
            },
        },
    }

    assert rune.json.schema(Ex[int]) == {
        "$ref": "#/$defs/Ex_int_",
        "$defs": {
            "Ex_int_": {
                "title": "Ex[int]",
                "description": "An example docstring",
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["x", "y"],
            },
        },
    }


def test_generic_struct_tagged_union():
    class Point(rune.Struct, Generic[T], tag=True):
        x: T
        y: T

    class Point3D(Point[T]):
        z: T

    sol = {
        "anyOf": [{"$ref": "#/$defs/Point_int_"}, {"$ref": "#/$defs/Point3D_int_"}],
        "discriminator": {
            "mapping": {
                "Point": "#/$defs/Point_int_",
                "Point3D": "#/$defs/Point3D_int_",
            },
            "propertyName": "type",
        },
        "$defs": {
            "Point_int_": {
                "properties": {
                    "type": {"enum": ["Point"]},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["type", "x", "y"],
                "title": "Point[int]",
                "type": "object",
            },
            "Point3D_int_": {
                "properties": {
                    "type": {"enum": ["Point3D"]},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "z": {"type": "integer"},
                },
                "required": ["type", "x", "y", "z"],
                "title": "Point3D[int]",
                "type": "object",
            },
        },
    }
    res = rune.json.schema(Point[int] | Point3D[int])
    assert res == sol


@pytest.mark.parametrize(
    "field, constraint",
    [
        ("ge", "minimum"),
        ("gt", "exclusiveMinimum"),
        ("le", "maximum"),
        ("lt", "exclusiveMaximum"),
        ("multiple_of", "multipleOf"),
    ],
)
def test_numeric_metadata(field, constraint):
    typ = Annotated[int, Meta(**{field: 2})]
    assert rune.json.schema(typ) == {"type": "integer", constraint: 2}


@pytest.mark.parametrize(
    "field, val, constraint",
    [
        ("pattern", "[a-z]*", "pattern"),
        ("min_length", 0, "minLength"),
        ("max_length", 3, "maxLength"),
    ],
)
def test_string_metadata(field, val, constraint):
    typ = Annotated[str, Meta(**{field: val})]
    assert rune.json.schema(typ) == {"type": "string", constraint: val}


@pytest.mark.parametrize(
    "field, val, constraint",
    [
        ("pattern", "[a-z]*", "pattern"),
        ("min_length", 0, "minLength"),
        ("max_length", 3, "maxLength"),
    ],
)
def test_dict_key_metadata(field, val, constraint):
    typ = Annotated[str, Meta(**{field: val})]
    assert rune.json.schema(dict[typ, int]) == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
        "propertyNames": {constraint: val},
    }


@pytest.mark.parametrize(
    "meta, property_names",
    [
        (Meta(title="key"), {"title": "key"}),
        (
            Meta(title="key", pattern="^A$"),
            {"title": "key", "pattern": "^A$"},
        ),
    ],
)
def test_dict_key_metadata_with_schema_metadata(meta, property_names):
    typ = Annotated[str, meta]
    assert rune.json.schema(dict[typ, int]) == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
        "propertyNames": property_names,
    }


@pytest.mark.parametrize("typ", [bytes, bytearray, memoryview])
@pytest.mark.parametrize(
    "field, n, constraint",
    [("min_length", 2, "minLength"), ("max_length", 7, "maxLength")],
)
def test_binary_metadata(typ, field, n, constraint):
    n2 = len(b64encode(b"x" * n))
    typ = Annotated[typ, Meta(**{field: n})]
    assert rune.json.schema(typ) == {
        "type": "string",
        constraint: n2,
        "contentEncoding": "base64",
    }


@pytest.mark.parametrize("typ", [list, tuple])
@pytest.mark.parametrize(
    "field, constraint",
    [("min_length", "minItems"), ("max_length", "maxItems")],
)
def test_array_metadata(typ, field, constraint):
    typ = Annotated[typ, Meta(**{field: 2})]
    assert rune.json.schema(typ) == {"type": "array", constraint: 2}


@pytest.mark.parametrize("typ", [set, frozenset])
@pytest.mark.parametrize(
    "field, constraint",
    [("min_length", "minItems"), ("max_length", "maxItems")],
)
def test_set_metadata(typ, field, constraint):
    typ = Annotated[typ[int], Meta(**{field: 2})]
    assert rune.json.schema(typ) == {
        "type": "array",
        "uniqueItems": True,
        "items": {"type": "integer"},
        constraint: 2,
    }


@pytest.mark.parametrize(
    "field, constraint",
    [("min_length", "minProperties"), ("max_length", "maxProperties")],
)
def test_object_metadata(field, constraint):
    typ = Annotated[dict, Meta(**{field: 2})]
    assert rune.json.schema(typ) == {"type": "object", constraint: 2}


def test_generic_metadata():
    typ = Annotated[
        int,
        Meta(
            title="the title",
            description="the description",
            examples=[1, 2, 3],
            extra_json_schema={"title": "an override", "default": 1},
        ),
    ]
    assert rune.json.schema(typ) == {
        "type": "integer",
        "title": "an override",
        "description": "the description",
        "examples": [1, 2, 3],
        "default": 1,
    }


def test_component_names_collide():
    s1 = """
    import rune
    Ex = rune.defstruct("Ex", [("x", int), ("y", int)])
    """

    s2 = """
    import rune
    Ex = rune.defstruct("Ex", [("a", str), ("b", str)])
    """

    with temp_module(s1) as m1, temp_module(s2) as m2:
        (r1, r2), components = rune.json.schema_components([m1.Ex, m2.Ex])

    assert r1 == {"$ref": f"#/$defs/{m1.__name__}.Ex"}
    assert r2 == {"$ref": f"#/$defs/{m2.__name__}.Ex"}
    assert components == {
        f"{m1.__name__}.Ex": {
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
            "title": "Ex",
            "type": "object",
        },
        f"{m2.__name__}.Ex": {
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a", "b"],
            "title": "Ex",
            "type": "object",
        },
    }


def test_schema_components_collects_subtypes():
    class ExEnum(enum.Enum):
        A = 1

    class ExStruct(rune.Struct):
        b: set[frozenset[ExEnum]] | int

    class ExDict(TypedDict):
        c: tuple[ExStruct, ...]

    class ExTuple(NamedTuple):
        d: list[ExDict]

    @dataclass
    class ExDataclass:
        e: list[ExTuple]

    (s,), components = rune.json.schema_components([dict[str, ExDataclass]])

    r1 = {"$ref": "#/$defs/ExEnum"}
    r2 = {"$ref": "#/$defs/ExStruct"}
    r3 = {"$ref": "#/$defs/ExDict"}
    r4 = {"$ref": "#/$defs/ExTuple"}
    r5 = {"$ref": "#/$defs/ExDataclass"}

    assert s == {"type": "object", "additionalProperties": r5}
    assert components == {
        "ExEnum": {"enum": [1], "title": "ExEnum"},
        "ExStruct": {
            "type": "object",
            "title": "ExStruct",
            "properties": {
                "b": {
                    "anyOf": [
                        {
                            "items": {
                                "items": r1,
                                "type": "array",
                                "uniqueItems": True,
                            },
                            "type": "array",
                            "uniqueItems": True,
                        },
                        {"type": "integer"},
                    ]
                }
            },
            "required": ["b"],
        },
        "ExDict": {
            "title": "ExDict",
            "type": "object",
            "properties": {"c": {"items": r2, "type": "array"}},
            "required": ["c"],
        },
        "ExTuple": {
            "title": "ExTuple",
            "type": "array",
            "prefixItems": [{"items": r3, "type": "array"}],
            "maxItems": 1,
            "minItems": 1,
        },
        "ExDataclass": {
            "title": "ExDataclass",
            "type": "object",
            "properties": {"e": {"items": r4, "type": "array"}},
            "required": ["e"],
        },
    }


def test_ref_template():
    class Ex1(rune.Struct):
        a: int

    class Ex2(rune.Struct):
        b: Ex1

    (s1, s2), components = rune.json.schema_components(
        [Ex1, Ex2], ref_template="#/definitions/{name}"
    )

    assert s1 == {"$ref": "#/definitions/Ex1"}
    assert s2 == {"$ref": "#/definitions/Ex2"}

    assert components == {
        "Ex1": {
            "title": "Ex1",
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        },
        "Ex2": {
            "title": "Ex2",
            "type": "object",
            "properties": {"b": s1},
            "required": ["b"],
        },
    }


def test_multiline_docstring():
    class Example(rune.Struct):
        """
            indented first line

        last line.
        """

        pass

    assert rune.json.schema(Example) == {
        "$ref": "#/$defs/Example",
        "$defs": {
            "Example": {
                "description": "    indented first line\n\nlast line.",
                "title": "Example",
                "type": "object",
                "properties": {},
                "required": [],
            }
        },
    }
