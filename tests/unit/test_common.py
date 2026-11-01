import base64
import collections
import datetime
import decimal
import enum
import gc
import sys
import textwrap
import types
import typing
import uuid
import weakref
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, make_dataclass
from types import ModuleType
from typing import Any, Literal

import pytest
from rune.struct import Struct

import rune
from rune import UNSET, UnsetType, ValidationError

from .utils import (
    PY311,
    PY312,
    emscripten_stack_limited,
    max_call_depth,
    py312_plus,
    py315_or_later_only,
    temp_module,
    win_arm64_py312_stack_limited,
)

if sys.version_info >= (3, 15):
    from builtins import frozendict


class TestEncodeSubclasses:
    def test_encode_dict_subclass(self, proto: Any) -> None:
        class subclass(dict):
            pass

        for msg in [{}, {"a": 1, "b": 2}]:
            assert proto.encode(subclass(msg)) == proto.encode(msg)

    @pytest.mark.parametrize("cls", [list, tuple, set, frozenset])
    def test_encode_sequence_subclass(
        self, cls: list | tuple | set | frozenset, proto: Any
    ) -> None:
        class subclass(cls):
            pass

        for msg in [[], [1, 2]]:
            assert proto.encode(subclass(msg)) == proto.encode(cls(msg))


class Custom:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class TestDecoder:
    def test_decoder_runtime_type_parameters(self, proto: ModuleType) -> None:
        dec = proto.Decoder[int](int)
        assert isinstance(dec, proto.Decoder)
        msg = proto.encode(2)
        assert dec.decode(msg) == 2

    def test_decoder_dec_hook_attribute(self, proto: ModuleType) -> None:
        def dec_hook(typ: type, obj: Any) -> None:
            pass

        dec = proto.Decoder()
        assert dec.dec_hook is None

        dec = proto.Decoder(dec_hook=None)
        assert dec.dec_hook is None

        dec = proto.Decoder(dec_hook=dec_hook)
        assert dec.dec_hook is dec_hook

    def test_decoder_dec_hook_not_callable(self, proto: ModuleType) -> None:
        with pytest.raises(TypeError):
            proto.Decoder(dec_hook=1)

    def test_decode_dec_hook(self, proto: ModuleType) -> None:
        def dec_hook(typ: type, obj: Any) -> Any:
            assert typ is Custom
            return typ(*obj)

        msg = proto.encode([1, 2])
        res = proto.decode(msg, type=Custom, dec_hook=dec_hook)
        assert res == Custom(1, 2)
        assert isinstance(res, Custom)

    def test_decoder_dec_hook(self, proto: ModuleType) -> None:
        called = False

        def dec_hook(typ: type, obj: Any) -> Custom:
            nonlocal called
            called = True
            assert typ is Custom
            return Custom(*obj)

        dec = proto.Decoder(type=list[Custom], dec_hook=dec_hook)
        buf = proto.encode([[1, 2], [3, 4], [5, 6]])
        msg = dec.decode(buf)
        assert called
        assert msg == [Custom(1, 2), Custom(3, 4), Custom(5, 6)]
        assert isinstance(msg[0], Custom)

    def test_decoder_dec_hook_optional_custom_type(self, proto: ModuleType) -> None:
        called = False

        def dec_hook(typ: type, obj: Any) -> None:
            nonlocal called
            called = True

        dec = proto.Decoder(type=Custom | None, dec_hook=dec_hook)
        msg = dec.decode(proto.encode(None))
        assert not called
        assert msg is None

    @pytest.mark.parametrize("err_cls", [TypeError, ValueError])
    def test_decode_dec_hook_errors_wrapped(
        self, err_cls: type[Exception], proto: ModuleType
    ) -> None:
        def dec_hook(typ: type, obj: Any) -> None:
            assert obj == "some string"
            raise err_cls("Oh no!")

        msg = proto.encode("some string")
        with pytest.raises(rune.ValidationError, match="Oh no!") as rec:
            proto.decode(msg, type=Custom, dec_hook=dec_hook)

        assert rec.value.__cause__ is rec.value.__context__
        assert type(rec.value.__cause__) is err_cls

        msg = proto.encode(["some string"])
        with pytest.raises(rune.ValidationError, match=r"Oh no! - at `\$\[0\]`"):
            proto.decode(msg, type=list[Custom], dec_hook=dec_hook)

    def test_decode_dec_hook_errors_passthrough(self, proto: ModuleType) -> None:
        def dec_hook(typ: type, obj: Any) -> None:
            assert obj == "some string"
            raise NotImplementedError("Oh no!")

        msg = proto.encode("some string")
        with pytest.raises(NotImplementedError, match="Oh no!"):
            proto.decode(msg, type=Custom, dec_hook=dec_hook)

        msg = proto.encode(["some string"])
        with pytest.raises(NotImplementedError, match=r"Oh no!"):
            proto.decode(msg, type=list[Custom], dec_hook=dec_hook)

    def test_decode_dec_hook_wrong_type(self, proto: ModuleType) -> None:
        dec = proto.Decoder(type=Custom, dec_hook=lambda t, o: o)

        msg = proto.encode([1, 2])
        with pytest.raises(
            rune.ValidationError,
            match="Expected `Custom`, got `list`",
        ):
            dec.decode(msg)

    def test_decode_dec_hook_wrong_type_in_struct(self, proto: ModuleType) -> None:
        class Test(Struct):
            point: Custom
            other: int

        dec = proto.Decoder(type=Test, dec_hook=lambda t, o: o)

        msg = proto.encode({"point": [1, 2], "other": 3})
        with pytest.raises(rune.ValidationError) as rec:
            dec.decode(msg)

        assert "Expected `Custom`, got `list` - at `$.point`" == str(rec.value)

    def test_decode_dec_hook_wrong_type_generic(self, proto: ModuleType) -> None:
        dec = proto.Decoder(type=deque[int], dec_hook=lambda t, o: o)

        msg = proto.encode([1, 2, 3])
        with pytest.raises(rune.ValidationError) as rec:
            dec.decode(msg)

        assert "Expected `collections.deque`, got `list`" == str(rec.value)

    def test_decode_dec_hook_isinstance_errors(self, proto: ModuleType) -> None:
        class Metaclass(type):
            def __instancecheck__(self, obj: Any) -> bool:
                raise TypeError("Oh no!")

        class Custom(metaclass=Metaclass):
            pass

        dec = proto.Decoder(type=Custom)

        msg = proto.encode(1)
        with pytest.raises(TypeError, match="Oh no!"):
            dec.decode(msg)

    @pytest.mark.parametrize("base_error_cls", [rune.ValidationError, rune.DecodeError])
    def test_dec_hook_validation_error_subclass_propagates_directly(
        self, proto: ModuleType, base_error_cls: type[Exception]
    ) -> None:
        class MyError(base_error_cls):
            pass

        class Foo:
            pass

        def dec_hook(tp: type, val: Any) -> None:
            raise MyError("custom message")

        dec = proto.Decoder(type=Foo, dec_hook=dec_hook)

        with pytest.raises(MyError, match="custom message"):
            dec.decode(b"[]")


@pytest.mark.skipif(
    PY312,
    reason=(
        "Python 3.12 harcodes the C recursion limit, making this behavior harder to test in CI"
    ),
)
class TestRecursion:
    @staticmethod
    def nested(n, is_array):
        if is_array:
            obj = []
            for _ in range(n):
                obj = [obj]
        else:
            obj = {}
            for _ in range(n):
                obj = {"": obj}
        return obj

    @pytest.mark.parametrize("is_array", [True, False])
    def test_encode_highly_recursive_msg_errors(self, is_array, proto):
        N = 200
        obj = self.nested(N, is_array)

        # Errors if above the recursion limit
        with max_call_depth(N // 2):
            with pytest.raises(RecursionError):
                proto.encode(obj)

        # Works if below the recursion limit
        with max_call_depth(N * 2):
            proto.encode(obj)

    @pytest.mark.parametrize("is_array", [True, False])
    def test_decode_highly_recursive_msg_errors(self, is_array, proto):
        """Ensure recursion is properly handled when decoding.
        Test case seen in https://github.com/ijl/orjson/issues/458."""
        N = 200
        obj = self.nested(N, is_array)

        with max_call_depth(N * 2):
            msg = proto.encode(obj)

        # Errors if above the recursion limit
        with max_call_depth(N // 2):
            with pytest.raises(RecursionError):
                proto.decode(msg)

        # Works if below the recursion limit
        with max_call_depth(N * 2):
            obj2 = proto.decode(msg)

        assert obj2


class TestThreadSafe:
    def test_encode_threadsafe(self, proto: ModuleType) -> None:
        class Nested:
            def __init__(self, x: int) -> None:
                self.x = x

        def enc_hook(obj: Nested) -> str:
            return base64.b64encode(enc.encode(obj.x)).decode("utf-8")

        enc = proto.Encoder(enc_hook=enc_hook)
        res = enc.encode({"x": Nested(1)})
        sol = proto.encode({"x": base64.b64encode(proto.encode(1)).decode("utf-8")})
        assert res == sol

    def test_decode_threadsafe(self, proto: ModuleType) -> None:
        class Custom:
            def __init__(self, node: Any) -> None:
                self.node = node

            def __eq__(self, other: object) -> bool:
                return type(other) is Custom and self.node == other.node

        def dec_hook(typ: type, obj: str) -> Custom:
            msg = base64.b64decode(obj)
            return Custom(dec.decode(msg))

        dec = proto.Decoder(tuple[Custom | None, int], dec_hook=dec_hook)
        msg = proto.encode((base64.b64encode(proto.encode((None, 1))).decode("utf-8"), 2))
        sol = (Custom((None, 1)), 2)
        res = dec.decode(msg)
        assert res == sol


class FruitInt(enum.IntEnum):
    APPLE = 1
    BANANA = 2


class TestIntEnum:
    def test_empty_errors(self, proto: ModuleType) -> None:
        class Empty(enum.IntEnum):
            pass

        with pytest.raises(TypeError, match="Enum types must have at least one item"):
            proto.Decoder(Empty)

    @pytest.mark.parametrize("base_cls", [enum.IntEnum, enum.Enum])
    def test_encode(self, proto: ModuleType, base_cls: type[enum.Enum]) -> None:
        class Test(base_cls):  # type: ignore[misc, valid-type]
            A = 1
            B = 2

        assert proto.encode(Test.A) == proto.encode(1)

    @pytest.mark.parametrize("base_cls", [enum.IntEnum, enum.Enum])
    def test_decode(self, proto: ModuleType, base_cls: type[enum.Enum]) -> None:
        class Test(base_cls):  # type: ignore[misc, valid-type]
            A = 1
            B = 2

        dec = proto.Decoder(Test)
        assert dec.decode(proto.encode(1)) is Test.A
        assert dec.decode(proto.encode(2)) is Test.B

        with pytest.raises(ValidationError, match="Invalid enum value 3"):
            dec.decode(proto.encode(3))

    def test_decode_nested(self, proto: ModuleType) -> None:
        class Test(Struct):
            fruit: FruitInt

        dec = proto.Decoder(Test)

        assert dec.decode(proto.encode({"fruit": 1})) == Test(FruitInt.APPLE)

        with pytest.raises(ValidationError, match=r"Invalid enum value 3 - at `\$.fruit`"):
            dec.decode(proto.encode({"fruit": 3}))

    def test_intenum_missing(self, proto: ModuleType) -> None:
        class Ex(enum.IntEnum):
            A = 1
            B = 2

            @classmethod
            def _missing_(cls, val: object) -> Any:
                if val == 3:
                    return cls.A
                elif val == -4:
                    return cls.B
                elif val == 5:
                    raise ValueError("oh no!")
                else:
                    return None

        dec = proto.Decoder(Ex)

        def roundtrip(msg: int) -> Any:
            return dec.decode(proto.encode(msg))

        assert roundtrip(1) is Ex.A
        assert roundtrip(3) is Ex.A
        assert roundtrip(-4) is Ex.B
        with pytest.raises(ValidationError, match="Invalid enum value 5"):
            roundtrip(5)
        with pytest.raises(ValidationError, match="Invalid enum value 6"):
            roundtrip(6)

    def test_intflag(self, proto: ModuleType) -> None:
        class Ex(enum.IntFlag):
            A = 0b001
            B = 0b010
            C = 0b100

        obj = Ex.A | Ex.C
        msg = proto.encode(obj)
        assert msg == proto.encode(int(obj))
        assert proto.decode(msg, type=Ex) == obj

    def test_int_lookup_reused(self) -> None:
        class Test(enum.IntEnum):
            A = 1
            B = 2

        dec = rune.msgpack.Decoder(Test)  # noqa: F841
        count = sys.getrefcount(Test.__rune_cache__)  # type: ignore[attr-defined]
        dec2 = rune.msgpack.Decoder(Test)
        count2 = sys.getrefcount(Test.__rune_cache__)  # type: ignore[attr-defined]
        assert count2 == count + 1

        # Reference count decreases when decoder is dropped
        del dec2
        gc.collect()
        count3 = sys.getrefcount(Test.__rune_cache__)  # type: ignore[attr-defined]
        assert count == count3

    def test_int_lookup_gc(self) -> None:
        class Test(enum.IntEnum):
            A = 1
            B = 2

        dec = rune.msgpack.Decoder(Test)
        assert gc.is_tracked(Test.__rune_cache__)  # type: ignore[attr-defined]

        # Deleting all references and running GC cleans up cycle
        ref = weakref.ref(Test)
        del Test
        del dec
        gc.collect()
        assert ref() is None

    @pytest.mark.parametrize(
        "values",
        [
            [0, 1, 2, -(2**63) - 1],
            [0, 1, 2, 2**63],
        ],
    )
    def test_int_lookup_values_out_of_range(self, values: list[int]) -> None:
        myenum = enum.IntEnum("myenum", [(f"x{i}", v) for i, v in enumerate(values)])

        with pytest.raises(NotImplementedError):
            rune.msgpack.Decoder(myenum)

    def test_msgspec_cache_overwritten(self) -> None:
        class Test(enum.IntEnum):
            A = 1

        Test.__rune_cache__ = 1  # type: ignore[attr-defined]

        with pytest.raises(RuntimeError, match="__rune_cache__"):
            rune.msgpack.Decoder(Test)

    @pytest.mark.parametrize(
        "values",
        [
            [0],
            [1],
            [-1],
            [3, 4, 5, 2, 1],
            [4, 3, 1, 2, 7],
            [-4, -3, -2, -1, 0, 1, 2, 3, 4],
            [-4, -3, -1, -2, -7],
            [-4, -3, 1, 0, -2, -1],
            [2**63 - 1, 2**63 - 2, 2**63 - 3],
            [-(2**63) + 1, -(2**63) + 2, -(2**63) + 3],
        ],
    )
    def test_compact(self, values: list[int]) -> None:
        myenum = enum.IntEnum("myenum", [(f"x{i}", v) for i, v in enumerate(values)])
        dec = rune.msgpack.Decoder(myenum)

        assert hasattr(myenum, "__rune_cache__")

        for val in myenum:
            msg = rune.msgpack.encode(val)
            val2 = dec.decode(msg)
            assert val == val2

        for bad in [-1000, min(values) - 1, max(values) + 1, 1000]:
            with pytest.raises(ValidationError):
                dec.decode(rune.msgpack.encode(bad))

    @pytest.mark.parametrize(
        "values",
        [
            [-(2**63), 2**63 - 1, 0],
            [2**63 - 2, 2**63 - 3, 2**63 - 1],
            [2**63 - 2, 2**63 - 3, 2**63 - 1, 0, 2, 3, 4, 5, 6],
        ],
    )
    def test_hashtable(self, values: list[int]) -> None:
        myenum = enum.IntEnum("myenum", [(f"x{i}", v) for i, v in enumerate(values)])
        dec = rune.msgpack.Decoder(myenum)

        assert hasattr(myenum, "__rune_cache__")

        for val in myenum:
            msg = rune.msgpack.encode(val)
            val2 = dec.decode(msg)
            assert val == val2

        for bad in [-2000, -1, 1, 2000]:
            with pytest.raises(ValidationError):
                dec.decode(rune.msgpack.encode(bad))

    @pytest.mark.parametrize(
        "values",
        [
            [8, 16, 24, 32, 40, 48],
            [-8, -16, -24, -32, -40, -48],
        ],
    )
    def test_hashtable_collisions(self, values: list[int]) -> None:
        myenum = enum.IntEnum("myenum", [(f"x{i}", v) for i, v in enumerate(values)])
        dec = rune.msgpack.Decoder(myenum)

        for val in myenum:
            msg = rune.msgpack.encode(val)
            val2 = dec.decode(msg)
            assert val == val2

        for bad in [0, 7, 9, 56, -min(values), -max(values), 2**64 - 1, -(2**63)]:
            with pytest.raises(ValidationError):
                dec.decode(rune.msgpack.encode(bad))


class FruitStr(enum.Enum):
    APPLE = "apple"
    BANANA = "banana"


class TestEnum:
    def test_empty_errors(self, proto: ModuleType) -> None:
        class Empty(enum.Enum):
            pass

        with pytest.raises(TypeError, match="Enum types must have at least one item"):
            proto.Decoder(Empty)

    def test_encode_complex(self, proto: ModuleType) -> None:
        class Complex(enum.Enum):
            A = 1.5

        res = proto.encode(Complex.A)
        sol = proto.encode(1.5)
        assert res == sol

        res = proto.encode({Complex.A: 1})
        sol = proto.encode({1.5: 1})
        assert res == sol

    def test_decode_complex_errors(self, proto: ModuleType) -> None:
        class Complex(enum.Enum):
            A = 1.5

        with pytest.raises(TypeError) as rec:
            proto.Decoder(Complex)

        assert "Enums must contain either all str or all int values" in str(rec.value)
        assert repr(Complex) in str(rec.value)

    @pytest.mark.parametrize(
        "values",
        [
            [("A", 1), ("B", 2), ("C", "c")],
            [("A", "a"), ("B", "b"), ("C", 3)],
        ],
    )
    def test_mixed_value_types_errors(
        self, values: list[tuple[str, int | str]], proto: ModuleType
    ) -> None:
        Bad = enum.Enum("Bad", values)  # type: ignore[misc]

        with pytest.raises(TypeError) as rec:
            proto.Decoder(Bad)

        assert "Enums must contain either all str or all int values" in str(rec.value)
        assert repr(Bad) in str(rec.value)

    @pytest.mark.parametrize("base_cls", [enum.StrEnum, enum.Enum])
    def test_encode(self, proto: ModuleType, base_cls: type[enum.Enum]) -> None:
        class Test(base_cls):  # type: ignore[misc, valid-type]
            A = "apple"
            B = "banana"

        assert proto.encode(Test.A) == proto.encode("apple")

    @pytest.mark.parametrize("base_metacls", [enum.EnumMeta] + ([enum.EnumType] if PY311 else []))
    def test_enum_with_custom_enum_metaclass(
        self, proto: ModuleType, base_metacls: type[enum.EnumMeta]
    ) -> None:
        class ChoicesMeta(base_metacls):  # type: ignore[misc, valid-type]
            """My custom enum metaclass"""

        class Test(enum.Enum, metaclass=ChoicesMeta):
            A = "apple"
            B = "banana"

        assert proto.encode(Test.A) == proto.encode("apple")
        assert proto.decode(proto.encode("apple"), type=Test) is Test.A

    @pytest.mark.parametrize("base_cls", [enum.StrEnum, enum.Enum])
    def test_decode(self, proto: ModuleType, base_cls: type[enum.Enum]) -> None:
        class Test(base_cls):  # type: ignore[misc, valid-type]
            A = "apple"
            B = "banana"

        dec = proto.Decoder(Test)
        assert dec.decode(proto.encode("apple")) is Test.A
        assert dec.decode(proto.encode("banana")) is Test.B

        with pytest.raises(ValidationError, match="Invalid enum value 'cherry'"):
            dec.decode(proto.encode("cherry"))

    def test_decode_nested(self, proto: ModuleType) -> None:
        class Test(Struct):
            fruit: FruitStr

        dec = proto.Decoder(Test)

        assert dec.decode(proto.encode({"fruit": "apple"})) == Test(FruitStr.APPLE)

        with pytest.raises(
            ValidationError,
            match=r"Invalid enum value 'cherry' - at `\$.fruit`",
        ):
            dec.decode(proto.encode({"fruit": "cherry"}))

    def test_str_lookup_reused(self) -> None:
        class Test(enum.Enum):
            A = "a"
            B = "b"

        dec: rune.msgpack.Decoder[Any] = rune.msgpack.Decoder(Test)  # noqa
        count = sys.getrefcount(Test.__rune_cache__)  # type: ignore[attr-defined]
        dec2: rune.msgpack.Decoder[Any] = rune.msgpack.Decoder(Test)
        count2 = sys.getrefcount(Test.__rune_cache__)  # type: ignore[attr-defined]
        assert count2 == count + 1

        # Reference count decreases when decoder is dropped
        del dec2
        gc.collect()
        count3 = sys.getrefcount(Test.__rune_cache__)  # type: ignore[attr-defined]
        assert count == count3

    def test_str_lookup_gc(self) -> None:
        class Test(enum.Enum):
            A = "a"
            B = "b"

        dec: rune.msgpack.Decoder[Any] = rune.msgpack.Decoder(Test)
        assert gc.is_tracked(Test.__rune_cache__)  # type: ignore[attr-defined]

        # Deleting all references and running GC cleans up cycle
        ref = weakref.ref(Test)
        del Test
        del dec
        gc.collect()
        assert ref() is None

    def test_msgspec_cache_overwritten(self) -> None:
        class Test(enum.Enum):
            A = 1

        Test.__rune_cache__ = 1  # type: ignore[attr-defined]

        with pytest.raises(RuntimeError, match="__rune_cache__"):
            rune.msgpack.Decoder(Test)

    @pytest.mark.parametrize("length", [2, 8, 16])
    @pytest.mark.parametrize("nitems", [1, 3, 6, 12, 24, 48])
    def test_random_enum_same_lengths(self, rand: Any, length: int, nitems: int) -> None:
        def strgen(length: int) -> Iterator[str]:
            """Yields unique random fixed-length strings"""
            seen: set[str] = set()
            while True:
                x = rand.str(length)
                if x in seen:
                    continue
                seen.add(x)
                yield x

        unique_str: Callable[[], str] = strgen(length).__next__

        myenum = enum.Enum(  # type: ignore[misc]
            "myenum", [(unique_str(), unique_str()) for _ in range(nitems)]
        )
        dec: rune.msgpack.Decoder[Any] = rune.msgpack.Decoder(myenum)

        for val in myenum:
            msg = rune.msgpack.encode(val.value)
            val2 = dec.decode(msg)
            assert val == val2

        for _ in range(10):
            key = unique_str()
            with pytest.raises(ValidationError):
                dec.decode(rune.msgpack.encode(key))

        # Try bad of different lengths
        for bad_length in [1, 7, 15, 30]:
            assert bad_length != length
            key = rand.str(bad_length)
            with pytest.raises(ValidationError):
                dec.decode(rune.msgpack.encode(key))

    @pytest.mark.parametrize("nitems", [1, 3, 6, 12, 24, 48])
    def test_random_enum_different_lengths(self, rand: Any, nitems: int) -> None:
        def strgen() -> Iterator[str]:
            """Yields unique random strings"""
            seen: set[str] = set()
            while True:
                x = rand.str(1, 32)
                if x in seen:
                    continue
                seen.add(x)
                yield x

        unique_str: Callable[[], str] = strgen().__next__

        myenum = enum.Enum(  # type: ignore[misc]
            "myenum", [(unique_str(), unique_str()) for _ in range(nitems)]
        )
        dec: rune.msgpack.Decoder[Any] = rune.msgpack.Decoder(myenum)

        for val in myenum:
            msg = rune.msgpack.encode(val.value)
            val2 = dec.decode(msg)
            assert val == val2

        for _ in range(10):
            key = unique_str()
            with pytest.raises(ValidationError):
                dec.decode(rune.msgpack.encode(key))

    def test_enum_missing(self, proto: ModuleType) -> None:
        class Ex(enum.Enum):
            A = "a"
            B = "b"

            @classmethod
            def _missing_(cls, val: object) -> "Ex | None":
                if val == "return-A":
                    return cls.A
                elif val == "return-B":
                    return cls.B
                elif val == "error":
                    raise ValueError("oh no!")
                else:
                    return None

        dec = proto.Decoder(Ex)

        def roundtrip(msg: str) -> Any:
            return dec.decode(proto.encode(msg))

        assert roundtrip("a") is Ex.A
        assert roundtrip("return-A") is Ex.A
        assert roundtrip("return-B") is Ex.B
        with pytest.raises(ValidationError, match="Invalid enum value 'error'"):
            roundtrip("error")
        with pytest.raises(ValidationError, match="Invalid enum value 'other'"):
            roundtrip("other")


class TestLiterals:
    def test_empty_errors(self) -> None:
        with pytest.raises(TypeError, match="Literal types must have at least one item"):
            rune.msgpack.Decoder(Literal[()])

    def test_native_literal_generic_alias(self) -> None:
        typ = Literal[1, 2]
        assert type(typ) is typing._LiteralGenericAlias  # type: ignore[attr-defined]
        dec: rune.json.Decoder[Any] = rune.json.Decoder(typ)
        info = typ.__rune_cache__  # type: ignore[attr-defined]
        assert info is not None
        # a second decoder reuses the info cached on the alias itself
        dec2: rune.json.Decoder[Any] = rune.json.Decoder(typ)
        assert typ.__rune_cache__ is info  # type: ignore[attr-defined]
        assert dec.decode(b"2") == 2
        assert dec2.decode(b"1") == 1
        with pytest.raises(ValidationError):
            dec.decode(b"3")

    def test_types_generic_alias_literal(self) -> None:
        typ = types.GenericAlias(typing.Literal, (1, 2))  # type: ignore[arg-type]
        assert type(typ) is types.GenericAlias
        dec: rune.json.Decoder[Any] = rune.json.Decoder(typ)
        assert dec.decode(b"1") == 1
        assert dec.decode(b"2") == 2
        with pytest.raises(ValidationError):
            dec.decode(b"3")
        # building a second decoder must not error either
        assert rune.json.Decoder(typ).decode(b"2") == 2

    @pytest.mark.parametrize(
        "values",
        [
            [0, 1, 2, 2**63],
            [0, 1, 2, -(2**63) - 1],
        ],
    )
    def test_int_literal_values_out_of_range(self, values: list[int]) -> None:
        literal = Literal[tuple(values)]  # type: ignore[valid-type]

        with pytest.raises(NotImplementedError):
            rune.msgpack.Decoder(literal)

    @pytest.mark.parametrize(
        "typ",
        [
            Literal["ok", b"bad"],
            Literal[1, object()],
            Literal[1, 2, list[int]],
            Literal[1, 2, list],
        ],
    )
    def test_invalid_values(self, typ: Any) -> None:
        with pytest.raises(TypeError, match="not supported"):
            rune.msgpack.Decoder(typ)

    def test_decode_literal_int_str_and_none_uncached_and_cached(self) -> None:
        values: tuple[int | str | None, ...] = (45987, "an_unlikely_string", None)
        literal = Literal[values]  # type: ignore[valid-type]
        assert not hasattr(literal, "__rune_cache__")
        uncached = rune.msgpack.Decoder(literal)
        assert hasattr(literal, "__rune_cache__")
        cached = rune.msgpack.Decoder(literal)

        for val in values:
            assert uncached.decode(rune.msgpack.encode(val)) == val
            assert cached.decode(rune.msgpack.encode(val)) == val

    def test_cache_refcounts(self) -> None:
        literal = Literal[1, 2, "three", "four"]
        dec: rune.msgpack.Decoder[Any] = rune.msgpack.Decoder(literal)  # noqa
        cache = literal.__rune_cache__  # type: ignore[attr-defined]
        count = sys.getrefcount(cache)
        dec2: rune.msgpack.Decoder[Any] = rune.msgpack.Decoder(literal)
        assert sys.getrefcount(cache) == count
        del dec2
        gc.collect()
        assert sys.getrefcount(cache) == count

    @pytest.mark.parametrize("val", [None, (), (1,), (1, 2), (1, 2, 3)])
    def test_msgspec_cache_overwritten(self, val: object) -> None:
        literal = Literal["a", "highly", "improbable", "set", "of", "strings"]

        literal.__rune_cache__ = val  # type: ignore[attr-defined]

        with pytest.raises(RuntimeError, match="__rune_cache__"):
            rune.msgpack.Decoder(literal)

    def test_multiple_literals(self) -> None:
        integers = Literal[-1, -2, -3]
        strings = Literal["apple", "banana"]
        both = integers | strings

        dec: rune.msgpack.Decoder[Any] = rune.msgpack.Decoder(both)

        assert not hasattr(both, "__rune_cache__")

        for val in [-1, -2, -3, "apple", "banana"]:
            assert dec.decode(rune.msgpack.encode(val)) == val

        with pytest.raises(ValidationError, match="Invalid enum value 4"):
            dec.decode(rune.msgpack.encode(4))

        with pytest.raises(ValidationError, match="Invalid enum value 'carrot'"):
            dec.decode(rune.msgpack.encode("carrot"))

    def test_nested_literals(self) -> None:
        integers = Literal[-1, -2, -3]
        strings = Literal["apple", "banana"]
        both = Literal[integers, strings]

        dec: rune.msgpack.Decoder[Any] = rune.msgpack.Decoder(both)

        assert hasattr(both, "__rune_cache__")

        for val in [-1, -2, -3, "apple", "banana"]:
            assert dec.decode(rune.msgpack.encode(val)) == val

        with pytest.raises(ValidationError, match="Invalid enum value 4"):
            dec.decode(rune.msgpack.encode(4))

        with pytest.raises(ValidationError, match="Invalid enum value 'carrot'"):
            dec.decode(rune.msgpack.encode("carrot"))

    @pytest.mark.parametrize(
        "typ, good, bad",
        [
            (Literal[True], [True], [False]),
            (Literal[False], [False], [True]),
            (Literal[True, False], [True, False], []),
            (Literal[1, False], [1, False], [True]),
            (Literal[True, "yes", None], [True, "yes", None], [False]),
        ],
    )
    def test_literal_bool(self, typ: Any, good: list[Any], bad: list[Any]) -> None:
        dec: rune.msgpack.Decoder[Any] = rune.msgpack.Decoder(typ)
        for val in good:
            assert dec.decode(rune.msgpack.encode(val)) == val
        for val in bad:
            with pytest.raises(ValidationError):
                dec.decode(rune.msgpack.encode(val))

    def test_literal_bool_error_message(self) -> None:
        dec: rune.msgpack.Decoder[Any] = rune.msgpack.Decoder(Literal[True])
        with pytest.raises(ValidationError, match="Invalid enum value False"):
            dec.decode(rune.msgpack.encode(False))

    def test_mix_bool_and_bool_literal(self) -> None:
        dec: rune.msgpack.Decoder[Any] = rune.msgpack.Decoder(Literal[True] | bool)
        assert dec.decode(rune.msgpack.encode(True)) is True
        assert dec.decode(rune.msgpack.encode(False)) is False

    def test_mix_int_and_int_literal(self) -> None:
        dec: rune.msgpack.Decoder[Any] = rune.msgpack.Decoder(Literal[-1, 1] | int)
        for x in [-1, 1, 10]:
            assert dec.decode(rune.msgpack.encode(x)) == x

    def test_mix_str_and_str_literal(self) -> None:
        dec: rune.msgpack.Decoder[Any] = rune.msgpack.Decoder(Literal["a", "b"] | str)
        for x in ["a", "b", "c"]:
            assert dec.decode(rune.msgpack.encode(x)) == x


class TestCallable:
    @pytest.mark.parametrize(
        "typ",
        [
            pytest.param(typing.Callable[[int], str], id="typing"),
            pytest.param(collections.abc.Callable[[int], str], id="collections.abc"),
        ],
    )
    def test_callable_generic_alias(self, typ: Any) -> None:
        # 'Callable[...]' is a '_CallableGenericAlias'. The 'typing' flavour subclasses
        # 'typing._GenericAlias' while the 'collections.abc' flavour subclasses
        # 'types.GenericAlias' - but neither is an *exact* 'types.GenericAlias', so the
        # conversion leaves both untouched
        dec: rune.json.Decoder[Any] = rune.json.Decoder(typ)
        with pytest.raises(ValidationError, match="Expected `Callable`"):
            dec.decode(b"null")


class Person(Struct):
    first: str
    last: str
    age: int


class PersonDict(typing.TypedDict):
    first: str
    last: str
    age: int


@dataclass
class PersonDataclass:
    first: str
    last: str
    age: int


class PersonArray(Struct, array_like=True):
    first: str
    last: str
    age: int


class PersonTuple(typing.NamedTuple):
    first: str
    last: str
    age: int


class VeggieInt(enum.IntEnum):
    CARROT = 1
    LETTUCE = 2


class VeggieStr(enum.Enum):
    CARROT = "carrot"
    LETTUCE = "banana"


class TestUnionTypeErrors:
    def test_decoder_unsupported_type(self, proto: ModuleType) -> None:
        with pytest.raises(TypeError):
            proto.Decoder(1)

    def test_decoder_validates_struct_definition_unsupported_types(self, proto: ModuleType) -> None:
        """Struct definitions aren't validated until first use"""

        class Test(Struct):
            a: 1  # type: ignore[valid-type]

        with pytest.raises(TypeError):
            proto.Decoder(Test)

    @pytest.mark.parametrize("typ", [int | deque, deque | int])
    def test_err_union_with_custom_type(self, typ: Any, proto: ModuleType) -> None:
        with pytest.raises(TypeError) as rec:
            proto.Decoder(typ)
        assert "custom type" in str(rec.value)
        assert repr(typ) in str(rec.value)

    @pytest.mark.parametrize(
        "typ",
        [
            dict | Person,
            Person | dict,
            PersonDict | dict,
            PersonDataclass | dict,
            Person | PersonDict,
        ],
    )
    def test_err_union_with_multiple_dict_like_types(self, typ: Any, proto: ModuleType) -> None:
        with pytest.raises(TypeError) as rec:
            proto.Decoder(typ)
        assert "more than one dict-like type" in str(rec.value)
        assert repr(typ) in str(rec.value)

    @py315_or_later_only
    def test_err_union_frozendict_dict_like_types(self, proto: ModuleType) -> None:
        for typ in [
            frozendict | Person,
            Person | frozendict,
            PersonDict | frozendict,
            typing.Union[PersonDataclass, frozendict],  # noqa
        ]:
            with pytest.raises(TypeError) as rec:
                proto.Decoder(typ)
            assert "more than one dict-like type" in str(rec.value)
            assert repr(typ) in str(rec.value)

    @py315_or_later_only
    def test_err_union_frozendict_and_dict(self, proto: ModuleType) -> None:
        for typ in [
            frozendict | dict,
            dict | frozendict,
            dict[str, int] | frozendict,
            frozendict[str, int] | dict,
            frozendict[str, int] | dict[str, bool],
        ]:
            with pytest.raises(TypeError) as rec:
                proto.Decoder(typ)
            assert "more than one dict type" in str(rec.value)
            assert repr(typ) in str(rec.value)

    @pytest.mark.parametrize(
        "typ",
        [
            PersonArray | list,
            tuple | PersonArray,
            PersonArray | PersonTuple,
            PersonTuple | frozenset,
            typing.Union[PersonArray, list],  # noqa
        ],
    )
    def test_err_union_with_struct_array_like_and_array(self, typ: Any, proto: ModuleType) -> None:
        with pytest.raises(TypeError) as rec:
            proto.Decoder(typ)
        assert "more than one array-like type" in str(rec.value)
        assert repr(typ) in str(rec.value)

    @pytest.mark.parametrize(
        "typ",
        [
            FruitInt | int,
            FruitInt | Literal[1, 2],
            typing.Union[FruitInt, int],  # noqa
        ],
    )
    def test_err_union_with_multiple_int_like_types(self, typ: Any, proto: ModuleType) -> None:
        with pytest.raises(TypeError) as rec:
            proto.Decoder(typ)
        assert "int-like" in str(rec.value)
        assert repr(typ) in str(rec.value)

    @pytest.mark.parametrize(
        "typ",
        [
            str,
            Literal["one", "two"],
            datetime.datetime,
            datetime.date,
            datetime.time,
            uuid.UUID,
        ],
    )
    def test_err_union_with_multiple_str_like_types(self, typ: Any, proto: ModuleType) -> None:
        union: Any = FruitStr | typ
        with pytest.raises(TypeError) as rec:
            proto.Decoder(union)
        assert "str-like" in str(rec.value)
        assert repr(union) in str(rec.value)

    @pytest.mark.parametrize(
        "typ,kind",
        [
            (FruitInt | VeggieInt, "int enum"),
            (FruitStr | VeggieStr, "str enum"),
            (dict[int, float] | dict, "dict"),
            (list[int] | list[float], "array-like"),
            (list[int] | tuple, "array-like"),
            (set | tuple, "array-like"),
            (tuple[int, ...] | list, "array-like"),
            (tuple[int, float, str] | set, "array-like"),
            (deque | int | Custom, "custom"),
        ],
    )
    def test_err_union_conflicts(self, typ: Any, kind: str, proto: ModuleType) -> None:
        with pytest.raises(TypeError) as rec:
            proto.Decoder(typ)
        assert f"more than one {kind}" in str(rec.value)
        assert repr(typ) in str(rec.value)

    def test_310_union_types(self, proto: ModuleType) -> None:
        dec: Any = proto.Decoder(int | str | None)
        for msg in [1, "abc", None]:
            assert dec.decode(proto.encode(msg)) == msg
        with pytest.raises(ValidationError):
            assert dec.decode(proto.encode(1.5))


class TestStructUnion:
    def test_err_union_struct_mix_array_like(self, proto: ModuleType) -> None:
        class Test1(Struct, tag=True, array_like=True):
            x: int

        class Test2(Struct, tag=True, array_like=False):
            x: int

        typ = Test1 | Test2

        with pytest.raises(TypeError) as rec:
            proto.Decoder(typ)

        assert "not supported" in str(rec.value)
        assert "array_like" in str(rec.value)
        assert repr(typ) in str(rec.value)

    @pytest.mark.parametrize("array_like", [False, True])
    @pytest.mark.parametrize("tag1", [False, True])
    def test_err_union_struct_not_tagged(
        self, array_like: bool, tag1: bool, proto: ModuleType
    ) -> None:
        class Test1(Struct, tag=tag1, array_like=array_like):
            x: int

        class Test2(Struct, array_like=array_like):
            x: int

        typ = Test1 | Test2

        with pytest.raises(TypeError) as rec:
            proto.Decoder(typ)

        assert "not supported" in str(rec.value)
        assert "must be tagged" in str(rec.value)
        assert repr(typ) in str(rec.value)

    @pytest.mark.parametrize("array_like", [False, True])
    def test_err_union_conflict_with_basic_type(self, array_like: bool, proto: ModuleType) -> None:
        class Test1(Struct, tag=True, array_like=array_like):
            x: int

        class Test2(Struct, tag=True, array_like=array_like):
            x: int

        other: type = list if array_like else dict

        typ: Any = Test1 | Test2 | other

        with pytest.raises(TypeError) as rec:
            proto.Decoder(typ)

        assert "not supported" in str(rec.value)
        if array_like:
            assert "more than one array-like type" in str(rec.value)
        else:
            assert "more than one dict-like type" in str(rec.value)
        assert repr(typ) in str(rec.value)

    @pytest.mark.parametrize("array_like", [False, True])
    def test_err_union_struct_different_fields(self, proto: ModuleType, array_like: bool) -> None:
        class Test1(Struct, tag_field="foo", array_like=array_like):
            x: int

        class Test2(Struct, tag_field="bar", array_like=array_like):
            x: int

        typ = Test1 | Test2

        with pytest.raises(TypeError) as rec:
            proto.Decoder(typ)

        assert "not supported" in str(rec.value)
        assert "the same `tag_field`" in str(rec.value)
        assert repr(typ) in str(rec.value)

    @pytest.mark.parametrize("array_like", [False, True])
    def test_err_union_struct_mix_int_str_tags(self, proto: ModuleType, array_like: bool) -> None:
        class Test1(Struct, tag=1, array_like=array_like):
            x: int

        class Test2(Struct, tag="two", array_like=array_like):
            x: int

        typ = Test1 | Test2

        with pytest.raises(TypeError) as rec:
            proto.Decoder(typ)

        assert "not supported" in str(rec.value)
        assert "both `int` and `str` tags" in str(rec.value)
        assert repr(typ) in str(rec.value)

    @pytest.mark.parametrize("array_like", [False, True])
    @pytest.mark.parametrize(
        "tags",
        [
            ("a", "b", "b"),
            ("a", "a", "b"),
            ("a", "b", "a"),
            (1, 2, 2),
            (1, 1, 2),
            (1, 2, 1),
        ],
    )
    def test_err_union_struct_non_unique_tag_values(
        self, proto: ModuleType, array_like: bool, tags: tuple[Any, ...]
    ) -> None:
        class Test1(Struct, tag=tags[0], array_like=array_like):
            x: int

        class Test2(Struct, tag=tags[1], array_like=array_like):
            x: int

        class Test3(Struct, tag=tags[2], array_like=array_like):
            x: int

        typ = Test1 | Test2 | Test3

        with pytest.raises(TypeError) as rec:
            proto.Decoder(typ)

        assert "not supported" in str(rec.value)
        assert "unique `tag`" in str(rec.value)
        assert repr(typ) in str(rec.value)

    @pytest.mark.parametrize(
        "tag1, tag2, unknown",
        [
            ("Test1", "Test2", "Test3"),
            (0, 1, 2),
            (123, -123, 0),
        ],
    )
    def test_decode_struct_union(
        self, proto: ModuleType, tag1: Any, tag2: Any, unknown: Any
    ) -> None:
        class Test1(Struct, tag=tag1):
            a: int
            b: int
            c: int = 0

        class Test2(Struct, tag=tag2):
            x: int
            y: int

        dec: Any = proto.Decoder(Test1 | Test2)
        enc: Any = proto.Encoder()

        # Tag can be in any position
        assert dec.decode(enc.encode({"type": tag1, "a": 1, "b": 2})) == Test1(1, 2)
        assert dec.decode(enc.encode({"a": 1, "type": tag1, "b": 2})) == Test1(1, 2)
        assert dec.decode(enc.encode({"x": 1, "y": 2, "type": tag2})) == Test2(1, 2)

        # Optional fields still work
        assert dec.decode(enc.encode({"type": tag1, "a": 1, "b": 2, "c": 3})) == Test1(1, 2, 3)
        assert dec.decode(enc.encode({"a": 1, "b": 2, "c": 3, "type": tag1})) == Test1(1, 2, 3)

        # Extra fields still ignored
        assert dec.decode(enc.encode({"a": 1, "b": 2, "d": 4, "type": tag1})) == Test1(1, 2)

        # Tag missing
        with pytest.raises(ValidationError) as rec:
            dec.decode(enc.encode({"a": 1, "b": 2}))
        assert "missing required field `type`" in str(rec.value)

        # Tag wrong type
        with pytest.raises(ValidationError) as rec:
            dec.decode(enc.encode({"type": 123.456, "a": 1, "b": 2}))
        assert f"Expected `{type(tag1).__name__}`" in str(rec.value)
        assert "`$.type`" in str(rec.value)

        # Tag unknown
        with pytest.raises(ValidationError) as rec:
            dec.decode(enc.encode({"type": unknown, "a": 1, "b": 2}))
        assert f"Invalid value {unknown!r} - at `$.type`" == str(rec.value)

    @pytest.mark.parametrize(
        "tag1, tag2, tag3, unknown",
        [
            ("Test1", "Test2", "Test3", "Test4"),
            (0, 1, 2, 3),
            (123, -123, 0, -1),
        ],
    )
    def test_decode_struct_array_union(
        self, proto: ModuleType, tag1: Any, tag2: Any, tag3: Any, unknown: Any
    ) -> None:
        class Test1(Struct, tag=tag1, array_like=True):
            a: int
            b: int
            c: int = 0

        class Test2(Struct, tag=tag2, array_like=True):
            x: int
            y: int

        class Test3(Struct, tag=tag3, array_like=True):
            pass

        dec: Any = proto.Decoder(Test1 | Test2 | Test3)
        enc: Any = proto.Encoder()

        # Decoding works
        assert dec.decode(enc.encode([tag1, 1, 2])) == Test1(1, 2)
        assert dec.decode(enc.encode([tag2, 3, 4])) == Test2(3, 4)
        assert dec.decode(enc.encode([tag3])) == Test3()

        # Optional & Extra fields still respected
        assert dec.decode(enc.encode([tag1, 1, 2, 3])) == Test1(1, 2, 3)
        assert dec.decode(enc.encode([tag1, 1, 2, 3, 4])) == Test1(1, 2, 3)

        # Missing required field
        with pytest.raises(ValidationError) as rec:
            dec.decode(enc.encode([tag1, 1]))
        assert "Expected `array` of at least length 3, got 2" in str(rec.value)

        # Type error has correct field index
        with pytest.raises(ValidationError) as rec:
            dec.decode(enc.encode([tag1, 1, "bad", 2]))
        assert "Expected `int`, got `str` - at `$[2]`" == str(rec.value)

        # Tag missing
        with pytest.raises(ValidationError) as rec:
            dec.decode(enc.encode([]))
        assert "Expected `array` of at least length 1, got 0" == str(rec.value)

        # Tag wrong type
        with pytest.raises(ValidationError) as rec:
            dec.decode(enc.encode([123.456, 2, 3, 4]))
        assert f"Expected `{type(tag1).__name__}`" in str(rec.value)
        assert "`$[0]`" in str(rec.value)

        # Tag unknown
        with pytest.raises(ValidationError) as rec:
            dec.decode(enc.encode([unknown, 1, 2, 3]))
        assert f"Invalid value {unknown!r} - at `$[0]`" == str(rec.value)

    @pytest.mark.parametrize("array_like", [False, True])
    def test_decode_struct_union_with_non_struct_types(
        self, array_like: bool, proto: ModuleType
    ) -> None:
        class Test1(Struct, tag=True, array_like=array_like):
            a: int
            b: int

        class Test2(Struct, tag=True, array_like=array_like):
            x: int
            y: int

        dec: Any = proto.Decoder(Test1 | Test2 | int | str | None)
        enc: Any = proto.Encoder()

        msgs: list[Any] = [Test1(1, 2), Test2(3, 4), None, 5, 6]
        for msg in msgs:
            assert dec.decode(enc.encode(msg)) == msg

        with pytest.raises(ValidationError) as rec:
            dec.decode(enc.encode(True))

        typ: str = "array" if array_like else "object"

        assert f"Expected `int | str | {typ} | null`, got `bool`" == str(rec.value)

    @pytest.mark.parametrize("array_like", [False, True])
    def test_struct_union_cached(self, array_like: bool, proto: ModuleType) -> None:
        from rune.rune import _struct_lookup_cache as cache

        cache.clear()

        class Test1(Struct, tag=True, array_like=array_like):
            a: int
            b: int

        class Test2(Struct, tag=True, array_like=array_like):
            x: int
            y: int

        typ1: Any = Test2 | Test1
        typ2: Any = Test1 | Test2
        typ3: Any = Test1 | Test2 | int | None

        for typ in [typ1, typ2, typ3]:
            for msg in [Test1(1, 2), Test2(3, 4)]:
                assert proto.decode(proto.encode(msg), type=typ) == msg

        assert len(cache) == 1
        assert frozenset((Test1, Test2)) in cache

    def test_struct_union_cache_evicted(self, proto: ModuleType) -> None:
        from rune.rune import _struct_lookup_cache as cache

        MAX_CACHE_SIZE: int = 64  # XXX: update if hardcoded value in `_core.c` changes

        cache.clear()

        def call_with_new_types() -> frozenset[type]:
            class Test1(Struct, tag=True):
                a: int

            class Test2(Struct, tag=True):
                x: int

            typ: tuple[type, type] = (Test1, Test2)

            proto.decode(proto.encode(Test1(1)), type=typing.Union[typ])  # noqa: UP007

            return frozenset(typ)

        first: frozenset[type] = call_with_new_types()
        assert first in cache

        # Fill up the cache
        for _ in range(MAX_CACHE_SIZE - 1):
            call_with_new_types()

        # Check that first item is still in cache and is first in order
        assert len(cache) == MAX_CACHE_SIZE
        assert first in cache
        assert first == next(iter(cache.keys()))

        # Add a new item, causing an item to be popped from the cache
        new: frozenset[type] = call_with_new_types()

        assert len(cache) == MAX_CACHE_SIZE
        assert first not in cache
        assert frozenset(new) in cache


T = typing.TypeVar("T")


def make_new_alias(alias: Any) -> Any:
    """Replicate the logic to produce a typing._GenericAlias from a types.GenericAlias"""
    generic_alias = typing._GenericAlias  # type: ignore[attr-defined]
    return generic_alias(alias.__origin__, alias.__origin__.__parameters__)[alias.__args__]


class TestGenericStruct:
    def test_generic_struct_info_cached(self, proto: ModuleType) -> None:
        class Ex(Struct, typing.Generic[T]):
            x: T

        typ: Any = Ex[int]
        assert Ex[int] is typ

        dec = proto.Decoder(typ)
        info = typ.__rune_cache__
        assert info is not None
        assert sys.getrefcount(info) <= 4  # info + attr + decoder + func call
        dec2 = proto.Decoder(typ)
        assert typ.__rune_cache__ is info
        assert sys.getrefcount(info) <= 5

        del dec
        del dec2
        assert sys.getrefcount(info) <= 3

    def test_generic_struct_invalid_types_not_cached(self, proto: ModuleType) -> None:
        class Ex(Struct, typing.Generic[T]):
            x: list[T] | tuple[float]

        candidates: list[Any] = [Ex, Ex[int]]
        for typ in candidates:
            for _ in range(2):
                with pytest.raises(TypeError, match="not supported"):
                    proto.Decoder(typ)

            assert not hasattr(typ, "__rune_cache__")

    def test_rune_cache_overwritten(self, proto: ModuleType) -> None:
        class Ex(Struct, typing.Generic[T]):
            x: T

        typ: Any = Ex[int]
        typ.__rune_cache__ = 1

        with pytest.raises(RuntimeError, match="__rune_cache__"):
            proto.Decoder(typ)

    @pytest.mark.parametrize("array_like", [False, True])
    def test_generic_struct(self, proto: ModuleType, array_like: bool) -> None:
        class Ex(Struct, typing.Generic[T], array_like=array_like):
            x: T
            y: list[T]

        sol = Ex(1, [1, 2])
        msg = proto.encode(sol)

        res = proto.decode(msg, type=Ex)
        assert res == sol

        res = proto.decode(msg, type=Ex[int])
        assert res == sol

        res = proto.decode(msg, type=Ex[int | str])
        assert res == sol

        res = proto.decode(msg, type=Ex[float])
        assert type(res.x) is float

        with pytest.raises(ValidationError, match="Expected `str`, got `int`"):
            proto.decode(msg, type=Ex[str])

    @py312_plus
    def test_generic_with_typevar_syntax(self, proto: ModuleType) -> None:
        source = """
        from rune import Struct
        class Ex[T](Struct):
            x: T
            y: list[T]
        """

        with temp_module(source) as mod:
            sol = mod.Ex(1, [1, 2])
            msg = proto.encode(sol)

            res = proto.decode(msg, type=mod.Ex)
            assert res == sol

            res = proto.decode(msg, type=mod.Ex[int])
            assert res == sol

            res = proto.decode(msg, type=mod.Ex[typing.Union[int, str]])  # noqa
            assert res == sol

            res = proto.decode(msg, type=mod.Ex[int | str])
            assert res == sol

            res = proto.decode(msg, type=mod.Ex[float])
            assert type(res.x) is float

            with pytest.raises(ValidationError, match="Expected `str`, got `int`"):
                proto.decode(msg, type=mod.Ex[str])

    @pytest.mark.parametrize(
        "optional_union",
        [
            pytest.param("Ex[T] | None", id="native-union"),
            pytest.param("Optional[Ex[T]]", id="typing.Optional"),
        ],
    )
    @pytest.mark.parametrize("array_like", [False, True])
    def test_recursive_generic_struct(
        self, proto: ModuleType, array_like: bool, optional_union: str
    ) -> None:
        source = f"""
        from __future__ import annotations
        from typing import Union, Generic, TypeVar, Optional
        from rune import Struct

        T = TypeVar("T")

        class Ex(Struct, Generic[T], array_like={array_like}):
            a: T
            b: {optional_union}
        """

        with temp_module(source) as mod:
            msg = mod.Ex(a=1, b=mod.Ex(a=2, b=None))
            msg2 = mod.Ex(a=1, b=mod.Ex(a="bad", b=None))
            assert proto.decode(proto.encode(msg), type=mod.Ex) == msg
            assert proto.decode(proto.encode(msg2), type=mod.Ex) == msg2
            assert proto.decode(proto.encode(msg), type=mod.Ex[int]) == msg

            with pytest.raises(ValidationError) as rec:
                proto.decode(proto.encode(msg2), type=mod.Ex[int])
            if array_like:
                assert "`$[1][0]`" in str(rec.value)
            else:
                assert "`$.b.a`" in str(rec.value)
            assert "Expected `int`, got `str`" in str(rec.value)

    @pytest.mark.parametrize("array_like", [False, True])
    def test_generic_struct_union(self, proto: ModuleType, array_like: bool) -> None:
        class Test1(Struct, typing.Generic[T], tag=True, array_like=array_like):
            a: T | None
            b: int

        class Test2(Struct, typing.Generic[T], tag=True, array_like=array_like):
            x: T
            y: int

        typ: Any = Test1[T] | Test2[T]

        msg1 = Test1(1, 2)
        s1 = proto.encode(msg1)
        msg2 = Test2("three", 4)
        s2 = proto.encode(msg2)
        msg3 = Test1(None, 4)
        s3 = proto.encode(msg3)

        assert proto.decode(s1, type=typ) == msg1
        assert proto.decode(s2, type=typ) == msg2
        assert proto.decode(s3, type=typ) == msg3

        assert proto.decode(s1, type=typ[int]) == msg1
        assert proto.decode(s3, type=typ[int]) == msg3
        assert proto.decode(s2, type=typ[str]) == msg2
        assert proto.decode(s3, type=typ[str]) == msg3

        with pytest.raises(ValidationError) as rec:
            proto.decode(s1, type=typ[str])
        assert "Expected `str | null`, got `int`" in str(rec.value)
        loc = "$[1]" if array_like else "$.a"
        assert loc in str(rec.value)

        with pytest.raises(ValidationError) as rec:
            proto.decode(s2, type=typ[int])
        assert "Expected `int`, got `str`" in str(rec.value)
        loc = "$[1]" if array_like else "$.x"
        assert loc in str(rec.value)

    def test_unbound_typevars_use_bound_if_set(self, proto: ModuleType) -> None:
        T_bound: typing.TypeVar = typing.TypeVar("T", bound=int | str)

        dec: Any = proto.Decoder(list[T_bound])
        sol: list[int | str] = [1, "two", 3, "four"]
        msg = proto.encode(sol)
        assert dec.decode(msg) == sol

        bad = proto.encode([1, {}])
        with pytest.raises(
            ValidationError,
            match=r"Expected `int \| str`, got `object` - at `\$\[1\]`",
        ):
            dec.decode(bad)

    def test_unbound_typevars_with_constraints_unsupported(self, proto: ModuleType) -> None:
        T_constr: typing.TypeVar = typing.TypeVar("T", int, str)
        with pytest.raises(TypeError) as rec:
            proto.Decoder(list[T_constr])

        assert "Unbound TypeVar `~T` has constraints" in str(rec.value)

    @pytest.mark.parametrize(
        "future",
        [pytest.param(False, id="no future"), pytest.param(True, id="future")],
    )
    @pytest.mark.parametrize("mapping_type", ["collections.abc.Mapping", "typing.Mapping"])
    def test_inherited_builtin_generic(self, mapping_type: str, future: bool) -> None:
        source = f"""
            from rune import Struct, StructMeta
            import collections
            import abc
            import typing

            T = typing.TypeVar("T")

            class CombinedMeta(StructMeta, abc.ABCMeta):
                pass

            class Foo({mapping_type}[str, T], Struct, typing.Generic[T], metaclass=CombinedMeta):
                data: dict[str, T]

                def __getitem__(self, x):
                    return self.data[x]

                def __len__(self):
                    return len(self.data)

                def __iter__(self):
                    return iter(self.data)
            """

        if future:
            source = "from __future__ import annotations\n" + textwrap.dedent(source)

        with temp_module(source) as mod, pytest.raises(ValidationError):
            rune.msgpack.decode(rune.msgpack.encode(mod.Foo({"x": "foo"})), type=mod.Foo[int])

        rune.msgpack.decode(rune.msgpack.encode(mod.Foo({"x": 1})), type=mod.Foo[int])

    @pytest.mark.parametrize("leaf", ["Child1", "Child2", "Child3"])
    @pytest.mark.parametrize(
        "future",
        [pytest.param(False, id="no future"), pytest.param(True, id="future")],
    )
    @pytest.mark.parametrize("mapping_type", ["collections.abc.Mapping", "typing.Mapping"])
    def test_inherited_builtin_generic_multilevel(
        self, mapping_type: str, future: bool, leaf: str
    ) -> None:
        # the builtin generic must be listed *before* 'typing.Generic[T]'. With
        # 'Generic' first its '__class_getitem__' wins, so 'Base[U]' resolves to a
        # 'typing._GenericAlias' instead of a 'types.GenericAlias'
        source = f"""
            from rune import Struct, StructMeta
            import collections
            import abc
            import typing

            T = typing.TypeVar("T")
            U = typing.TypeVar("U")
            V = typing.TypeVar("V")
            W = typing.TypeVar("W")

            class CombinedMeta(StructMeta, abc.ABCMeta):
                pass

            class Base({mapping_type}[str, T], Struct, typing.Generic[T], metaclass=CombinedMeta):
                data: dict[str, T]

                def __getitem__(self, x):
                    return self.data[x]

                def __len__(self):
                    return len(self.data)

                def __iter__(self):
                    return iter(self.data)

            class Child1(Base[U], typing.Generic[U]):
                pass

            class Child2(Child1[V], typing.Generic[V]):
                pass

            class Child3(Child2[W], typing.Generic[W]):
                pass
            """

        if future:
            source = "from __future__ import annotations\n" + textwrap.dedent(source)

        with temp_module(source) as mod:
            typ = getattr(mod, leaf)[int]
            rune.json.decode(b'{"data": {"x": 1}}', type=typ)
            with pytest.raises(ValidationError):
                rune.json.decode(b'{"data": {"x": "not_an_int"}}', type=typ)

    @pytest.mark.parametrize("leaf", ["Child1", "Child2", "Child3"])
    @pytest.mark.parametrize("mapping_type", ["collections.abc.Mapping", "typing.Mapping"])
    @py312_plus
    def test_inherited_builtin_generic_multilevel_typevar_syntax(
        self, mapping_type: str, leaf: str
    ) -> None:
        source = f"""
            from rune import Struct, StructMeta
            import collections
            import abc
            import typing

            class CombinedMeta(StructMeta, abc.ABCMeta):
                pass

            class Base[T]({mapping_type}[str, T], Struct, metaclass=CombinedMeta):
                data: dict[str, T]

                def __getitem__(self, x):
                    return self.data[x]

                def __len__(self):
                    return len(self.data)

                def __iter__(self):
                    return iter(self.data)

            class Child1[U](Base[U]):
                pass

            class Child2[V](Child1[V]):
                pass

            class Child3[W](Child2[W]):
                pass
            """

        with temp_module(source) as mod:
            typ = getattr(mod, leaf)[int]
            rune.json.decode(b'{"data": {"x": 1}}', type=typ)
            with pytest.raises(ValidationError):
                rune.json.decode(b'{"data": {"x": "not_an_int"}}', type=typ)

    @pytest.mark.parametrize(
        "future",
        [pytest.param(False, id="no future"), pytest.param(False, id="future")],
    )
    @pytest.mark.parametrize("mapping_type", ["collections.abc.Mapping", "typing.Mapping"])
    @py312_plus
    def test_inherited_builtin_generic_typevar_syntax(
        self, mapping_type: str, future: bool
    ) -> None:
        source = f"""
            from rune import Struct, StructMeta
            import collections
            import abc
            import typing

            class CombinedMeta(StructMeta, abc.ABCMeta):
                pass

            class Foo[T]({mapping_type}[str, T], Struct, metaclass=CombinedMeta):
                data: dict[str, T]

                def __getitem__(self, x):
                    return self.data[x]

                def __len__(self):
                    return len(self.data)

                def __iter__(self):
                    return iter(self.data)
            """

        if future:
            source = "from __future__ import annotations\n" + source

        with temp_module(source) as mod, pytest.raises(ValidationError):
            rune.msgpack.decode(rune.msgpack.encode(mod.Foo({"x": "foo"})), type=mod.Foo[int])

        rune.msgpack.decode(rune.msgpack.encode(mod.Foo({"x": 1})), type=mod.Foo[int])

    @pytest.mark.parametrize("mapping_type", ["collections.abc.Mapping", "typing.Mapping"])
    @py312_plus
    def test_inherited_builtin_generic_typevar_syntax_info_cached(self, mapping_type: str) -> None:
        source = f"""
            from rune import Struct, StructMeta
            import collections
            import abc
            import typing

            class CombinedMeta(StructMeta, abc.ABCMeta):
                pass

            class Foo[T]({mapping_type}[str, T], Struct, metaclass=CombinedMeta):
                data: dict[str, T]

                def __getitem__(self, x):
                    return self.data[x]

                def __len__(self):
                    return len(self.data)

                def __iter__(self):
                    return iter(self.data)
            """

        with temp_module(source) as mod:
            typ = mod.Foo[int]
            dec = rune.json.Decoder(typ)
            info = make_new_alias(typ).__rune_cache__
            gc.collect()
            assert info is not None
            assert sys.getrefcount(info) <= 4  # info + attr + decoder + func call
            dec2 = rune.json.Decoder(typ)
            gc.collect()
            assert make_new_alias(typ).__rune_cache__ is info
            assert sys.getrefcount(info) <= 5

            del dec
            del dec2
            gc.collect()
            assert sys.getrefcount(info) <= 3

    @pytest.mark.parametrize(
        "future",
        [pytest.param(False, id="no future"), pytest.param(True, id="future")],
    )
    @pytest.mark.parametrize("mapping_type", ["collections.abc.Mapping", "typing.Mapping"])
    def test_inherited_builtin_generic_multiple_typevars(
        self, mapping_type: str, future: bool
    ) -> None:
        # struct inheriting a builtin generic with *two* type vars. 'Pair[int, str]'
        # is a 'types.GenericAlias' carrying two args; the conversion must subscript
        # the rebuilt alias with the args *tuple* (not spread the args), otherwise
        # generics with more than one type var raise a 'TypeError'
        source = f"""
            from rune import Struct, StructMeta
            import collections
            import abc
            import typing

            K = typing.TypeVar("K")
            V = typing.TypeVar("V")

            class CombinedMeta(StructMeta, abc.ABCMeta):
                pass

            class Pair({mapping_type}[K, V], Struct, typing.Generic[K, V], metaclass=CombinedMeta):
                key: K
                val: V

                def __getitem__(self, x):
                    return getattr(self, x)

                def __len__(self):
                    return 2

                def __iter__(self):
                    return iter(("key", "val"))
            """

        if future:
            source = "from __future__ import annotations\n" + textwrap.dedent(source)

        with temp_module(source) as mod:
            typ = mod.Pair[int, str]
            assert type(typ) is types.GenericAlias
            assert rune.json.decode(b'{"key": 1, "val": "a"}', type=typ) == mod.Pair(1, "a")
            with pytest.raises(ValidationError, match="Expected `int`"):
                rune.json.decode(b'{"key": "x", "val": "a"}', type=typ)
            with pytest.raises(ValidationError, match="Expected `str`"):
                rune.json.decode(b'{"key": 1, "val": 2}', type=typ)

            # the 'types.GenericAlias' must also be normalised when nested inside a
            # union (collected via recursion) or a container (collected as its own
            # node), not just at the top level
            msg = b'{"key": 1, "val": "a"}'
            assert rune.json.decode(msg, type=typing.Optional[typ]) == mod.Pair(1, "a")  # noqa
            assert rune.json.decode(b"[" + msg + b"]", type=list[typ]) == [mod.Pair(1, "a")]

    def test_manual_types_generic_alias_robustness(self) -> None:
        # 'types.GenericAlias' instances can be built manually around arbitrary
        # origins (e.g. 'types.GenericAlias(...)' or a C-level 'Py_GenericAlias'),
        # not just via ordinary subscription. Feeding such aliases to rune used to
        # segfault when the origin wasn't a regularly-parametrised generic; ensure
        # they now either decode or raise a clean Python error.
        class GenericStruct(Struct, typing.Generic[T]):
            x: T

        class PlainStruct(Struct):
            x: int

        # well-formed manual alias behaves exactly like the native subscription
        alias: Any = types.GenericAlias(GenericStruct, (int,))
        assert rune.json.decode(b'{"x": 1}', type=alias) == GenericStruct(1)
        with pytest.raises(ValidationError, match="Expected `int`"):
            rune.json.decode(b'{"x": "bad"}', type=alias)

        # parametrising a non-generic type is meaningless
        with pytest.raises(TypeError, match="not a generic type"):
            rune.json.Decoder(types.GenericAlias(PlainStruct, (int,)))

        # too many
        with pytest.raises(TypeError):
            rune.json.Decoder(types.GenericAlias(GenericStruct, (int, str)))


class TestNamedTuple:
    def test_types_generic_alias_non_generic_errors(self):
        # Parametrizing a non-generic NamedTuple via a manually-built
        # 'types.GenericAlias' is meaningless and must raise, not silently ignore the
        # args.
        class Ex(typing.NamedTuple):
            a: int

        with pytest.raises(TypeError, match="not a generic type"):
            rune.json.Decoder(types.GenericAlias(Ex, (int,)))

    def test_type_cached(self, proto):
        class Ex(typing.NamedTuple):
            a: int
            b: str

        msg = (1, "two")

        dec = proto.Decoder(Ex)
        info = Ex.__rune_cache__
        assert info is not None
        dec2 = proto.Decoder(Ex)
        assert Ex.__rune_cache__ is info

        assert dec.decode(proto.encode(msg)) == msg
        assert dec2.decode(proto.encode(msg)) == msg

    def test_msgspec_cache_overwritten(self, proto):
        class Ex(typing.NamedTuple):
            x: int

        Ex.__rune_cache__ = 1

        with pytest.raises(RuntimeError, match="__rune_cache__"):
            proto.Decoder(Ex)

    def test_multiple_namedtuple_errors(self, proto):
        class Ex1(typing.NamedTuple):
            a: int

        class Ex2(typing.NamedTuple):
            b: int

        with pytest.raises(TypeError, match="may not contain more than one NamedTuple"):
            proto.Decoder(Ex1 | Ex2)
        with pytest.raises(TypeError, match="may not contain more than one NamedTuple"):
            proto.Decoder(typing.Union[Ex1, Ex2])  # noqa

    def test_subtype_error(self, proto):
        class Ex(typing.NamedTuple):
            a: int
            b: list | tuple

        with pytest.raises(TypeError, match="may not contain more than one array-like"):
            proto.Decoder(Ex)
        assert not hasattr(Ex, "__rune_cache__")

    def test_recursive_type(self, proto):
        source = """
        from __future__ import annotations
        from typing import NamedTuple, Union

        class Ex(NamedTuple):
            a: int
            b: Ex | None
        """

        with temp_module(source) as mod:
            msg = mod.Ex(1, mod.Ex(2, None))
            dec = proto.Decoder(mod.Ex)
            assert dec.decode(proto.encode(msg)) == msg

            with pytest.raises(ValidationError) as rec:
                dec.decode(proto.encode(mod.Ex(1, ("bad", "two"))))
            assert "`$[1][0]`" in str(rec.value)
            assert "Expected `int`, got `str`" in str(rec.value)

    @pytest.mark.parametrize("use_typing", [True, False])
    def test_decode_namedtuple_no_defaults(self, proto, use_typing):
        if use_typing:

            class Example(typing.NamedTuple):
                a: int
                b: int
                c: int

        else:
            Example = collections.namedtuple("Example", "a b c")

        dec = proto.Decoder(Example)
        msg = Example(1, 2, 3)
        res = dec.decode(proto.encode(msg))
        assert res == msg

        suffix = ", got 1" if proto is rune.msgpack else ""
        with pytest.raises(ValidationError, match=f"length 3{suffix}"):
            dec.decode(proto.encode((1,)))

        suffix = ", got 6" if proto is rune.msgpack else ""
        with pytest.raises(ValidationError, match=f"length 3{suffix}"):
            dec.decode(proto.encode((1, 2, 3, 4, 5, 6)))

    @pytest.mark.parametrize("use_typing", [True, False])
    def test_decode_namedtuple_with_defaults(self, proto, use_typing):
        if use_typing:

            class Example(typing.NamedTuple):
                a: int
                b: int
                c: int = -3
                d: int = -4
                e: int = -5

        else:
            Example = collections.namedtuple("Example", "a b c d e", defaults=(-3, -4, -5))

        dec = proto.Decoder(Example)
        for args in [(1, 2), (1, 2, 3), (1, 2, 3, 4), (1, 2, 3, 4, 5)]:
            msg = Example(*args)
            res = dec.decode(proto.encode(msg))
            assert res == msg

        suffix = ", got 1" if proto is rune.msgpack else ""
        with pytest.raises(ValidationError, match=f"length 2 to 5{suffix}"):
            dec.decode(proto.encode((1,)))

        suffix = ", got 6" if proto is rune.msgpack else ""
        with pytest.raises(ValidationError, match=f"length 2 to 5{suffix}"):
            dec.decode(proto.encode((1, 2, 3, 4, 5, 6)))

    def test_decode_namedtuple_field_wrong_type(self, proto):
        dec = proto.Decoder(PersonTuple)
        msg = proto.encode((1, "bad", 2))
        with pytest.raises(ValidationError, match=r"Expected `str`, got `int` - at `\$\[0\]`"):
            dec.decode(msg)

    def test_decode_namedtuple_not_array(self, proto):
        dec = proto.Decoder(PersonTuple)
        msg = proto.encode({})
        with pytest.raises(ValidationError, match="Expected `array`, got `object`"):
            dec.decode(msg)

    def test_generic_namedtuple_info_cached(self, proto):
        NamedTuple = pytest.importorskip("typing_extensions").NamedTuple

        class Ex(NamedTuple, typing.Generic[T]):
            x: T

        typ = Ex[int]
        assert Ex[int] is typ

        dec = proto.Decoder(typ)
        info = typ.__rune_cache__
        assert info is not None
        assert sys.getrefcount(info) <= 4  # info + attr + decoder + func call
        dec2 = proto.Decoder(typ)
        assert typ.__rune_cache__ is info
        assert sys.getrefcount(info) <= 5

        del dec
        del dec2
        assert sys.getrefcount(info) <= 3

    def test_generic_namedtuple_invalid_types_not_cached(self, proto):
        NamedTuple = pytest.importorskip("typing_extensions").NamedTuple

        class Ex(NamedTuple, typing.Generic[T]):
            x: list[T] | tuple[float]

        for typ in [Ex, Ex[int]]:
            for _ in range(2):
                with pytest.raises(TypeError, match="not supported"):
                    proto.Decoder(typ)

            assert not hasattr(typ, "__rune_cache__")

    def test_generic_namedtuple(self, proto):
        NamedTuple = pytest.importorskip("typing_extensions").NamedTuple

        class Ex(NamedTuple, typing.Generic[T]):
            x: T
            y: list[T]

        sol = Ex(1, [1, 2])
        msg = proto.encode(sol)

        res = proto.decode(msg, type=Ex)
        assert res == sol

        res = proto.decode(msg, type=Ex[int])
        assert res == sol

        res = proto.decode(msg, type=Ex[int | str])
        assert res == sol

        res = proto.decode(msg, type=Ex[float])
        assert type(res.x) is float

        with pytest.raises(ValidationError, match="Expected `str`, got `int`"):
            proto.decode(msg, type=Ex[str])

    def test_recursive_generic_namedtuple(self, proto):
        pytest.importorskip("typing_extensions")

        source = """
        from __future__ import annotations
        from typing import Union, Generic, TypeVar
        from typing_extensions import NamedTuple

        T = TypeVar("T")

        class Ex(NamedTuple, Generic[T]):
            a: T
            b: Ex[T] | None
        """

        with temp_module(source) as mod:
            msg = mod.Ex(a=1, b=mod.Ex(a=2, b=None))
            msg2 = mod.Ex(a=1, b=mod.Ex(a="bad", b=None))
            assert proto.decode(proto.encode(msg), type=mod.Ex) == msg
            assert proto.decode(proto.encode(msg2), type=mod.Ex) == msg2
            assert proto.decode(proto.encode(msg), type=mod.Ex[int]) == msg

            with pytest.raises(ValidationError) as rec:
                proto.decode(proto.encode(msg2), type=mod.Ex[int])
            assert "`$[1][0]`" in str(rec.value)
            assert "Expected `int`, got `str`" in str(rec.value)

    def test_namedtuple_hash_preserved_after_roundtrip(self, proto):
        """Hash of deserialized NamedTuple matches hash of original.

        Regression test for gh-967. In Python 3.14+, tuples cache their hash
        in ob_hash. When msgspec allocates NamedTuples via tp_alloc (which
        zero-initializes memory), ob_hash was left at 0 — a valid cached hash
        — so hash() returned 0 for all deserialized NamedTuples."""

        class Key(typing.NamedTuple):
            value: int

        original = Key(value=42)
        encoded = proto.encode(original)
        decoded = proto.decode(encoded, type=Key)

        assert original == decoded
        assert hash(original) == hash(decoded)

        # Verify dict lookup works (requires matching hash)
        d = {original: "found"}
        assert decoded in d
        assert d[decoded] == "found"


class TestTypeAlias:
    @py312_plus
    def test_simple(self, proto):
        with temp_module("type Ex = str | None") as mod:
            dec = proto.Decoder(mod.Ex)
            assert dec.decode(proto.encode("test")) == "test"
            assert dec.decode(proto.encode(None)) is None
            with pytest.raises(ValidationError):
                dec.decode(proto.encode(1))

    @py312_plus
    def test_generic(self, proto):
        with temp_module("type Pair[T] = tuple[T, T]") as mod:
            dec = proto.Decoder(mod.Pair)
            assert dec.decode(proto.encode((1, 2))) == (1, 2)
            for bad in [1, [1, 2, 3]]:
                with pytest.raises(ValidationError):
                    dec.decode(proto.encode(bad))

    @py312_plus
    def test_parametrized_generic(self, proto):
        with temp_module("type Pair[T] = tuple[T, T]") as mod:
            dec = proto.Decoder(mod.Pair[int])
            assert dec.decode(proto.encode((1, 2))) == (1, 2)
            for bad in [1, [1, 2, 3], [1, "a"]]:
                with pytest.raises(ValidationError):
                    dec.decode(proto.encode(bad))

    @py312_plus
    def test_typealias_wrapping_typealias(self, proto):
        src = """
        type Pair[T] = tuple[T, T]
        type Pairs[T] = list[Pair[T]]
        """
        with temp_module(src) as mod:
            dec = proto.Decoder(mod.Pairs)
            for good in [[], [(1, 2), (3, 4)]]:
                assert dec.decode(proto.encode(good)) == good
            for bad in [1, [1], [(1, 2, 3)]]:
                with pytest.raises(ValidationError):
                    dec.decode(proto.encode(bad))

            dec = proto.Decoder(mod.Pairs[int])
            for good in [[], [(1, 2)], [(1, 2), (3, 4)]]:
                assert dec.decode(proto.encode(good)) == good
            for bad in [1, [1], [(1, "a")]]:
                with pytest.raises(ValidationError):
                    dec.decode(proto.encode(bad))

    @py312_plus
    def test_typealias_with_constraints(self, proto):
        src = """
        import rune
        from typing import Annotated
        type Key = Annotated[str, rune.Meta(max_length=4)]
        """
        with temp_module(src) as mod:
            dec = proto.Decoder(mod.Key)
            for good in ["", "abc", "abcd"]:
                assert dec.decode(proto.encode(good)) == good
            for bad in [1, "abcde"]:
                with pytest.raises(ValidationError):
                    dec.decode(proto.encode(bad))

    @py312_plus
    def test_typealias_parametrized_generic_too_many_parameters(self):
        with temp_module("type Pair[T] = tuple[T, T]") as mod:
            with pytest.raises(TypeError):
                rune.json.Decoder(mod.Pair[int, int])

    @py312_plus
    @pytest.mark.parametrize(
        "src",
        [
            "type Ex = Ex | None",
            "type Ex = tuple[Ex, int]",
            "type Ex[T] = tuple[T, Ex[T]]",
            "type Temp[T] = tuple[T, Temp[T]]; Ex = Temp[int]",
            "type Temp[T] = tuple[T, Ex[T]]; type Ex[T] = tuple[Temp[T], T];",
        ],
    )
    @emscripten_stack_limited
    @win_arm64_py312_stack_limited
    def test_recursive_typealias_errors(self, src):
        """Eventually we should support this, but for now just test that it
        errors cleanly"""
        with temp_module(src) as mod:
            with pytest.raises(RecursionError):
                rune.json.Decoder(mod.Ex)

    @py312_plus
    def test_typealias_invalid_type(self):
        with temp_module("type Ex = int | complex") as mod:
            with pytest.raises(TypeError):
                rune.json.Decoder(mod.Ex)


class TestStructPostInit:
    @pytest.mark.parametrize("array_like", [False, True])
    @pytest.mark.parametrize("union", [False, True])
    def test_struct_post_init(self, array_like, union, proto):
        count = 0
        singleton = object()

        class Ex(Struct, array_like=array_like, tag=union):
            x: int

            def __post_init__(self):
                nonlocal count
                count += 1
                return singleton

        if union:

            class Ex2(Struct, array_like=array_like, tag=True):
                pass

            typ = Ex | Ex2
        else:
            typ = Ex

        msg = Ex(1)
        buf = proto.encode(msg)
        res = proto.decode(buf, type=typ)
        assert res == msg
        assert count == 2  # 1 for Ex(), 1 for decode
        assert sys.getrefcount(singleton) <= 2  # 1 for ref, 1 for call

    @pytest.mark.parametrize("array_like", [False, True])
    @pytest.mark.parametrize("union", [False, True])
    @pytest.mark.parametrize("exc_class", [ValueError, TypeError, OSError])
    def test_struct_post_init_errors(self, array_like, union, exc_class, proto):
        error = False

        class Ex(Struct, array_like=array_like, tag=union):
            x: int

            def __post_init__(self):
                if error:
                    raise exc_class("Oh no!")

        if union:

            class Ex2(Struct, array_like=array_like, tag=True):
                pass

            typ = Ex | Ex2
        else:
            typ = Ex

        msg = proto.encode([Ex(1)])
        error = True

        if exc_class in (ValueError, TypeError):
            expected = ValidationError
        else:
            expected = exc_class

        with pytest.raises(expected, match="Oh no!") as rec:
            proto.decode(msg, type=list[typ])

        if expected is ValidationError:
            assert "- at `$[0]`" in str(rec.value)


class TestDecimal:
    def test_encoder_decimal_format(self, proto):
        assert proto.Encoder().decimal_format == "string"
        assert proto.Encoder(decimal_format="string").decimal_format == "string"
        assert proto.Encoder(decimal_format="number").decimal_format == "number"

        def fn(x):
            return str(x)

        enc = proto.Encoder(decimal_format=fn)
        assert enc.decimal_format is fn

    @pytest.mark.parametrize("decimal_format", ["bad", 1, {}])
    def test_encoder_invalid_decimal_format(self, proto, decimal_format):
        with pytest.raises(
            ValueError,
            match=f"must be 'string', 'number', or a callable, got {decimal_format!r}",
        ):
            proto.Encoder(decimal_format=decimal_format)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (decimal.Decimal("1.3449"), "1.345"),
            ([decimal.Decimal("1.3449")], ["1.345"]),
            (
                [decimal.Decimal("1.3449"), [decimal.Decimal("1.3449")]],
                ["1.345", ["1.345"]],
            ),
            (
                (decimal.Decimal("1.3449"), decimal.Decimal("1.3449")),
                ("1.345", "1.345"),
            ),
            (
                {"key": decimal.Decimal("1.3449"), decimal.Decimal("1.3449"): "value"},
                {"key": "1.345", "1.345": "value"},
            ),
            (
                {
                    "key": {
                        "sub_key": [
                            decimal.Decimal("1.3449"),
                            decimal.Decimal("1.3449"),
                        ]
                    }
                },
                {"key": {"sub_key": ["1.345", "1.345"]}},
            ),
        ],
    )
    def test_encoder_decimal_callable_convert_value_to_string(self, proto, value, expected):
        enc = proto.Encoder(decimal_format=lambda d: str(d.quantize(decimal.Decimal("0.000"))))
        msg = enc.encode(value)
        assert msg == enc.encode(expected)

    @pytest.mark.parametrize(
        ("fn", "expected"),
        [
            (lambda x: [str(x), str(x)], ["1.21", "1.21"]),
            (lambda x: {"key": str(x)}, {"key": "1.21"}),
            (lambda x: {str(x): "value"}, {"1.21": "value"}),
            (lambda x: {str(x)}, {"1.21"}),
            (lambda x: (str(x),), ("1.21",)),
            (lambda x: float(x.quantize(decimal.Decimal("0.0"))), 1.2),
            (lambda x: int(x.quantize(decimal.Decimal("0.0")) * 100), 120),
        ],
    )
    def test_encoder_decimal_callable_convert_value_into_different_types(self, proto, fn, expected):
        enc = proto.Encoder(decimal_format=lambda d: fn(d))
        msg = enc.encode(decimal.Decimal("1.21"))
        assert msg == enc.encode(expected)

    @pytest.mark.parametrize(
        "fn",
        [
            lambda x: [x],
            lambda x: {"key": x},
            lambda x: {x: "value"},
            lambda x: {x},
            lambda x: (x,),
            lambda x: x.quantize(decimal.Decimal("0.0")),
        ],
    )
    def test_encoder_decimal_callable_returned_a_value_containing_a_decimal(
        self,
        proto,
        fn: Callable[[decimal.Decimal], Any],
    ):
        enc = proto.Encoder(decimal_format=fn)
        with pytest.raises(TypeError, match="callable returned a value containing a Decimal"):
            enc.encode(decimal.Decimal("1"))

    def test_encoder_decimal_callable_raise_error(self, proto):
        def bad_fn(x):
            raise NotImplementedError

        enc = proto.Encoder(decimal_format=bad_fn)
        with pytest.raises(NotImplementedError):
            enc.encode(decimal.Decimal("1.5"))

    def test_encoder_encode_decimal(self, proto):
        enc = proto.Encoder()
        d = decimal.Decimal("1.5")
        s = str(d)
        assert enc.encode(d) == enc.encode(s)

    def test_Encoder_encode_decimal_string(self, proto):
        enc = proto.Encoder(decimal_format="string")
        d = decimal.Decimal("1.5")
        sol = enc.encode(str(d))

        assert enc.encode(d) == sol

        buf = bytearray()
        enc.encode_into(d, buf)
        assert buf == sol

    def test_Encoder_encode_decimal_number(self, proto):
        enc = proto.Encoder(decimal_format="number")
        d = decimal.Decimal("1.5")
        sol = enc.encode(float(d))

        assert enc.encode(d) == sol

        buf = bytearray()
        enc.encode_into(d, buf)
        assert buf == sol

    def test_encode_decimal(self, proto):
        d = decimal.Decimal("1.5")
        s = str(d)
        assert proto.encode(d) == proto.encode(s)

    @pytest.mark.parametrize("val", ["1.5", "InF", "-iNf", "iNfInItY", "-InFiNiTy", "NaN"])
    def test_decode_decimal_str(self, val, proto):
        sol = decimal.Decimal(val)
        msg = proto.encode(sol)
        res = proto.decode(msg, type=decimal.Decimal)
        assert str(res) == str(sol)
        assert type(res) is decimal.Decimal

    def test_decode_decimal_str_invalid(self, proto):
        msg = proto.encode("1..5")
        with pytest.raises(ValidationError, match="Invalid decimal string"):
            proto.decode(msg, type=decimal.Decimal)

    @pytest.mark.parametrize("val", [-1, -1234, 1, 1234])
    def test_decode_decimal_int(self, val, proto):
        msg = proto.encode(val)
        sol = decimal.Decimal(str(val))
        res = proto.decode(msg, type=decimal.Decimal)
        assert type(res) is decimal.Decimal
        assert res == sol

    @pytest.mark.parametrize("val", [0.0, 1.3, float("nan"), float("inf"), float("-inf")])
    def test_decode_decimal_float(self, val, proto):
        msg = proto.encode(val)
        if msg == b"null":
            pytest.skip("nonfinite values not supported")
        sol = decimal.Decimal(str(val))
        res = proto.decode(msg, type=decimal.Decimal)
        assert str(res) == str(sol)
        assert type(res) is decimal.Decimal


class TestAbstractTypes:
    @pytest.mark.parametrize(
        "typ",
        [
            typing.Collection,
            typing.MutableSequence,
            typing.Sequence,
            collections.abc.Collection,
            collections.abc.MutableSequence,
            collections.abc.Sequence,
            typing.MutableSet,
            typing.AbstractSet,
            collections.abc.MutableSet,
            collections.abc.Set,
        ],
    )
    def test_abstract_sequence(self, proto, typ):
        # Hacky, but it works
        if "Set" in str(typ):
            sol = {1, 2}
        else:
            sol = [1, 2]
        msg = proto.encode(sol)
        assert proto.decode(msg, type=typ) == sol
        with pytest.raises(ValidationError, match="Expected `array`, got `str`"):
            proto.decode(proto.encode("a"), type=typ)

        assert proto.decode(msg, type=typ[int]) == sol
        with pytest.raises(ValidationError, match="Expected `int`, got `str`"):
            proto.decode(proto.encode(["a"]), type=typ[int])

    @pytest.mark.parametrize(
        "typ",
        [
            typing.MutableMapping,
            typing.Mapping,
            collections.abc.MutableMapping,
            collections.abc.Mapping,
        ],
    )
    def test_abstract_mapping(self, proto, typ):
        sol = {"x": 1, "y": 2}
        msg = proto.encode(sol)
        assert proto.decode(msg, type=typ) == sol
        with pytest.raises(ValidationError, match="Expected `object`, got `str`"):
            proto.decode(proto.encode("a"), type=typ)

        assert proto.decode(msg, type=typ[str, int]) == sol
        with pytest.raises(ValidationError, match="Expected `int`, got `str`"):
            proto.decode(proto.encode({"a": "b"}), type=typ[str, int])

    @py315_or_later_only
    @pytest.mark.parametrize(
        "typ",
        [
            typing.MutableMapping,
            typing.Mapping,
            collections.abc.MutableMapping,
            collections.abc.Mapping,
        ],
    )
    def test_abstract_mapping_frozendict(self, proto, typ):
        sol = frozendict({"x": 1, "y": 2})
        msg = proto.encode(sol)
        res = proto.decode(msg, type=typ)
        assert res == sol
        assert type(res) is dict  # does not roundtrip
        with pytest.raises(ValidationError, match="Expected `object`, got `str`"):
            proto.decode(proto.encode("a"), type=typ)

        assert proto.decode(msg, type=typ[str, int]) == sol
        with pytest.raises(ValidationError, match="Expected `int`, got `str`"):
            proto.decode(proto.encode(frozendict({"a": "b"})), type=typ[str, int])


class TestStructOmitDefaults:
    def test_omit_defaults(self, proto):
        class Test(Struct, omit_defaults=True):
            a: int = 0
            b: bool = False
            c: str | None = None
            d: list = []  # noqa
            e: list | set = set()  # noqa
            f: dict = {}  # noqa

        cases = [
            (Test(), {}),
            (Test(1), {"a": 1}),
            (Test(1, False), {"a": 1}),
            (Test(1, True), {"a": 1, "b": True}),
            (Test(1, c=None), {"a": 1}),
            (Test(1, c="test"), {"a": 1, "c": "test"}),
            (Test(1, d=[1]), {"a": 1, "d": [1]}),
            (Test(1, e={1}), {"a": 1, "e": [1]}),
            (Test(1, e=[]), {"a": 1, "e": []}),
            (Test(1, f={"a": 1}), {"a": 1, "f": {"a": 1}}),
        ]

        for obj, sol in cases:
            res = proto.decode(proto.encode(obj))
            assert res == sol

    @pytest.mark.parametrize("typ", [tuple, list, set, frozenset, dict])
    def test_omit_defaults_collections(self, proto, typ):
        """Check that using empty collections as default values are detected
        regardless if they're specified by value or as a default_factory."""

        class Test(Struct, omit_defaults=True):
            a: typ = rune.field(default_factory=typ)
            b: typ = rune.field(default=typ())
            c: typ = typ()

        ex = {"x": 1} if typ is dict else [1]

        assert proto.encode(Test()) == proto.encode({})
        for n in ["a", "b", "c"]:
            assert proto.encode(Test(**{n: typ(ex)})) == proto.encode({n: ex})

    def test_omit_defaults_positional(self, proto):
        class Test(Struct, omit_defaults=True):
            a: int
            b: bool = False

        cases = [
            (Test(1), {"a": 1}),
            (Test(1, False), {"a": 1}),
            (Test(1, True), {"a": 1, "b": True}),
        ]

        for obj, sol in cases:
            res = proto.decode(proto.encode(obj))
            assert res == sol

    def test_omit_defaults_tagged(self, proto):
        class Test(Struct, omit_defaults=True, tag=True):
            a: int
            b: bool = False

        cases = [
            (Test(1), {"type": "Test", "a": 1}),
            (Test(1, False), {"type": "Test", "a": 1}),
            (Test(1, True), {"type": "Test", "a": 1, "b": True}),
        ]

        for obj, sol in cases:
            res = proto.decode(proto.encode(obj))
            assert res == sol

    def test_omit_defaults_ignored_for_array_like(self, proto):
        class Test(Struct, omit_defaults=True, array_like=True):
            a: int
            b: bool = False

        cases = [
            (Test(1), [1, False]),
            (Test(1, False), [1, False]),
            (Test(1, True), [1, True]),
        ]

        for obj, sol in cases:
            res = proto.decode(proto.encode(obj))
            assert res == sol


class TestStructForbidUnknownFields:
    def test_forbid_unknown_fields(self, proto):
        class Test(Struct, forbid_unknown_fields=True):
            x: int
            y: int

        good = Test(1, 2)
        assert proto.decode(proto.encode(good), type=Test) == good

        bad = proto.encode({"x": 1, "y": 2, "z": 3})
        with pytest.raises(ValidationError, match="Object contains unknown field `z`"):
            proto.decode(bad, type=Test)

    def test_forbid_unknown_fields_array_like(self, proto):
        class Test(Struct, forbid_unknown_fields=True, array_like=True):
            x: int
            y: int

        good = Test(1, 2)
        assert proto.decode(proto.encode(good), type=Test) == good

        bad = proto.encode([1, 2, 3])
        with pytest.raises(ValidationError, match="Expected `array` of at most length 2"):
            proto.decode(bad, type=Test)


class PointUpper(Struct, rename="upper"):
    x: int
    y: int


class TestStructRename:
    def test_rename_encode_struct(self, proto):
        res = proto.encode(PointUpper(1, 2))
        exp = proto.encode({"X": 1, "Y": 2})
        assert res == exp

    def test_rename_decode_struct(self, proto):
        msg = proto.encode({"X": 1, "Y": 2})
        res = proto.decode(msg, type=PointUpper)
        assert res == PointUpper(1, 2)

    def test_rename_decode_struct_wrong_type(self, proto):
        msg = proto.encode({"X": 1, "Y": "bad"})
        with pytest.raises(ValidationError) as rec:
            proto.decode(msg, type=PointUpper)
        assert "Expected `int`, got `str` - at `$.Y`" == str(rec.value)

    def test_rename_decode_struct_missing_field(self, proto):
        msg = proto.encode({"X": 1})
        with pytest.raises(ValidationError) as rec:
            proto.decode(msg, type=PointUpper)
        assert "Object missing required field `Y`" == str(rec.value)


class TestStructKeywordOnly:
    def test_keyword_only_object(self, proto):
        class Test(Struct, kw_only=True):
            a: int
            b: int = 2
            c: int
            d: int = 4

        sol = Test(a=1, b=2, c=3, d=4)
        msg = proto.encode({"a": 1, "b": 2, "c": 3, "d": 4})
        res = proto.decode(msg, type=Test)
        assert res == sol

        msg = proto.encode({"a": 1, "c": 3})
        res = proto.decode(msg, type=Test)
        assert res == sol

        sol = Test(a=1, b=3, c=5)
        msg = proto.encode({"a": 1, "b": 3, "c": 5})
        res = proto.decode(msg, type=Test)
        assert res == sol

        msg = proto.encode({"a": 1, "b": 2})
        with pytest.raises(
            ValidationError,
            match="missing required field `c`",
        ):
            proto.decode(msg, type=Test)

        msg = proto.encode({"c": 1, "b": 2})
        with pytest.raises(
            ValidationError,
            match="missing required field `a`",
        ):
            proto.decode(msg, type=Test)

    def test_keyword_only_array(self, proto):
        class Test(Struct, kw_only=True, array_like=True):
            a: int
            b: int = 2
            c: int
            d: int = 4

        msg = proto.encode([5, 6, 7, 8])
        res = proto.decode(msg, type=Test)
        assert res == Test(a=5, b=6, c=7, d=8)

        msg = proto.encode([5, 6, 7])
        res = proto.decode(msg, type=Test)
        assert res == Test(a=5, b=6, c=7, d=4)

        msg = proto.encode([5, 6])
        with pytest.raises(
            ValidationError,
            match="Expected `array` of at least length 3, got 2",
        ):
            proto.decode(msg, type=Test)

        msg = proto.encode([])
        with pytest.raises(
            ValidationError,
            match="Expected `array` of at least length 3, got 0",
        ):
            proto.decode(msg, type=Test)


class TestStructDefaults:
    def test_struct_defaults(self, proto):
        class Test(Struct):
            a: int = 1
            b: list = []  # noqa
            c: int = rune.field(default=2)
            d: dict = rune.field(default_factory=dict)

        sol = Test()

        res = proto.decode(proto.encode(sol), type=Test)
        assert res == sol

        res = proto.decode(proto.encode({}), type=Test)
        assert res == sol

    def test_struct_default_factory_errors(self, proto):
        def bad():
            raise ValueError("Oh no!")

        class Test(Struct):
            a: int = rune.field(default_factory=bad)

        msg = proto.encode({})
        with pytest.raises(Exception, match="Oh no!"):
            proto.decode(msg, type=Test)


class TestDate:
    def test_encode_date(self, proto):
        # All fields, zero padded
        x = datetime.date(1, 2, 3)
        s = proto.decode(proto.encode(x))
        assert s == "0001-02-03"

        # All fields, no zeros
        x = datetime.date(1234, 12, 31)
        s = proto.decode(proto.encode(x))
        assert s == "1234-12-31"

    @pytest.mark.parametrize(
        "s",
        [
            "0001-01-01",
            "9999-12-31",
            "0001-02-03",
            "2020-02-29",
        ],
    )
    def test_decode_date(self, proto, s):
        sol = datetime.date.fromisoformat(s)
        res = proto.decode(proto.encode(s), type=datetime.date)
        assert type(res) is datetime.date
        assert res == sol

    def test_decode_date_wrong_type(self, proto):
        msg = proto.encode([])
        with pytest.raises(ValidationError, match="Expected `date`, got `array`"):
            proto.decode(msg, type=datetime.date)

    @pytest.mark.parametrize(
        "s",
        [
            # Incorrect field lengths
            "001-02-03",
            "0001-2-03",
            "0001-02-3",
            # Trailing data
            "0001-02-0300",
            # Truncated
            "0001-02-",
            # Invalid characters
            "000a-02-03",
            "0001-0a-03",
            "0001-02-0a",
            # Year out of range
            "0000-02-03",
            # Month out of range
            "0001-00-03",
            "0001-13-03",
            # Day out of range for month
            "0001-02-00",
            "0001-02-29",
            "2000-02-30",
        ],
    )
    def test_decode_date_malformed(self, proto, s):
        msg = proto.encode(s)
        with pytest.raises(ValidationError, match="Invalid RFC3339"):
            proto.decode(msg, type=datetime.date)


UTC = datetime.timezone.utc


class TestTime:
    @staticmethod
    def parse(t_str):
        t_str = t_str.replace("Z", "+00:00")
        return datetime.time.fromisoformat(t_str)

    @pytest.mark.parametrize(
        "t",
        [
            "00:00:00",
            "01:02:03",
            "01:02:03.000004",
            "12:34:56.789000",
            "23:59:59.999999",
        ],
    )
    def test_encode_time_naive(self, proto, t):
        res = proto.encode(self.parse(t))
        sol = proto.encode(t)
        assert res == sol

    @pytest.mark.parametrize(
        "t",
        [
            "00:00:00",
            "01:02:03",
            "01:02:03.000004",
            "12:34:56.789000",
            "23:59:59.999999",
        ],
    )
    def test_decode_time_naive(self, proto, t):
        sol = self.parse(t)
        res = proto.decode(proto.encode(t), type=datetime.time)
        assert type(res) is datetime.time
        assert res == sol

    def test_decode_time_wrong_type(self, proto):
        msg = proto.encode([])
        with pytest.raises(ValidationError, match="Expected `time`, got `array`"):
            proto.decode(msg, type=datetime.time)

    @pytest.mark.parametrize(
        "offset",
        [
            datetime.timedelta(0),
            datetime.timedelta(days=1, microseconds=-1),
            datetime.timedelta(days=-1, microseconds=1),
            datetime.timedelta(days=1, seconds=-29),
            datetime.timedelta(days=-1, seconds=29),
            datetime.timedelta(days=0, seconds=30),
            datetime.timedelta(days=0, seconds=-30),
        ],
    )
    def test_encode_time_offset_is_appx_equal_to_utc(self, proto, offset):
        x = datetime.time(14, 56, 27, 123456, datetime.timezone(offset))
        res = proto.encode(x)
        sol = proto.encode("14:56:27.123456Z")
        assert res == sol

    @pytest.mark.parametrize(
        "offset, t_str",
        [
            (
                datetime.timedelta(days=1, seconds=-30),
                "14:56:27.123456+23:59",
            ),
            (
                datetime.timedelta(days=-1, seconds=30),
                "14:56:27.123456-23:59",
            ),
            (
                datetime.timedelta(minutes=19, seconds=32, microseconds=130000),
                "14:56:27.123456+00:20",
            ),
        ],
    )
    def test_encode_time_offset_rounds_to_nearest_minute(self, proto, offset, t_str):
        x = datetime.time(14, 56, 27, 123456, datetime.timezone(offset))
        res = proto.encode(x)
        sol = proto.encode(t_str)
        assert res == sol

    def test_encode_time_zoneinfo(self):
        import zoneinfo

        try:
            x = datetime.datetime(2023, 1, 2, 3, 4, 5, 678, zoneinfo.ZoneInfo("America/Chicago"))
        except zoneinfo.ZoneInfoNotFoundError:
            # Some envs in CI do not have `tzdata`:
            pytest.skip(reason="Failed to load timezone")
        sol = rune.json.encode(x.isoformat())
        res = rune.json.encode(x)
        assert res == sol

    @pytest.mark.parametrize(
        "dt",
        [
            "04:05:06.000007",
            "04:05:06.007",
            "04:05:06",
            "21:19:22.123456",
        ],
    )
    @pytest.mark.parametrize("suffix", ["", "Z", "+00:00", "-00:00"])
    def test_decode_time_utc(self, proto, dt, suffix):
        dt += suffix
        sol = self.parse(dt)
        msg = proto.encode(sol)
        res = proto.decode(msg, type=datetime.time)
        assert res == sol

    @pytest.mark.parametrize("t", ["00:00:01", "12:01:01"])
    @pytest.mark.parametrize("sign", ["-", "+"])
    @pytest.mark.parametrize("hour", [0, 8, 12, 16, 23])
    @pytest.mark.parametrize("minute", [0, 30])
    def test_decode_time_with_timezone(self, proto, t, sign, hour, minute):
        s = f"{t}{sign}{hour:02}:{minute:02}"
        msg = proto.encode(s)
        res = proto.decode(msg, type=datetime.time)
        sol = self.parse(s)
        assert res == sol

    @pytest.mark.parametrize("z", ["Z", "z"])
    def test_decode_time_not_case_sensitive(self, proto, z):
        """Z can be upper/lowercase"""
        sol = datetime.time(4, 5, 6, 7, UTC)
        res = proto.decode(proto.encode(f"04:05:06.000007{z}"), type=datetime.time)
        assert res == sol

    @pytest.mark.parametrize(
        "lax, strict",
        [
            ("03:04:05+0102", "03:04:05+01:02"),
            ("03:04:05-0102", "03:04:05-01:02"),
        ],
    )
    def test_decode_time_rfc3339_relaxed(self, lax, strict, proto):
        """msgspec supports a few relaxations of the RFC3339 format."""
        sol = datetime.time.fromisoformat(strict)
        msg = proto.encode(lax)
        res = proto.decode(msg, type=datetime.time)
        assert res == sol

    @pytest.mark.parametrize(
        "t, sol",
        [
            (
                "03:04:05.1234564Z",
                datetime.time(3, 4, 5, 123456, UTC),
            ),
            (
                "03:04:05.1234565Z",
                datetime.time(3, 4, 5, 123457, UTC),
            ),
            (
                "03:04:05.12345650000000000001Z",
                datetime.time(3, 4, 5, 123457, UTC),
            ),
            (
                "03:04:05.9999995Z",
                datetime.time(3, 4, 6, 0, UTC),
            ),
            (
                "03:04:59.9999995Z",
                datetime.time(3, 5, 0, 0, UTC),
            ),
            (
                "03:59:59.9999995Z",
                datetime.time(4, 0, 0, 0, UTC),
            ),
            (
                "23:59:59.9999995Z",
                datetime.time(0, 0, 0, 0, UTC),
            ),
        ],
    )
    def test_decode_time_nanos(self, proto, t, sol):
        msg = proto.encode(t)
        res = proto.decode(msg, type=datetime.time)
        assert res == sol

    @pytest.mark.parametrize(
        "s",
        [
            # Incorrect field lengths
            "1:02:03.0000004Z",
            "01:2:03.0000004Z",
            "01:02:3.0000004Z",
            "01:02:03.0000004+5:06",
            "01:02:03.0000004+05:6",
            "01:02:03.0000004+056",
            "01:02:03.0000004+05600",
            # Trailing data
            "01:02:030",
            "01:02:03a",
            "01:02:03.a",
            "01:02:03.0a",
            "01:02:03.0000004a",
            "01:02:03.0000004+00:000",
            "01:02:03.0000004+00000",
            "01:02:03.0000004Z0",
            # Truncated
            "01:02:3",
            # Missing +/-
            "01:02:0300:00",
            # Missing digits after decimal
            "01:02:03.",
            "01:02:03.Z",
            # Invalid characters
            "0a:02:03.004+05:06",
            "01:0a:03.004+05:06",
            "01:02:0a.004+05:06",
            "01:02:03.00a+05:06",
            "01:02:03.004+0a:06",
            "01:02:03.004+05:0a",
            "01:02:03.004+0a06",
            "01:02:03.004+050a",
            # Hour out of range
            "24:02:03.004",
            # Minute out of range
            "01:60:03.004",
            # Second out of range
            "01:02:60.004",
            # Timezone hour out of range
            "01:02:03.004+24:00",
            "01:02:03.004-24:00",
            # Timezone minute out of range
            "01:02:03.004+00:60",
            "01:02:03.004-00:60",
        ],
    )
    def test_decode_time_malformed(self, proto, s):
        msg = proto.encode(s)
        with pytest.raises(ValidationError, match="Invalid RFC3339"):
            proto.decode(msg, type=datetime.time)


class TestTimeDelta:
    @pytest.mark.parametrize("neg", [False, True])
    @pytest.mark.parametrize(
        "td, msg",
        [
            (datetime.timedelta(), "P0D"),
            (datetime.timedelta(1), "P1D"),
            (datetime.timedelta(10), "P10D"),
            (datetime.timedelta(123456789), "P123456789D"),
            (datetime.timedelta(0, 1), "PT1S"),
            (datetime.timedelta(0, 10), "PT10S"),
            (datetime.timedelta(0, 12345), "PT12345S"),
            (datetime.timedelta(0, 0, 1), "PT0.000001S"),
            (datetime.timedelta(0, 0, 10), "PT0.00001S"),
            (datetime.timedelta(0, 0, 100), "PT0.0001S"),
            (datetime.timedelta(0, 0, 1000), "PT0.001S"),
            (datetime.timedelta(0, 0, 10000), "PT0.01S"),
            (datetime.timedelta(0, 0, 100000), "PT0.1S"),
            (datetime.timedelta(123456789, 54321, 123456), "P123456789DT54321.123456S"),
            (datetime.timedelta(0, 86399, 999999), "PT86399.999999S"),
        ],
    )
    def test_roundtrip_timedelta(self, proto, td, msg, neg):
        if neg and td:
            td = -td
            msg = "-" + msg

        buf = proto.encode(td)

        res = proto.decode(buf)
        assert res == msg

        td2 = proto.decode(buf, type=datetime.timedelta)
        assert td2 == td

    @pytest.mark.parametrize(
        "msg, sol",
        [
            ("PT0S", datetime.timedelta()),
            ("+P1DT2S", datetime.timedelta(1, 2)),
            ("-P1DT2S", -datetime.timedelta(1, 2)),
            ("P000DT000.000S", datetime.timedelta()),
            ("-P000DT000.000S", datetime.timedelta()),
            ("P00012DT0045.670000000S", datetime.timedelta(12, 45, 670000)),
            ("P123456789.12345678912D", datetime.timedelta(123456789, 10666, 666580)),
            ("P123456789.12345678912999D", datetime.timedelta(123456789, 10666, 666580)),
            ("P123456789.12345678913D", datetime.timedelta(123456789, 10666, 666581)),
            ("PT0123H", datetime.timedelta(0, 123 * 60 * 60)),
            ("PT0123.456H", datetime.timedelta(0, 123.456 * 60 * 60)),
            ("PT0123M", datetime.timedelta(0, 123 * 60)),
            ("PT0123.456M", datetime.timedelta(0, 123.456 * 60)),
        ],
    )
    def test_decode_timedelta(self, proto, msg, sol):
        buf = proto.encode(msg)
        res = proto.decode(buf, type=datetime.timedelta)
        assert res == sol

    def test_decode_timedelta_case_insensitive(self, proto):
        buf = proto.encode("p1dt2h3m4s")
        res = proto.decode(buf, type=datetime.timedelta)
        assert res == datetime.timedelta(1, 2 * 60 * 60 + 3 * 60 + 4)

    @pytest.mark.parametrize(
        "msg",
        [
            "P999999999DT86399.999999S",
            "P999999998DT24H86399.999999S",
            "P999999999DT86399.9999994S",
        ],
    )
    def test_decode_timedelta_max(self, proto, msg):
        buf = proto.encode(msg)
        res = proto.decode(buf, type=datetime.timedelta)
        assert res == datetime.timedelta.max

    @pytest.mark.parametrize(
        "msg",
        [
            "-P999999999D",
            "-P999999998DT24H",
            "-P999999998DT23H3600S",
            "-P999999998DT86399.9999995S",
        ],
    )
    def test_decode_timedelta_min(self, proto, msg):
        buf = proto.encode(msg)
        res = proto.decode(buf, type=datetime.timedelta)
        assert res == datetime.timedelta.min

    def test_decode_timedelta_wrong_type(self, proto):
        bad = proto.encode([])
        with pytest.raises(ValidationError, match="Expected `duration`, got `array`"):
            proto.decode(bad, type=datetime.timedelta)

    @pytest.mark.parametrize(
        "msg",
        [
            # No P
            "",
            "-",
            "+",
            # Just P
            "P",
            "-P",
            "+P",
            # Missing Number
            "PD",
            "P.0D",
            # Missing digit after decimal place
            "P123.",
            "P123.D",
            # Missing Unit
            "P0",
            "P0.0",
            "P0.00",
            "P0.000000000000123",
            # Trailing T
            "PT",
            "P0DT",
            # Missing T
            "P1D2H",
            "P1D2S",
            # Repeat T
            "PTT0S",
            # Repeat Units
            "P1D2D",
            "PT1H2H",
            "PT1M2M",
            "PT1S2S",
            # Units in wrong order
            "PT1H1D",
            "PT1M1H",
            "PT1S1M",
            # Non-fractional after fractional
            "PT1.2H1M",
            "P1.2DT1H",
            "PT1.2H0S",
            # Invalid characters
            "1P1D",
            "P-1D",
            "P1.-D",
            "P1.0-D",
            "P1.000000000000123-D",
            "P1D-",
        ],
    )
    def test_decode_timedelta_malformed(self, proto, msg):
        encoded = proto.encode(msg)
        with pytest.raises(ValidationError, match="Invalid ISO8601 duration"):
            proto.decode(encoded, type=datetime.timedelta)

    @pytest.mark.parametrize(
        "msg",
        [
            "P1000000000D",
            "PT140737488355329S",
            "P999999999DT86399.9999995S",
            "-P999999999DT0.0000005S",
            "P999999998DT48H",
            "-P999999998DT24H01S",
        ],
    )
    def test_decode_timedelta_out_of_range(self, proto, msg):
        encoded = proto.encode(msg)
        with pytest.raises(ValidationError, match="Duration is out of range"):
            proto.decode(encoded, type=datetime.timedelta)

    @pytest.mark.parametrize("unit", ["Y", "M", "W"])
    def test_decode_timedelta_unsupported_unit(self, proto, unit):
        upper = f"P1{unit}"
        for msg in [upper, upper.lower()]:
            encoded = proto.encode(msg)
            with pytest.raises(ValidationError, match="Only units 'D'"):
                proto.decode(encoded, type=datetime.timedelta)


class TestNewType:
    def test_decode_newtype(self, proto):
        UserId = typing.NewType("UserId", int)
        assert proto.decode(proto.encode(1), type=UserId) == 1

        with pytest.raises(ValidationError):
            proto.decode(proto.encode("bad"), type=UserId)

        # Nested NewId works
        UserId2 = typing.NewType("UserId2", UserId)
        assert proto.decode(proto.encode(1), type=UserId2) == 1

        with pytest.raises(ValidationError):
            proto.decode(proto.encode("bad"), type=UserId2)

    def test_decode_annotated_newtype(self, proto):
        UserId = typing.NewType("UserId", int)
        dec = proto.Decoder(typing.Annotated[UserId, rune.Meta(ge=0)])
        assert dec.decode(proto.encode(1)) == 1

        with pytest.raises(ValidationError):
            dec.decode(proto.encode(-1))

    def test_decode_newtype_annotated(self, proto):
        UserId = typing.NewType("UserId", typing.Annotated[int, rune.Meta(ge=0)])
        dec = proto.Decoder(UserId)
        assert dec.decode(proto.encode(1)) == 1

        with pytest.raises(ValidationError):
            dec.decode(proto.encode(-1))

    def test_decode_annotated_newtype_annotated(self, proto):
        UserId = typing.Annotated[
            typing.NewType("UserId", typing.Annotated[int, rune.Meta(ge=0)]), rune.Meta(le=10)
        ]
        dec = proto.Decoder(UserId)
        assert dec.decode(proto.encode(1)) == 1

        for bad in [-1, 11]:
            with pytest.raises(ValidationError):
                dec.decode(proto.encode(bad))


class TestFinal:
    def test_decode_final(self, proto):
        dec = proto.Decoder(typing.Final[int])

        assert dec.decode(proto.encode(1)) == 1
        with pytest.raises(ValidationError):
            dec.decode(proto.encode("bad"))

    def test_decode_final_annotated(self, proto):
        dec = proto.Decoder(typing.Final[typing.Annotated[int, rune.Meta(ge=0)]])

        assert dec.decode(proto.encode(1)) == 1
        with pytest.raises(ValidationError):
            dec.decode(proto.encode(-1))

    def test_decode_final_newtype(self, proto):
        UserId = typing.NewType("UserId", int)
        dec = proto.Decoder(typing.Final[UserId])

        assert dec.decode(proto.encode(1)) == 1
        with pytest.raises(ValidationError):
            dec.decode(proto.encode("bad"))


class TestSet:
    def test_encode_set_doesnt_leak_item_refs(self, proto):
        class Ex:
            pass

        items = {Ex(), Ex(), Ex()}
        refs = [weakref.ref(item) for item in items]

        proto.Encoder(enc_hook=lambda x: None).encode(items)

        del items
        gc.collect()

        assert all(ref() is None for ref in refs)


class TestOrder:
    def test_encoder_order_attribute(self, proto):
        enc = proto.Encoder()
        assert enc.order is None

        enc = proto.Encoder(order=None)
        assert enc.order is None

        enc = proto.Encoder(order="deterministic")
        assert enc.order == "deterministic"

        enc = proto.Encoder(order="sorted")
        assert enc.order == "sorted"

    def test_order_invalid(self, proto):
        with pytest.raises(ValueError, match="`order` must be one of"):
            proto.Encoder(order="bad")

        with pytest.raises(ValueError, match="`order` must be one of"):
            proto.encode(1, order="bad")

    @pytest.mark.parametrize("msg", [{}, {"y": 1, "x": 2, "z": 3}])
    @pytest.mark.parametrize("order", [None, "deterministic", "sorted"])
    @pytest.mark.parametrize("use_encoder", [False, True])
    def test_order_dict(self, msg, order, use_encoder, proto):
        if use_encoder:
            res = proto.Encoder(order=order).encode(msg)
        else:
            res = proto.encode(msg, order=order)

        if order is not None:
            sol = proto.encode(dict(sorted(msg.items())))
        else:
            sol = proto.encode(msg)

        assert res == sol

    def test_order_dict_non_str_errors(self, proto):
        with pytest.raises(TypeError, match="Only dicts with str keys"):
            proto.encode({"b": 2, 1: "a"}, order="deterministic")

    def test_order_dict_unsortable(self, proto):
        with pytest.raises(TypeError):
            proto.encode({"x": 1, 1: 2}, order="deterministic")

    @py315_or_later_only
    @pytest.mark.parametrize("msg", [{}, {"y": 1, "x": 2, "z": 3}])
    @pytest.mark.parametrize("order", [None, "deterministic", "sorted"])
    @pytest.mark.parametrize("use_encoder", [False, True])
    def test_order_frozendict(self, msg, order, use_encoder, proto):
        if use_encoder:
            res = proto.Encoder(order=order).encode(frozendict(msg))
        else:
            res = proto.encode(frozendict(msg), order=order)

        # `frozendict` must be equal to the regular `dict`:
        assert isinstance(msg, dict)
        if order is not None:
            sol = proto.encode(dict(sorted(msg.items())))
        else:
            sol = proto.encode(msg)

        assert isinstance(res, bytes)
        assert res == sol

    @py315_or_later_only
    def test_order_frozendict_non_str_errors(self, proto):
        with pytest.raises(TypeError, match="Only dicts with str keys"):
            proto.encode(frozendict({"b": 2, 1: "a"}), order="deterministic")

    @py315_or_later_only
    def test_order_frozendict_unsortable(self, proto):
        with pytest.raises(TypeError):
            proto.encode(frozendict({"x": 1, 1: 2}), order="deterministic")

    @pytest.mark.parametrize("typ", [set, frozenset])
    @pytest.mark.parametrize("order", ["deterministic", "sorted"])
    def test_order_set(self, typ, proto, rand, order):
        assert proto.encode(typ(), order=order) == proto.encode([])

        msg = typ(rand.str(10) for _ in range(20))

        res = proto.encode(msg, order=order)
        sol = proto.encode(list(sorted(msg)))
        assert res == sol

        res = proto.encode(msg)
        sol = proto.encode(list(msg))
        assert res == sol

    def test_order_set_unsortable(self, proto):
        with pytest.raises(TypeError):
            proto.encode({"x", 1}, order="deterministic")

    @pytest.mark.parametrize("n", [0, 1, 2])
    @pytest.mark.parametrize(
        "kind",
        [
            "struct",
            "dataclass",
            "attrs",
            "attrs-dict",
        ],
    )
    def test_order_object(self, kind, n, proto):
        fields = [f"x{i}" for i in range(n)]
        fields.reverse()
        if kind == "struct":
            cls = rune.defstruct("Test", fields)
        elif kind == "dataclass":
            cls = make_dataclass("Test", fields)
        else:
            attrs = pytest.importorskip("attrs")
            cls = attrs.make_class("Test", fields, slots=(kind == "attrs"))
        msg = cls(*range(n))

        if kind in ("struct", "dataclass"):
            # we currently don't guarantee field order with attrs types
            sol = proto.encode(dict(zip(fields, range(n), strict=False)))
            res = proto.encode(msg)
            assert res == sol

            res = proto.encode(msg, order="deterministic")
            assert res == sol

        res = proto.encode(msg, order="sorted")
        sol = proto.encode(dict(sorted(zip(fields, range(n), strict=False))))
        assert res == sol

    @pytest.mark.parametrize("kind", ["struct", "dataclass", "attrs", "attrs-dict"])
    def test_order_unset(self, kind, proto):
        if kind == "struct":

            class Ex(Struct):
                z: int | UnsetType = UNSET
                x: int | UnsetType = UNSET
        elif kind == "dataclass":

            @dataclass
            class Ex:
                z: int | UnsetType = UNSET
                x: int | UnsetType = UNSET
        else:
            attrs = pytest.importorskip("attrs")

            @attrs.define(slots=(kind == "attrs"))
            class Ex:
                z: int | UnsetType = UNSET
                x: int | UnsetType = UNSET

        res = proto.encode(Ex(), order="sorted")
        sol = proto.encode({})
        assert res == sol

        res = proto.encode(Ex(z=10), order="sorted")
        sol = proto.encode({"z": 10})
        assert res == sol

        res = proto.encode(Ex(z=10, x=-1), order="sorted")
        sol = proto.encode({"x": -1, "z": 10})
        assert res == sol

    def test_order_struct_omit_defaults(self, proto):
        class Ex(Struct, omit_defaults=True):
            z: int = 0
            x: int = 1
            y: int = 2

        res = proto.encode(Ex(), order="sorted")
        sol = proto.encode({})
        assert res == sol

        res = proto.encode(Ex(z=10), order="sorted")
        sol = proto.encode({"z": 10})
        assert res == sol

        res = proto.encode(Ex(z=10, x=-1), order="sorted")
        sol = proto.encode({"x": -1, "z": 10})
        assert res == sol

    def test_order_struct_tag(self, proto):
        class Ex(Struct, tag_field="y", tag=2):
            z: int
            x: int

        res = proto.encode(Ex(0, 1), order="sorted")
        sol = proto.encode({"x": 1, "y": 2, "z": 0})
        assert res == sol

    @pytest.mark.parametrize("n", [0, 2, 3, 7, 15, 16, 17, 32, 100, 500, 1000, 10000])
    def test_order_sort_implementation(self, rand, n):
        keys = [f"x_{i}" for i in range(n)]
        rand.shuffle(keys)
        msg = dict(zip(keys, range(n), strict=False))
        res = rune.json.encode(msg, order="deterministic")
        sol = rune.json.encode(dict(sorted(msg.items())))
        assert res == sol
