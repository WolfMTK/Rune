import base64
import collections
import enum
import gc
import sys
import types
import typing
import weakref
from collections import deque
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any, Literal

import pytest
from rune.struct import Struct

import rune
from rune import ValidationError

from .utils import PY311, PY312, max_call_depth


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
