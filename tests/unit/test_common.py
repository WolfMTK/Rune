import base64
from collections import deque
from types import ModuleType
from typing import Any

import pytest
from rune.struct import Struct

import rune

from .utils import PY312, max_call_depth


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
