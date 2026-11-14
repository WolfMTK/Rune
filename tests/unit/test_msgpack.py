import array
import datetime
import enum
import gc
import itertools
import math
import pickle
import struct
import sys
from typing import (
    Any,
    Literal,
    Optional,
)

import pytest

import rune

from .utils import emscripten_stack_limited

UTC = datetime.timezone.utc


class FruitInt(enum.IntEnum):
    APPLE = 1
    BANANA = 2


class FruitStr(enum.Enum):
    APPLE = "apple"
    BANANA = "banana"


class Person(rune.Struct):
    first: str
    last: str
    age: int
    prefect: bool = False


class PersonArray(rune.Struct, array_like=True):
    first: str
    last: str
    age: int
    prefect: bool = False


PERSON = Person("harry", "potter", 13)
PERSON_AA = PersonArray("harry", "potter", 13)


class Node(rune.Struct):
    left: "Node | None" = None
    right: "Node | None" = None


class Custom:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


INTS = [
    -(2**63),
    -(2**31 + 1),
    -(2**31),
    -(2**15 + 1),
    -(2**15),
    -(2**7 + 1),
    -(2**7),
    -(2**5 + 1),
    -(2**5),
    -1,
    0,
    1,
    2**7 - 1,
    2**7,
    2**8 - 1,
    2**8,
    2**16 - 1,
    2**16,
    2**32 - 1,
    2**32,
    2**63 - 1,
    2**64 - 1,
]

FLOATS = [
    -1.5,
    0.0,
    1.5,
    -float("inf"),
    float("inf"),
    float("nan"),
    sys.float_info.max,
    sys.float_info.min,
    -sys.float_info.max,
    -sys.float_info.min,
]

SIZES = [0, 1, 31, 32, 2**8 - 1, 2**8, 2**16 - 1, 2**16]


def assert_eq(x, y):
    if isinstance(x, float) and math.isnan(x):
        assert math.isnan(y)
    else:
        assert x == y


class TestEncodeFunction:
    def test_encode(self):
        dec = rune.msgpack.Decoder()
        assert dec.decode(rune.msgpack.encode(1)) == 1

    def test_encode_bad_arguments(self):
        with pytest.raises(TypeError, match="Missing 1 required arguments"):
            rune.msgpack.encode()

        with pytest.raises(TypeError, match="Extra positional"):
            rune.msgpack.encode(1, 2)

        with pytest.raises(TypeError, match="enc_hook must be callable"):
            rune.msgpack.encode(1, enc_hook="bad")

    def test_encode_error(self):
        with pytest.raises(TypeError):
            rune.msgpack.encode(object())

    def test_encode_large_object(self):
        """Check that buffer resize works"""
        data = b"x" * 4097
        dec = rune.msgpack.Decoder()
        assert dec.decode(rune.msgpack.encode(data)) == data

    def test_encode_no_enc_hook(self):
        class Foo:
            pass

        with pytest.raises(TypeError, match="Encoding objects of type Foo is unsupported"):
            rune.msgpack.encode(Foo())

        with pytest.raises(TypeError, match="Encoding objects of type Foo is unsupported"):
            rune.msgpack.encode(Foo(), enc_hook=None)

    def test_encode_enc_hook(self):
        unsupported = object()

        def enc_hook(x):
            assert x is unsupported
            return "hello"

        orig_refcount = sys.getrefcount(enc_hook)

        res = rune.msgpack.encode(unsupported, enc_hook=enc_hook)
        assert rune.msgpack.encode("hello") == res
        assert sys.getrefcount(enc_hook) == orig_refcount

    def test_encode_enc_hook_errors(self):
        def enc_hook(x):
            raise TypeError("bad")

        orig_refcount = sys.getrefcount(enc_hook)

        with pytest.raises(TypeError, match="bad"):
            rune.msgpack.encode(object(), enc_hook=enc_hook)

        assert sys.getrefcount(enc_hook) == orig_refcount

    def test_encode_parse_arguments_errors(self):
        with pytest.raises(TypeError, match="Missing 1 required argument"):
            rune.msgpack.encode()

        with pytest.raises(TypeError, match="Extra positional arguments"):
            rune.msgpack.encode(1, lambda x: None)

        with pytest.raises(TypeError, match="Extra positional arguments"):
            rune.msgpack.encode(1, 2, 3)

        with pytest.raises(TypeError, match="Extra keyword arguments"):
            rune.msgpack.encode(1, bad=1)

        with pytest.raises(TypeError, match="Extra keyword arguments"):
            rune.msgpack.encode(1, enc_hook=lambda x: None, extra="extra")


class TestDecodeFunction:
    def setup_method(self):
        self.buf = rune.msgpack.encode([1, 2, 3])

    def test_decode(self):
        assert rune.msgpack.decode(self.buf) == [1, 2, 3]

    def test_decode_type_keyword(self):
        assert rune.msgpack.decode(self.buf, type=list[int]) == [1, 2, 3]

        with pytest.raises(rune.ValidationError):
            assert rune.msgpack.decode(self.buf, type=list[str])

    def test_decode_type_any(self):
        assert rune.msgpack.decode(self.buf, type=Any) == [1, 2, 3]

    @pytest.mark.parametrize("array_like", [False, True])
    def test_decode_type_struct(self, array_like):
        class Point(rune.Struct, array_like=array_like):
            x: int
            y: int

        msg = rune.msgpack.encode(Point(1, 2))

        for _ in range(2):
            assert rune.msgpack.decode(msg, type=Point) == Point(1, 2)

    def test_decode_type_struct_not_json_compatible(self):
        class Test(rune.Struct):
            x: dict[int, str]

        msg = rune.msgpack.encode(Test({1: "two"}))
        rune.msgpack.decode(msg, type=Test) == Test({1, "two"})  # noqa

    def test_decode_type_struct_invalid_type(self):
        class Test(rune.Struct):
            x: 1

        with pytest.raises(TypeError):
            rune.msgpack.decode(b"{}", type=Test)

    def test_decode_invalid_type(self):
        with pytest.raises(TypeError, match="Type '1' is not supported"):
            rune.msgpack.decode(self.buf, type=1)

    def test_decode_invalid_buf(self):
        with pytest.raises(TypeError):
            rune.msgpack.decode(1)

    def test_decode_parse_arguments_errors(self):
        with pytest.raises(TypeError, match="Missing 1 required argument"):
            rune.msgpack.decode()

        with pytest.raises(TypeError, match="Extra positional arguments"):
            rune.msgpack.decode(self.buf, list[int])

        with pytest.raises(TypeError, match="Extra positional arguments"):
            rune.msgpack.decode(self.buf, 2, 3)

        with pytest.raises(TypeError, match="Extra keyword arguments"):
            rune.msgpack.decode(self.buf, bad=1)

        with pytest.raises(TypeError, match="Extra keyword arguments"):
            rune.msgpack.decode(self.buf, type=list[int], extra=1)

    def test_decode_with_trailing_characters_errors(self):
        msg = rune.msgpack.encode([1, 2, 3]) + b"trailing"

        with pytest.raises(rune.DecodeError):
            rune.msgpack.decode(msg)


class TestEncoderMisc:
    def test_encoder_init_errors(self):
        with pytest.raises(TypeError):
            rune.msgpack.Encoder(bad=1)

        with pytest.raises(TypeError, match="enc_hook must be callable"):
            rune.msgpack.Encoder(enc_hook=1)

    @pytest.mark.parametrize("x", [-(2**63) - 1, 2**64])
    def test_encode_integer_limits(self, x):
        enc = rune.msgpack.Encoder()
        with pytest.raises(OverflowError):
            enc.encode(x)

    def rec_obj1(self):
        o = []
        o.append(o)
        return o

    def rec_obj2(self):
        o = ([],)
        o[0].append(o)
        return o

    def rec_obj3(self):
        o = {}
        o["a"] = o
        return o

    def rec_obj4(self):
        class Box(rune.Struct):
            a: "Box"

        o = Box(None)
        o.a = o
        return o

    @pytest.mark.parametrize("case", [1, 2, 3, 4])
    @emscripten_stack_limited
    def test_encode_infinite_recursive_object_errors(self, case):
        enc = rune.msgpack.Encoder()
        o = getattr(self, "rec_obj%d" % case)()  # noqa
        with pytest.raises(RecursionError):
            enc.encode(o)

    def test_encode_no_enc_hook(self):
        class Foo:
            pass

        enc = rune.msgpack.Encoder()
        assert enc.enc_hook is None

        enc = rune.msgpack.Encoder(enc_hook=None)
        assert enc.enc_hook is None

        with pytest.raises(TypeError, match="Encoding objects of type Foo is unsupported"):
            enc.encode(Foo())

    def test_encode_enc_hook(self):
        unsupported = object()

        def enc_hook(x):
            assert x is unsupported
            return "hello"

        orig_refcount = sys.getrefcount(enc_hook)

        enc = rune.msgpack.Encoder(enc_hook=enc_hook)

        assert enc.enc_hook is enc_hook
        assert sys.getrefcount(enc.enc_hook) == orig_refcount + 2
        assert sys.getrefcount(enc_hook) == orig_refcount + 1

        res = enc.encode(unsupported)
        assert enc.encode("hello") == res

        del enc
        assert sys.getrefcount(enc_hook) == orig_refcount

    def test_encode_enc_hook_errors(self):
        def enc_hook(x):
            raise TypeError("bad")

        enc = rune.msgpack.Encoder(enc_hook=enc_hook)

        with pytest.raises(TypeError, match="bad"):
            enc.encode(object())

    def test_encode_enc_hook_recurses(self):
        class Node:
            def __init__(self, a):
                self.a = a

        def enc_hook(x):
            return {"type": "Node", "a": x.a}

        enc = rune.msgpack.Encoder(enc_hook=enc_hook)

        msg = enc.encode(Node(Node(1)))
        res = rune.msgpack.decode(msg)
        assert res == {"type": "Node", "a": {"type": "Node", "a": 1}}

    @emscripten_stack_limited
    def test_encode_enc_hook_recursion_error(self):
        enc = rune.msgpack.Encoder(enc_hook=lambda x: x)

        with pytest.raises(RecursionError):
            enc.encode(object())

    def test_encode_bad_arguments(self):
        enc = rune.msgpack.Encoder()

        with pytest.raises(TypeError, match="Missing 1 required arguments"):
            enc.encode()

        with pytest.raises(TypeError, match="Extra positional"):
            enc.encode(1, 2)

    def test_encode_into_bad_arguments(self):
        enc = rune.msgpack.Encoder()

        with pytest.raises(TypeError, match="bytearray"):
            enc.encode_into(1, b"test")

        with pytest.raises(TypeError):
            enc.encode_into(1, bytearray(), "bad")

        with pytest.raises(TypeError):
            enc.encode_into(1, bytearray(), None)

        with pytest.raises(ValueError, match="offset"):
            enc.encode_into(1, bytearray(), -2)

        with pytest.raises(TypeError, match="Missing 1 required arguments"):
            enc.encode_into(1)

        with pytest.raises(TypeError, match="Extra positional"):
            enc.encode_into(1, bytearray(), 2, 3)

    @pytest.mark.parametrize("buf_size", [0, 1, 16, 55, 60])
    def test_encode_into(self, buf_size):
        enc = rune.msgpack.Encoder()

        msg = {"key": "x" * 48}
        encoded = rune.msgpack.encode(msg)

        buf = bytearray(buf_size)
        out = enc.encode_into(msg, buf)
        assert out is None
        assert buf == encoded

    def test_encode_into_offset(self):
        enc = rune.msgpack.Encoder()
        msg = {"key": "value"}
        encoded = enc.encode(msg)

        # Offset 0 is default
        buf = bytearray()
        enc.encode_into(msg, buf, 0)
        assert buf == encoded

        # Offset in bounds uses the provided offset
        buf = bytearray(b"01234")
        enc.encode_into(msg, buf, 2)
        assert buf == b"01" + encoded

        # Offset out of bounds extends
        buf = bytearray(b"01234")
        enc.encode_into(msg, buf, 10)
        assert buf[:5] == b"01234"
        assert buf[10:] == encoded

        # Offset -1 means append at end
        buf = bytearray(b"01234")
        enc.encode_into(msg, buf, -1)
        assert buf == b"01234" + encoded

    def test_encode_into_handles_errors_properly(self):
        enc = rune.msgpack.Encoder()
        out1 = enc.encode([1, 2, 3])

        msg = [1, 2, object()]
        buf = bytearray()
        with pytest.raises(TypeError):
            enc.encode_into(msg, buf)

        assert buf  # buffer isn't reset upon error

        # Encoder still works
        out2 = enc.encode([1, 2, 3])
        assert out1 == out2

    @pytest.mark.parametrize(
        "dt, dt_str",
        [
            (datetime.datetime(1, 2, 3, 4, 5, 6), "0001-02-03T04:05:06"),
            (
                datetime.datetime(1234, 12, 31, 14, 56, 27, 123456),
                "1234-12-31T14:56:27.123456",
            ),
        ],
    )
    def test_encode_datetime_naive(self, dt, dt_str):
        res = rune.msgpack.encode(dt)
        sol = rune.msgpack.encode(dt_str)
        assert res == sol

    def test_encode_datetime_non_utc_tzinfo(self):
        tzinfo = datetime.timezone(datetime.timedelta(hours=1))
        x = datetime.datetime.now(tzinfo)
        x2 = x.astimezone(datetime.timezone.utc)
        res = rune.msgpack.encode(x)
        sol = rune.msgpack.encode(x2)
        assert res == sol


class TestDecoderMisc:
    def test_decoder_type_attribute(self):
        dec = rune.msgpack.Decoder()
        assert dec.type is Any

        dec = rune.msgpack.Decoder(int)
        assert dec.type is int

    def test_decoder_ext_hook_attribute(self):
        def ext_hook(code, buf):
            pass

        dec = rune.msgpack.Decoder()
        assert dec.ext_hook is None

        dec = rune.msgpack.Decoder(ext_hook=None)
        assert dec.ext_hook is None

        dec = rune.msgpack.Decoder(ext_hook=ext_hook)
        assert dec.ext_hook is ext_hook

    def test_decoder_bad_args(self):
        with pytest.raises(TypeError):
            rune.msgpack.Decoder(bad=1)

    def test_decoder_ext_hook_not_callable(self):
        with pytest.raises(TypeError):
            rune.msgpack.Decoder(ext_hook=1)

    def test_decoder_repr(self):
        typ = list[dict[int, float]]
        dec = rune.msgpack.Decoder(typ)
        assert repr(dec) == f"rune.msgpack.Decoder({typ!r})"

        dec = rune.msgpack.Decoder()
        assert repr(dec) == f"rune.msgpack.Decoder({Any!r})"

    def test_decode_with_trailing_characters_errors(self):
        dec = rune.msgpack.Decoder()

        msg = rune.msgpack.encode([1, 2, 3]) + b"trailing"

        with pytest.raises(rune.DecodeError):
            dec.decode(msg)

    def test_decode_invalid_opcode_error(self):
        with pytest.raises(rune.DecodeError, match="invalid opcode '\\\\xc1'"):
            rune.msgpack.decode(b"\xc1abc")

    @pytest.mark.parametrize(
        "hex_data",
        [
            "818001",  # map used as map key
            "878048",  # map used as map key
            "8bd4ec5c03",  # ext used as map key
        ],
    )
    def test_decode_unhashable_map_key_raises_decode_error(self, hex_data):
        with pytest.raises(rune.DecodeError, match="map keys must be hashable"):
            rune.msgpack.decode(bytes.fromhex(hex_data))

    def test_decode_hashable_array_map_key_ok(self):
        assert rune.msgpack.decode(bytes.fromhex("819001")) == {(): 1}

    def test_decode_skip_invalid_submessage_raises(self):
        """Ensure errors in submessage skipping are raised"""

        class Test(rune.Struct):
            x: int

        msg = rune.msgpack.encode({"x": 1, "y": ["one", "two", "three"]})

        # Break the message
        msg = msg.replace(b"three", b"tree")

        with pytest.raises(rune.DecodeError, match="truncated"):
            rune.msgpack.decode(msg, type=Test)

    @pytest.mark.parametrize("length", [3, 31, 33])
    @pytest.mark.parametrize("typed", [False, True])
    @pytest.mark.skipif(
        hasattr(sys.flags, "gil") and not sys.flags.gil,
        reason="cache is disabled without GIL",
    )
    def test_decode_dict_string_cache(self, length, typed):
        key = "x" * length
        msg = [{key: 1}, {key: 2}, {key: 3}]
        if typed:
            dec = rune.msgpack.Decoder(list[dict[str, int]])
        else:
            dec = rune.msgpack.Decoder()
        res = dec.decode(rune.msgpack.encode(msg))
        assert msg == res
        ids = {id(k) for d in res for k in d.keys()}
        if length > 32:
            assert len(ids) == 3
        else:
            assert len(ids) == 1

    def test_decode_dict_string_cache_ascii_only(self):
        """Short non-ascii strings aren't cached"""
        s = "123 á 456"
        msg = [{s: 1}, {s: 2}, {s: 3}]
        res = rune.msgpack.decode(rune.msgpack.encode(msg))
        ids = {id(k) for d in res for k in d.keys()}
        assert len(ids) == 3

    @pytest.mark.parametrize("type", [None, list, tuple, set])
    def test_decoding_large_arrays_doesnt_preallocate(self, type):
        # <maximum sized array, truncated>
        b = b"\xdd\xff\xff\xff\xff"
        with pytest.raises(rune.DecodeError, match="truncated"):
            if type is None:
                rune.msgpack.decode(b)
            else:
                rune.msgpack.decode(b, type=type)

    def test_decoding_large_arrays_as_keys_doesnt_preallocate(self):
        # {"x": <maximum sized array, truncated>}
        b = b"\x81\xa1x\xdd\xff\xff\xff\xff"
        with pytest.raises(rune.DecodeError, match="truncated"):
            rune.msgpack.decode(b)


class TestTypedDecoder:
    def check_unexpected_type(self, dec_type, val, msg):
        dec = rune.msgpack.Decoder(dec_type)
        s = rune.msgpack.Encoder().encode(val)
        with pytest.raises(rune.ValidationError, match=msg):
            dec.decode(s)

    def test_any(self):
        dec = rune.msgpack.Decoder(Any)
        assert dec.decode(rune.msgpack.encode([1, 2, 3])) == [1, 2, 3]

        # A union that includes `Any` is just `Any`
        dec = rune.msgpack.Decoder(Any | float | int | None)
        assert dec.decode(rune.msgpack.encode([1, 2, 3])) == [1, 2, 3]

    def test_none(self):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(None)
        assert dec.decode(enc.encode(None)) is None
        with pytest.raises(rune.ValidationError, match="Expected `null`"):
            assert dec.decode(enc.encode(1))

    @pytest.mark.parametrize("x", [False, True])
    def test_bool(self, x):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(bool)
        assert dec.decode(enc.encode(x)) is x

    def test_bool_unexpected_type(self):
        self.check_unexpected_type(bool, "a", "Expected `bool`")

    @pytest.mark.parametrize("x", INTS)
    def test_int(self, x):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(int)
        assert dec.decode(enc.encode(x)) == x

    def test_int_unexpected_type(self):
        self.check_unexpected_type(int, "a", "Expected `int`")

    @pytest.mark.parametrize("x", FLOATS + INTS)
    def test_float(self, x):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(float)
        res = dec.decode(enc.encode(x))
        sol = float(x)
        if math.isnan(sol):
            assert math.isnan(res)
        else:
            assert res == sol

    def test_float_unexpected_type(self):
        self.check_unexpected_type(float, "a", "Expected `float`")

    def test_decode_float4(self):
        x = 1.2
        packed = struct.pack(">f", x)
        # Loss of resolution in float32 leads to some rounding error
        x4 = struct.unpack(">f", packed)[0]
        msg = b"\xca" + packed
        assert rune.msgpack.decode(msg) == x4
        assert rune.msgpack.decode(msg, type=float) == x4

    @pytest.mark.parametrize("size", SIZES)
    def test_str(self, size):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(str)
        x = "a" * size
        res = dec.decode(enc.encode(x))
        assert res == x

    def test_str_unexpected_type(self):
        self.check_unexpected_type(str, 1, "Expected `str`")

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("typ", [bytes, bytearray, memoryview])
    def test_binary(self, size, typ):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(typ)
        sol = b"a" * size
        res = dec.decode(enc.encode(typ(sol)))
        assert isinstance(res, typ)
        assert bytes(res) == sol

    @pytest.mark.parametrize("typ", [bytes, bytearray, memoryview])
    def test_binary_unexpected_type(self, typ):
        self.check_unexpected_type(typ, 1, "Expected `bytes`")

    @pytest.mark.parametrize("input_type", [bytes, bytearray, memoryview])
    def test_decode_memoryview_zerocopy(self, input_type):
        msg = rune.msgpack.encode(b"abcde")
        ref = msg if input_type is memoryview else None
        msg = input_type(msg)
        res = rune.msgpack.decode(msg, type=memoryview)
        assert isinstance(res, memoryview)
        assert bytes(res) == b"abcde"
        assert len(res) == 5
        if input_type is memoryview:
            assert sys.getrefcount(ref) <= 3
            del msg
            assert sys.getrefcount(ref) <= 3
            del res
            assert sys.getrefcount(ref) <= 2
        elif input_type is bytes:
            assert sys.getrefcount(msg) <= 3

    def test_datetime_aware_ext(self):
        dec = rune.msgpack.Decoder(datetime.datetime)
        x = datetime.datetime.now(UTC)
        res = dec.decode(rune.msgpack.encode(x))
        assert x == res

    @pytest.mark.parametrize(
        "s",
        [
            "1234-01-02T03:04:05Z",
            "1234-01-02T03:04:05.123Z",
            "1234-01-02T03:04:05.123+00:00",
        ],
    )
    def test_decode_datetime_aware_str(self, s):
        sol = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        msg = rune.msgpack.encode(s)
        res = rune.msgpack.decode(msg, type=datetime.datetime)
        assert sol == res

    @pytest.mark.parametrize(
        "s",
        [
            "1234-01-02T03:04:05",
            "1234-01-02T03:04:05.123",
            "1234-01-02T03:04:05.123456",
        ],
    )
    def test_decode_datetime_naive(self, s):
        sol = datetime.datetime.fromisoformat(s)
        msg = rune.msgpack.encode(s)
        res = rune.msgpack.decode(msg, type=datetime.datetime)
        assert sol == res

    def test_datetime_unexpected_type(self):
        self.check_unexpected_type(datetime.datetime, 1, "Expected `datetime`")
        self.check_unexpected_type(
            datetime.datetime, rune.msgpack.Ext(1, b"test"), "Expected `datetime`"
        )

    def test_datetime_invalid(self):
        msg = rune.msgpack.encode(rune.msgpack.Ext(-1, b"\x01\x02\x03"))
        with pytest.raises(rune.ValidationError, match="Invalid MessagePack timestamp"):
            rune.msgpack.decode(msg, type=datetime.datetime)

    @pytest.mark.parametrize("size", SIZES)
    def test_list_lengths(self, size):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(list)
        x = list(range(size))
        res = dec.decode(enc.encode(x))
        assert res == x

    @pytest.mark.parametrize("typ", [list, list, list[Any]])
    def test_list_any(self, typ):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(typ)
        x = [1, "two", b"three"]
        res = dec.decode(enc.encode(x))
        assert res == x
        with pytest.raises(rune.ValidationError, match="Expected `array`"):
            dec.decode(enc.encode(1))

    def test_list_typed(self):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(list[int])
        x = [1, 2, 3]
        res = dec.decode(enc.encode(x))
        assert res == x
        with pytest.raises(
            rune.ValidationError,
            match=r"Expected `int`, got `str` - at `\$\[2\]`",
        ):
            dec.decode(enc.encode([1, 2, "three"]))

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("typ", [set, frozenset])
    def test_set_lengths(self, size, typ):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(typ)
        x = typ(range(size))
        res = dec.decode(enc.encode(x))
        assert res == x
        assert isinstance(res, typ)

    @pytest.mark.parametrize(
        "typ",
        [
            set,
            set[Any],
            set,
            set[Any],
            frozenset,
            frozenset[Any],
            frozenset,
            frozenset[Any],
        ],
    )
    def test_set_any(self, typ):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(typ)
        real_type = getattr(typ, "__origin__", typ)
        x = {1, "two", b"three"}
        res = dec.decode(enc.encode(x))
        assert res == x
        assert type(res) is real_type
        with pytest.raises(rune.ValidationError, match="Expected `array`"):
            dec.decode(enc.encode(1))

    @pytest.mark.parametrize("typ", [set, frozenset, set, frozenset])
    def test_set_typed(self, typ):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(typ[int])
        x = {1, 2, 3}
        res = dec.decode(enc.encode(x))
        assert res == x
        assert type(res) is getattr(typ, "__origin__", typ)
        with pytest.raises(
            rune.ValidationError,
            match=r"Expected `int`, got `str` - at `\$\[2\]`",
        ):
            dec.decode(enc.encode([1, 2, "three"]))

    @pytest.mark.parametrize("size", SIZES)
    def test_vartuple_lengths(self, size):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(tuple)
        x = tuple(f"x{i}x" for i in range(size))
        res = dec.decode(enc.encode(x))
        assert res == x
        if res:
            assert sys.getrefcount(res[0]) <= 3  # 1 tuple, 1 index, 1 func call

    @pytest.mark.parametrize("typ", [tuple, tuple[Any, ...], tuple, tuple[Any, ...]])
    def test_vartuple_any(self, typ):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(typ)
        x = (1, "two", b"three")
        res = dec.decode(enc.encode(x))
        assert res == x
        with pytest.raises(rune.ValidationError, match="Expected `array`, got `int`"):
            dec.decode(enc.encode(1))

    def test_vartuple_typed(self):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(tuple[int, ...])
        x = (1, 2, 3)
        res = dec.decode(enc.encode(x))
        assert res == x
        with pytest.raises(
            rune.ValidationError,
            match=r"Expected `int`, got `str` - at `\$\[2\]`",
        ):
            dec.decode(enc.encode((1, 2, "three")))

    def test_fixtuple_any(self):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(tuple[Any, Any, Any])
        x = (1, "two", b"three")
        res = dec.decode(enc.encode(x))
        assert res == x
        with pytest.raises(rune.ValidationError, match="Expected `array`, got `int`"):
            dec.decode(enc.encode(1))
        with pytest.raises(rune.ValidationError, match="Expected `array` of length 3, got 2"):
            dec.decode(enc.encode((1, 2)))

    def test_fixtuple_typed(self):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(tuple[int, str, bytes])
        x = (1, "two", b"three")
        res = dec.decode(enc.encode(x))
        assert res == x
        with pytest.raises(rune.ValidationError, match="Expected `bytes`"):
            dec.decode(enc.encode((1, "two", "three")))
        with pytest.raises(rune.ValidationError, match="Expected `array` of length 3, got 2"):
            dec.decode(enc.encode((1, 2)))

    @pytest.mark.parametrize("size", SIZES)
    def test_dict_lengths(self, size):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(dict)
        x = {i: i for i in range(size)}
        res = dec.decode(enc.encode(x))
        assert res == x

    @pytest.mark.parametrize("typ", [dict, dict, dict[Any, Any]])
    def test_dict_any_any(self, typ):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(typ)
        x = {1: "one", "two": 2, b"three": 3.0}
        res = dec.decode(enc.encode(x))
        assert res == x
        with pytest.raises(rune.ValidationError, match=r"Expected `object`, got `int`"):
            dec.decode(enc.encode(1))

    def test_dict_any_val(self):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(dict[str, Any])
        x = {"a": 1, "b": "two", "c": b"three"}
        res = dec.decode(enc.encode(x))
        assert res == x
        with pytest.raises(
            rune.ValidationError,
            match=r"Expected `str`, got `int` - at `key` in `\$`",
        ):
            dec.decode(enc.encode({1: 2}))

    def test_dict_any_key(self):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(dict[Any, str])
        x = {1: "a", "two": "b", b"three": "c"}
        res = dec.decode(enc.encode(x))
        assert res == x
        with pytest.raises(
            rune.ValidationError, match=r"Expected `str`, got `int` - at `\$\[...\]`"
        ):
            dec.decode(enc.encode({1: 2}))

    def test_dict_str_subclass_key(self):
        class mystr(str):
            pass

        msg1 = rune.msgpack.encode({mystr("test"): 1})
        msg2 = rune.msgpack.encode({"test": 1})
        assert msg1 == msg2

    def test_dict_typed(self):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(dict[str, int])
        x = {"a": 1, "b": 2}
        res = dec.decode(enc.encode(x))
        assert res == x
        with pytest.raises(
            rune.ValidationError,
            match=r"Expected `str`, got `int` - at `key` in `\$`",
        ):
            dec.decode(enc.encode({1: 2}))
        with pytest.raises(
            rune.ValidationError, match=r"Expected `int`, got `str` - at `\$\[...\]`"
        ):
            dec.decode(enc.encode({"a": "two"}))

    def test_dict_typed_non_str_key(self):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(dict[int, int])
        x = {0: 1, 2: 3}
        res = dec.decode(enc.encode(x))
        assert res == x
        with pytest.raises(
            rune.ValidationError,
            match=r"Expected `int`, got `str` - at `key` in `\$`",
        ):
            dec.decode(enc.encode({"bad": 2}))

    def test_enum(self):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(FruitStr)

        a = enc.encode(FruitStr.APPLE)
        assert enc.encode("apple") == a
        assert dec.decode(a) == FruitStr.APPLE

        with pytest.raises(rune.DecodeError, match="truncated"):
            dec.decode(a[:-2])

        with pytest.raises(rune.ValidationError, match="Invalid enum value 'MISSING'"):
            dec.decode(enc.encode("MISSING"))

        with pytest.raises(
            rune.ValidationError,
            match=r"Invalid enum value 'MISSING' - at `\$\[0\]`",
        ):
            rune.msgpack.decode(enc.encode(["MISSING"]), type=list[FruitStr])

        with pytest.raises(rune.ValidationError):
            dec.decode(enc.encode(1))

    def test_int_enum(self):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(FruitInt)

        a = enc.encode(FruitInt.APPLE)
        assert enc.encode(1) == a
        assert dec.decode(a) == FruitInt.APPLE

        with pytest.raises(rune.DecodeError, match="truncated"):
            dec.decode(a[:-2])

        with pytest.raises(rune.ValidationError, match="Invalid enum value 1000"):
            dec.decode(enc.encode(1000))

        with pytest.raises(rune.ValidationError, match=r"Invalid enum value 1000 - at `\$\[0\]`"):
            rune.msgpack.decode(enc.encode([1000]), type=list[FruitInt])

        with pytest.raises(rune.ValidationError):
            dec.decode(enc.encode("INVALID"))

    def test_str_literal(self):
        literal = Literal["one", "two"]
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(literal)

        assert dec.decode(enc.encode("one")) == "one"

        with pytest.raises(rune.ValidationError, match="Invalid enum value 'MISSING'"):
            dec.decode(enc.encode("MISSING"))

        with pytest.raises(
            rune.ValidationError,
            match=r"Invalid enum value 'MISSING' - at `\$\[0\]`",
        ):
            rune.msgpack.decode(enc.encode(["MISSING"]), type=list[literal])

    def test_int_literal(self):
        literal = Literal[1, 2, 3]
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(literal)

        assert dec.decode(enc.encode(1)) == 1

        with pytest.raises(rune.ValidationError, match="Invalid enum value 1000"):
            dec.decode(enc.encode(1000))

        with pytest.raises(rune.ValidationError, match=r"Invalid enum value 1000 - at `\$\[0\]`"):
            rune.msgpack.decode(enc.encode([1000]), type=list[literal])

    @pytest.mark.parametrize(
        "typ, value",
        [
            (bool, False),
            (bool, True),
            (int, 1),
            (float, 2.5),
            (str, "a"),
            (bytes, b"a"),
            (bytearray, bytearray(b"a")),
            (FruitInt, FruitInt.APPLE),
            (FruitStr, FruitStr.APPLE),
            (Person, Person("harry", "potter", 13)),
            (list, [1]),
            (set, {1}),
            (tuple, (1, 2)),
            (tuple[int, int], (1, 2)),
            (dict, {1: 2}),
            (datetime.datetime, datetime.datetime.now(UTC)),
        ],
    )
    @pytest.mark.parametrize(
        "optional_builder",
        [
            lambda typ: Optional[typ],  # noqa
            lambda typ: typ | None,
        ],
    )
    def test_optional(self, typ, value, optional_builder):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(optional_builder(typ))

        s = enc.encode(value)
        s2 = enc.encode(None)
        assert dec.decode(s) == value
        assert dec.decode(s2) is None

        dec = rune.msgpack.Decoder(typ)
        with pytest.raises(rune.ValidationError):
            dec.decode(s2)

    @pytest.mark.parametrize(
        "typ, value",
        [
            (list[int | None], [1, None]),
            (list[int | None], [1, None]),
            (tuple[int | None, int], (None, 1)),
            (set[int | None], {1, None}),
            (frozenset[int | None], frozenset({1, None})),
            (dict[str, int | None], {"a": 1, "b": None}),
            (dict[str | None, int], {"a": 1, None: 2}),
        ],
    )
    def test_optional_nested(self, typ, value):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(typ)

        s = enc.encode(value)
        assert dec.decode(s) == value

    @pytest.mark.parametrize(
        "types, vals",
        [
            ([int, float], [1, 2.5]),
            (
                [float, rune.msgpack.Ext, int, str],
                [1.5, rune.msgpack.Ext(1, b"two"), 1, "two"],
            ),
            ([bool, None, float, str], [True, None, 1.5, "test"]),
        ],
    )
    def test_old_union(self, types, vals):
        dec = rune.msgpack.Decoder(list[tuple(types)])
        s = rune.msgpack.encode(vals)
        res = dec.decode(s)
        assert res == vals
        for t, v in zip(types, res, strict=False):
            if t is not None:
                t = getattr(t, "__origin__", t)
                assert type(v) == t  # noqa

    @pytest.mark.parametrize(
        "typ, vals",
        [
            (int | float, [1, 2.5]),
            (
                float | rune.msgpack.Ext | int | str,
                [1.5, rune.msgpack.Ext(1, b"two"), 1, "two"],
            ),
            (bool | float | str | None, [True, None, 1.5, "test"]),
        ],
    )
    def test_union(self, typ, vals):
        dec = rune.msgpack.Decoder(list[typ])
        s = rune.msgpack.encode(vals)
        res = dec.decode(s)
        assert res == vals
        for v in res:
            assert isinstance(v, typ)

    @pytest.mark.parametrize(
        "types, vals",
        [
            (
                [PersonArray, FruitInt, FruitStr, dict[int, str]],
                [PERSON_AA, FruitInt.APPLE, FruitStr.BANANA, {1: "two"}],
            ),
            (
                [Person, FruitInt, FruitStr, tuple[int, ...]],
                [PERSON, FruitInt.APPLE, FruitStr.BANANA, (1, 2, 3)],
            ),
            (
                [Person, FruitInt, FruitStr, list[int]],
                [PERSON, FruitInt.APPLE, FruitStr.BANANA, [1, 2, 3]],
            ),
            (
                [Person, FruitInt, FruitStr, set[int]],
                [PERSON, FruitInt.APPLE, FruitStr.BANANA, {1, 2, 3}],
            ),
            (
                [Person, FruitInt, FruitStr, tuple[int, str, float]],
                [PERSON, FruitInt.APPLE, FruitStr.BANANA, (1, "two", 3.5)],
            ),
            (
                [dict[int, str], FruitInt, FruitStr, tuple[int, ...]],
                [{1: "two"}, FruitInt.APPLE, FruitStr.BANANA, (1, 2, 3)],
            ),
            (
                [dict[int, str], FruitInt, FruitStr, list[int]],
                [{1: "two"}, FruitInt.APPLE, FruitStr.BANANA, [1, 2, 3]],
            ),
            (
                [dict[int, str], FruitInt, FruitStr, set[int]],
                [{1: "two"}, FruitInt.APPLE, FruitStr.BANANA, {1, 2, 3}],
            ),
            (
                [dict[int, str], FruitInt, FruitStr, tuple[int, str, float]],
                [{1: "two"}, FruitInt.APPLE, FruitStr.BANANA, (1, "two", 3.5)],
            ),
        ],
    )
    def test_compound_type_unions(self, types, vals):
        typ_vals = list(zip(types, vals, strict=False))

        for N in range(2, len(typ_vals)):
            for typ_vals_subset in itertools.combinations(typ_vals, N):
                types, vals = zip(*typ_vals_subset, strict=False)
                vals = list(vals)
                dec = rune.msgpack.Decoder(list[types])
                s = rune.msgpack.encode(vals)
                res = dec.decode(s)
                assert res == vals
                for t, v in zip(types, res, strict=False):
                    t = getattr(t, "__origin__", t)
                    assert type(v) == t  # noqa

    def test_union_error(self):
        msg = rune.msgpack.encode(1)
        with pytest.raises(rune.ValidationError, match="Expected `bool | string`, got `int`"):  # noqa
            rune.msgpack.decode(msg, type=bool | str)

    def test_decoding_error_no_struct_toplevel(self):
        b = rune.msgpack.Encoder().encode([{"a": 1}])
        dec = rune.msgpack.Decoder(list[dict[str, str]])
        with pytest.raises(
            rune.ValidationError,
            match=r"Expected `str`, got `int` - at `\$\[0\]\[...\]`",
        ):
            dec.decode(b)


class TestExt:
    @pytest.mark.parametrize("data", [b"test", bytearray(b"test"), memoryview(b"test")])
    def test_init(self, data):
        x = rune.msgpack.Ext(1, data)
        assert x.code == 1
        assert x.data == data

    def test_init_bad_arguments(self):
        with pytest.raises(TypeError, match="Ext takes no keyword arguments"):
            rune.msgpack.Ext(code=1, data=b"two")

        with pytest.raises(TypeError, match="expected 2 arguments, got 3"):
            rune.msgpack.Ext(1, b"two", 3)

        with pytest.raises(TypeError, match="expected 2 arguments, got 0"):
            rune.msgpack.Ext()

    def test_compare(self):
        x = rune.msgpack.Ext(1, b"two")
        x2 = rune.msgpack.Ext(1, b"two")
        x3 = rune.msgpack.Ext(1, b"three")
        x4 = rune.msgpack.Ext(2, b"two")
        assert x == x2
        assert not (x != x2)
        assert x != x3
        assert not (x == x3)
        assert x != x4
        assert not (x == x4)

        assert not (x == 1)
        assert x != 1
        with pytest.raises(TypeError):
            x >= x2  # noqa

    @pytest.mark.parametrize("code", [-128, -2, 0, 2, 127])
    def test_code_roundtrip(self, code):
        ext = rune.msgpack.Ext(code, b"")
        assert ext.code == code
        ext2 = rune.msgpack.decode(rune.msgpack.encode(ext))
        assert ext2.code == code

    @pytest.mark.parametrize("code", [-129, 128, 2**65])
    def test_code_out_of_range(self, code):
        with pytest.raises(ValueError):
            rune.msgpack.Ext(code, b"bad")

    def test_data_wrong_type(self):
        with pytest.raises(TypeError):
            rune.msgpack.Ext(1, 2)

    def test_code_wrong_type(self):
        with pytest.raises(TypeError):
            rune.msgpack.Ext(b"bad", b"bad")

    def test_immutable(self):
        x = rune.msgpack.Ext(1, b"two")
        with pytest.raises(AttributeError):
            x.code = 2

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickleable(self, protocol):
        x = rune.msgpack.Ext(1, b"two")
        x2 = pickle.loads(pickle.dumps(x, protocol=protocol))
        assert x2.code == 1
        assert x2.data == b"two"

    @pytest.mark.parametrize("size", sorted({0, 1, 2, 4, 8, 16, *SIZES}))
    def test_serialize_compatibility(self, size):
        msgpack = pytest.importorskip("msgpack")
        data = b"x" * size
        code = 5

        rune_bytes = rune.msgpack.encode(rune.msgpack.Ext(code, data))
        msgpack_bytes = msgpack.dumps(msgpack.ExtType(code, data))
        assert rune_bytes == msgpack_bytes

    @pytest.mark.parametrize("typ", [bytearray, memoryview])
    def test_serialize_other_types(self, typ):
        buf = b"test"
        a = rune.msgpack.encode(rune.msgpack.Ext(1, buf))

        ext = rune.msgpack.Ext(1, typ(buf))
        assert isinstance(ext.data, typ)
        b = rune.msgpack.encode(ext)
        assert a == b

        decoded = rune.msgpack.decode(b)
        assert isinstance(decoded.data, bytes)
        assert decoded.data == buf

    def test_other_buffers(self):
        buf = array.array("i", [1, 2, 3, 4])
        out = rune.msgpack.decode(
            rune.msgpack.encode(rune.msgpack.Ext(1, buf)),
        )

        assert isinstance(out.data, bytes)
        assert array.array("i", out.data) == buf

    @pytest.mark.parametrize("size", sorted({0, 1, 2, 4, 8, 16, *SIZES}))
    def test_roundtrip(self, size):
        data = b"x" * size
        code = 5

        buf = rune.msgpack.encode(rune.msgpack.Ext(code, data))
        out = rune.msgpack.decode(buf)
        assert out.code == code
        assert out.data == data

    @pytest.mark.parametrize("size", sorted({0, 1, 2, 4, 8, 16, *SIZES}))
    def test_roundtrip_typed_decoder(self, size):
        dec = rune.msgpack.Decoder(rune.msgpack.Ext)

        ext = rune.msgpack.Ext(5, b"x" * size)
        buf = rune.msgpack.encode(ext)
        out = dec.decode(buf)
        assert out == ext

    @pytest.mark.parametrize("type", [None, rune.msgpack.Ext])
    def test_decode_ext_no_reference_leak(self, type):
        """https://github.com/msgspec/msgspec/issues/1108"""
        buf = rune.msgpack.encode(rune.msgpack.Ext(5, b"ext-leak-test-payload"))
        kwargs = {} if type is None else {"type": type}
        ext = rune.msgpack.decode(buf, **kwargs)
        data = ext.data
        del ext
        assert sys.getrefcount(data) <= 2  # `data` + getrefcount argument

    def test_typed_decoder_skips_ext_hook(self):
        def ext_hook(code, data):
            raise AssertionError("shouldn't ever get called")

        msg = [None, rune.msgpack.Ext(1, b"test")]
        dec = rune.msgpack.Decoder(list[rune.msgpack.Ext | None])
        buf = rune.msgpack.encode(msg)
        out = dec.decode(buf)
        assert out == msg

    def test_ext_typed_decoder_error(self):
        dec = rune.msgpack.Decoder(rune.msgpack.Ext)
        with pytest.raises(rune.ValidationError, match="Expected `ext`, got `int`"):
            assert dec.decode(rune.msgpack.encode(1))

    @pytest.mark.parametrize("use_function", [True, False])
    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_decoder_ext_hook(self, use_function, protocol):
        obj = {"x": range(10)}

        def enc_hook(x):
            return rune.msgpack.Ext(5, pickle.dumps(x, protocol=protocol))

        def ext_hook(code, buf):
            assert isinstance(buf, memoryview)
            assert code == 5
            return pickle.loads(buf)

        msg = rune.msgpack.encode(obj, enc_hook=enc_hook)
        if use_function:
            out = rune.msgpack.decode(msg, ext_hook=ext_hook)
        else:
            dec = rune.msgpack.Decoder(ext_hook=ext_hook)
            out = dec.decode(msg)
        assert out == obj

    def test_decoder_ext_hook_bad_signature(self):
        msg = rune.msgpack.encode(range(5), enc_hook=lambda x: rune.msgpack.Ext(1, b"test"))
        with pytest.raises(TypeError):
            rune.msgpack.decode(msg, ext_hook=lambda: None)

    def test_decoder_ext_hook_raises(self):
        class CustomError(Exception):
            pass

        def ext_hook(code, buf):
            raise CustomError

        msg = rune.msgpack.encode(range(5), enc_hook=lambda x: rune.msgpack.Ext(1, b"test"))
        with pytest.raises(CustomError):
            rune.msgpack.decode(msg, ext_hook=ext_hook)


class TestTimestampExt:
    def check(self, dt, msg):
        assert rune.msgpack.encode(dt) == msg
        assert rune.msgpack.decode(msg) == dt

    def test_timestamp32_lower(self):
        dt = datetime.datetime.fromtimestamp(0, UTC)
        msg = b"\xd6\xff\x00\x00\x00\x00"
        self.check(dt, msg)

    def test_timestamp32_upper(self):
        dt = datetime.datetime.fromtimestamp(2**32 - 1, UTC)
        msg = b"\xd6\xff\xff\xff\xff\xff"
        self.check(dt, msg)

    def test_timestamp64_lower(self):
        dt = datetime.datetime.fromtimestamp(1e-6, UTC)
        msg = b"\xd7\xff\x00\x00\x0f\xa0\x00\x00\x00\x00"
        self.check(dt, msg)

    def test_timestamp64_upper(self):
        dt = datetime.datetime.fromtimestamp(2**34, UTC) - datetime.timedelta(microseconds=1)
        msg = b"\xd7\xff\xeek\x18c\xff\xff\xff\xff"
        self.check(dt, msg)

    def test_timestamp96_lower(self):
        dt = datetime.datetime.fromtimestamp(-1e-6, UTC)
        msg = b"\xc7\x0c\xff;\x9a\xc6\x18\xff\xff\xff\xff\xff\xff\xff\xff"
        self.check(dt, msg)

    def test_timestamp96_upper(self):
        dt = datetime.datetime.fromtimestamp(2**34, UTC)
        msg = b"\xc7\x0c\xff\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00"
        self.check(dt, msg)

    @pytest.mark.parametrize(
        "msg, secs, micros",
        [
            (b"\xd7\xff\x00\x00\x07\xd0\x00\x00\x00\x00", 0, 1),
            (b"\xd7\xff\x00\x00\x07\xcc\x00\x00\x00\x00", 0, 0),
            (b"\xd7\xff\xeek 0\x00\x00\x00\x00", 1, 0),
            (b"\xd7\xff\xeek ,\x00\x00\x00\x00", 0, 999999),
        ],
    )
    def test_timestamp_rounds_nanos(self, msg, secs, micros):
        res = rune.msgpack.decode(msg)
        assert res.second == secs
        assert res.microsecond == micros


class CommonTypeTestBase:
    """Test rune untyped encode/decode"""

    def test_none(self):
        self.check(None)

    @pytest.mark.parametrize("x", [False, True])
    def test_bool(self, x):
        self.check(x)

    @pytest.mark.parametrize("x", INTS)
    def test_int(self, x):
        self.check(x)

    @pytest.mark.parametrize("x", FLOATS)
    def test_float(self, x):
        self.check(x)

    @pytest.mark.parametrize("size", SIZES)
    def test_str(self, size):
        self.check(" " * size)

    @pytest.mark.parametrize("size", SIZES)
    def test_bytes(self, size):
        self.check(b" " * size)

    @pytest.mark.parametrize("size", SIZES)
    def test_dict(self, size):
        self.check({str(i): i for i in range(size)})

    @pytest.mark.parametrize("size", SIZES)
    def test_list(self, size):
        self.check(list(range(size)))


class TestDecodeArrayTypeUsesTupleIfHashableRequired:
    def test_decode_tuple_dict_keys_as_tuples(self):
        orig = {(1, 2): [1, 2, [3, 4]], (1, (2, 3)): [4, 5, 6], tuple(range(32)): []}
        data = rune.msgpack.encode(orig)
        out = rune.msgpack.decode(data)
        assert orig == out

    @pytest.mark.parametrize(
        "typ",
        [
            dict[tuple[int, tuple[int, int]], list[int]],
            dict[tuple[int, tuple[int, ...]], Any],
            dict[tuple, list[int]],
            dict[tuple[Any, ...], Any],
            dict[tuple[Any, Any], Any],
        ],
    )
    def test_decode_dict_key_status_forwarded_through_typed_tuples(self, typ):
        orig = {(1, (2, 3)): [1, 2, 3]}
        data = rune.msgpack.encode(orig)
        out = rune.msgpack.Decoder(typ).decode(data)
        assert orig == out

    def test_decode_tuple_set_keys_as_tuples(self):
        orig = {(1, 2), (3, (4, 5)), 6}
        data = rune.msgpack.encode(orig)
        out = rune.msgpack.decode(data, type=set)
        assert orig == out

    def test_decode_hashable_struct_in_key(self):
        class Test(rune.Struct):
            data: list[int]

            def __hash__(self):
                return hash(tuple(self.data))

        orig = {(1, Test([1, 2])): [1, 2]}
        data = rune.msgpack.encode(orig)
        out = rune.msgpack.Decoder(dict[tuple[int, Test], list[int]]).decode(data)
        assert orig == out


class TestUntypedDecoder(CommonTypeTestBase):
    """Check the untyped deserializer works for common types"""

    def check(self, x):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder()
        assert_eq(dec.decode(enc.encode(x)), x)


class TestCompatibility(CommonTypeTestBase):
    """Test compatibility with the existing python msgpack library"""

    def check(self, x):
        msgpack = pytest.importorskip("msgpack")

        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder()

        assert_eq(dec.decode(msgpack.dumps(x)), x)
        assert_eq(msgpack.loads(enc.encode(x)), x)


class TestStruct:
    @pytest.mark.parametrize("tag", [False, "Test", 123])
    def test_encode_empty_struct(self, tag):
        class Test(rune.Struct, tag=tag):
            pass

        if tag:
            msg = {"type": tag}
        else:
            msg = {}
        s = rune.msgpack.encode(Test())
        s2 = rune.msgpack.encode(msg)
        assert s == s2

    @pytest.mark.parametrize("tag", [False, "Test", 123])
    def test_encode_one_field_struct(self, tag):
        class Test(rune.Struct, tag=tag):
            a: int

        if tag:
            msg = {"type": tag, "a": 1}
        else:
            msg = {"a": 1}
        s = rune.msgpack.encode(Test(a=1))
        s2 = rune.msgpack.encode(msg)
        assert s == s2

    @pytest.mark.parametrize("tag", [False, "Test", 123])
    def test_encode_two_field_struct(self, tag):
        class Test(rune.Struct, tag=tag):
            a: int
            b: str

        if tag:
            msg = {"type": tag, "a": 1, "b": "two"}
        else:
            msg = {"a": 1, "b": "two"}
        s = rune.msgpack.encode(Test(a=1, b="two"))
        s2 = rune.msgpack.encode(msg)
        assert s == s2

    def test_decode_struct(self):
        dec = rune.msgpack.Decoder(Person)
        msg = rune.msgpack.encode({"first": "harry", "last": "potter", "age": 13, "prefect": False})
        x = dec.decode(msg)
        assert x == Person("harry", "potter", 13, False)

        with pytest.raises(rune.ValidationError, match="Expected `object`, got `int`"):
            dec.decode(rune.msgpack.encode(1))

    def test_decode_struct_field_wrong_type(self):
        dec = rune.msgpack.Decoder(Person)

        msg = rune.msgpack.encode({"first": "harry", "last": "potter", "age": "bad"})
        with pytest.raises(rune.ValidationError, match=r"Expected `int`, got `str` - at `\$.age`"):
            dec.decode(msg)

    def test_decode_struct_missing_fields(self):
        bad = rune.msgpack.encode({"first": "harry", "last": "potter"})
        with pytest.raises(rune.ValidationError, match="Object missing required field `age`"):
            rune.msgpack.decode(bad, type=Person)

        bad = rune.msgpack.encode({})
        with pytest.raises(rune.ValidationError, match="Object missing required field `first`"):
            rune.msgpack.decode(bad, type=Person)

        bad = rune.msgpack.encode([{"first": "harry", "last": "potter"}])
        with pytest.raises(
            rune.ValidationError,
            match=r"Object missing required field `age` - at `\$\[0\]`",
        ):
            rune.msgpack.decode(bad, type=list[Person])

    @pytest.mark.parametrize(
        "extra",
        [
            None,
            False,
            True,
            1,
            2.0,
            "three",
            b"four",
            [1, 2],
            {3: 4},
            rune.msgpack.Ext(1, b"12345"),
            rune.msgpack.Ext(1, b""),
        ],
    )
    def test_decode_struct_ignore_extra_fields(self, extra):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(Person)

        a = enc.encode(
            {
                "extra1": extra,
                "first": "harry",
                "extra2": extra,
                "last": "potter",
                "age": 13,
                "extra3": extra,
            }
        )
        res = dec.decode(a)
        assert res == Person("harry", "potter", 13)

    def test_decode_struct_defaults_missing_fields(self):
        dec = rune.msgpack.Decoder(Person)

        a = rune.msgpack.encode({"first": "harry", "last": "potter", "age": 13})
        res = dec.decode(a)
        assert res == Person("harry", "potter", 13)
        assert res.prefect is False

    @pytest.mark.parametrize("array_like", [False, True])
    def test_struct_gc_maybe_untracked_on_decode(self, array_like):
        class Test(rune.Struct, array_like=array_like):
            x: Any
            y: Any
            z: tuple = ()

        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(list[Test])

        ts = [
            Test(1, 2),
            Test(3, "hello"),
            Test([], []),
            Test({}, {}),
            Test(None, None, ()),
        ]
        a, b, c, d, e = dec.decode(enc.encode(ts))
        assert not gc.is_tracked(a)
        assert not gc.is_tracked(b)
        assert gc.is_tracked(c)
        assert gc.is_tracked(d)
        assert not gc.is_tracked(e)

    @pytest.mark.parametrize("array_like", [False, True])
    def test_struct_gc_false_always_untracked_on_decode(self, array_like):
        class Test(rune.Struct, array_like=array_like, gc=False):
            x: Any
            y: Any

        dec = rune.msgpack.Decoder(list[Test])

        ts = [
            Test(1, 2),
            Test([], []),
            Test({}, {}),
        ]
        for obj in dec.decode(rune.msgpack.encode(ts)):
            assert not gc.is_tracked(obj)

    def test_struct_recursive_definition(self):
        enc = rune.msgpack.Encoder()
        dec = rune.msgpack.Decoder(Node)

        x = Node(Node(Node(), Node(Node())))
        s = enc.encode(x)
        res = dec.decode(s)
        assert res == x

    @pytest.mark.parametrize("tag", ["Test", 123, -123])
    def test_decode_tagged_struct(self, tag):
        class Test(rune.Struct, tag=tag):
            a: int
            b: int

        dec = rune.msgpack.Decoder(Test)

        # Test decode with and without tag
        for msg in [
            {"a": 1, "b": 2},
            {"type": tag, "a": 1, "b": 2},
            {"a": 1, "type": tag, "b": 2},
        ]:
            res = dec.decode(rune.msgpack.encode(msg))
            assert res == Test(1, 2)

        # Tag incorrect type
        with pytest.raises(rune.ValidationError) as rec:
            dec.decode(rune.msgpack.encode({"type": 123.456}))
        assert f"Expected `{type(tag).__name__}`" in str(rec.value)
        assert "`$.type`" in str(rec.value)

        # Tag incorrect value
        bad = -3 if isinstance(tag, int) else "bad"
        with pytest.raises(rune.ValidationError) as rec:
            dec.decode(rune.msgpack.encode({"type": bad}))
        assert f"Invalid value {bad!r}" in str(rec.value)
        assert "`$.type`" in str(rec.value)

    @pytest.mark.parametrize("tag", [i for i in INTS if -(2**63) <= i < 2**63])
    def test_decode_tagged_struct_int_ranges(self, tag):
        class Test(rune.Struct, tag=tag):
            a: int
            b: int

        dec = rune.msgpack.Decoder(Test)
        t = Test(1, 2)
        assert dec.decode(rune.msgpack.encode(t))

    def test_decode_tagged_struct_int_tag_uint64_always_invalid(self):
        """Uint64 values aren't currently valid tag values, but we still want
        to raise a good error message."""

        class Test(rune.Struct, tag=123):
            pass

        with pytest.raises(rune.ValidationError) as rec:
            rune.msgpack.decode(rune.msgpack.encode({"type": 2**64 - 1}), type=Test)
        assert f"Invalid value {2**64 - 1}" in str(rec.value)
        assert "`$.type`" in str(rec.value)

    @pytest.mark.parametrize("tag", ["Test", 123, -123])
    def test_decode_tagged_empty_struct(self, tag):
        class Test(rune.Struct, tag=tag):
            pass

        dec = rune.msgpack.Decoder(Test)

        # Tag missing
        res = dec.decode(rune.msgpack.encode({}))
        assert res == Test()

        # Tag present
        res = dec.decode(rune.msgpack.encode({"type": tag}))
        assert res == Test()


class TestStructArray:
    @pytest.mark.parametrize("tag", [False, "Test", 123])
    def test_encode_empty_struct(self, tag):
        class Test(rune.Struct, array_like=True, tag=tag):
            pass

        s = rune.msgpack.encode(Test())
        if tag:
            msg = [tag]
        else:
            msg = []
        s2 = rune.msgpack.encode(msg)
        assert s == s2

    @pytest.mark.parametrize("tag", [False, "Test", 123])
    def test_encode_one_field_struct(self, tag):
        class Test(rune.Struct, array_like=True, tag=tag):
            a: int

        s = rune.msgpack.encode(Test(a=1))
        if tag:
            msg = [tag, 1]
        else:
            msg = [1]
        s2 = rune.msgpack.encode(msg)
        assert s == s2

    @pytest.mark.parametrize("tag", [False, "Test", 123])
    def test_encode_two_field_struct(self, tag):
        class Test(rune.Struct, array_like=True, tag=tag):
            a: int
            b: str

        s = rune.msgpack.encode(Test(a=1, b="two"))
        if tag:
            msg = [tag, 1, "two"]
        else:
            msg = [1, "two"]
        s2 = rune.msgpack.encode(msg)
        assert s == s2

    def test_struct_array_like(self):
        dec = rune.msgpack.Decoder(PersonArray)

        x = PersonArray(first="harry", last="potter", age=13)
        a = rune.msgpack.encode(x)
        assert rune.msgpack.encode(("harry", "potter", 13, False)) == a
        assert dec.decode(a) == x

        with pytest.raises(rune.ValidationError, match="Expected `array`, got `int`"):
            dec.decode(b"1")

        # Wrong field type
        bad = rune.msgpack.encode(("harry", "potter", "thirteen"))
        with pytest.raises(rune.ValidationError, match=r"Expected `int`, got `str` - at `\$\[2\]`"):
            dec.decode(bad)

        # Missing fields
        bad = rune.msgpack.encode(("harry", "potter"))
        with pytest.raises(
            rune.ValidationError,
            match="Expected `array` of at least length 3, got 2",
        ):
            dec.decode(bad)

        bad = rune.msgpack.encode(())
        with pytest.raises(
            rune.ValidationError,
            match="Expected `array` of at least length 3, got 0",
        ):
            dec.decode(bad)

        # Extra fields ignored
        dec2 = rune.msgpack.Decoder(list[PersonArray])
        msg = rune.msgpack.encode(
            [
                ("harry", "potter", 13, False, 1, 2, 3, 4),
                ("ron", "weasley", 13, False, 5, 6),
            ]
        )
        res = dec2.decode(msg)
        assert res == [
            PersonArray("harry", "potter", 13),
            PersonArray("ron", "weasley", 13),
        ]

        # Defaults applied
        res = dec.decode(rune.msgpack.encode(("harry", "potter", 13)))
        assert res == PersonArray("harry", "potter", 13)
        assert res.prefect is False

    def test_struct_map_and_array_like_messages_cant_mix(self):
        array_msg = rune.msgpack.encode(("harry", "potter", 13))
        map_msg = rune.msgpack.encode({"first": "harry", "last": "potter", "age": 13})
        sol = Person("harry", "potter", 13)
        array_sol = PersonArray("harry", "potter", 13)

        dec = rune.msgpack.Decoder(Person)
        array_dec = rune.msgpack.Decoder(PersonArray)

        assert array_dec.decode(array_msg) == array_sol
        assert dec.decode(map_msg) == sol
        with pytest.raises(rune.ValidationError, match="Expected `object`, got `array`"):
            dec.decode(array_msg)
        with pytest.raises(rune.ValidationError, match="Expected `array`, got `object`"):
            array_dec.decode(map_msg)

    @pytest.mark.parametrize("tag", ["Test", -123, 123])
    def test_decode_tagged_struct(self, tag):
        class Test(rune.Struct, tag=tag, array_like=True):
            a: int
            b: int
            c: int = 0

        dec = rune.msgpack.Decoder(Test)

        # Decode with tag
        res = dec.decode(rune.msgpack.encode([tag, 1, 2]))
        assert res == Test(1, 2)
        res = dec.decode(rune.msgpack.encode([tag, 1, 2, 3]))
        assert res == Test(1, 2, 3)

        # Trailing fields ignored
        res = dec.decode(rune.msgpack.encode([tag, 1, 2, 3, 4]))
        assert res == Test(1, 2, 3)

        # Missing required field errors
        with pytest.raises(rune.ValidationError) as rec:
            dec.decode(rune.msgpack.encode([tag, 1]))
        assert "Expected `array` of at least length 3, got 2" in str(rec.value)

        # Tag missing
        with pytest.raises(rune.ValidationError) as rec:
            dec.decode(rune.msgpack.encode([]))
        assert "Expected `array` of at least length 3, got 0" in str(rec.value)

        # Tag incorrect type
        with pytest.raises(rune.ValidationError) as rec:
            dec.decode(rune.msgpack.encode([123.456, 2, 3]))
        assert f"Expected `{type(tag).__name__}`" in str(rec.value)
        assert "`$[0]`" in str(rec.value)

        # Tag incorrect value
        bad = -3 if isinstance(tag, int) else "bad"
        with pytest.raises(rune.ValidationError) as rec:
            dec.decode(rune.msgpack.encode([bad, 1, 2]))
        assert f"Invalid value {bad!r}" in str(rec.value)
        assert "`$[0]`" in str(rec.value)

        # Field incorrect type correct index
        with pytest.raises(rune.ValidationError) as rec:
            dec.decode(rune.msgpack.encode([tag, "a", 2]))
        assert "Expected `int`, got `str`" in str(rec.value)
        assert "`$[1]`" in str(rec.value)

    @pytest.mark.parametrize("tag", ["Test", 123, -123])
    def test_decode_tagged_empty_struct(self, tag):
        class Test(rune.Struct, tag=tag, array_like=True):
            pass

        dec = rune.msgpack.Decoder(Test)

        # Decode with tag
        res = dec.decode(rune.msgpack.encode([tag, 1, 2]))
        assert res == Test()

        # Tag missing
        with pytest.raises(rune.ValidationError) as rec:
            dec.decode(rune.msgpack.encode([]))
        assert "Expected `array` of at least length 1, got 0" in str(rec.value)


class TestRaw:
    def test_encode_raw(self):
        b = rune.msgpack.encode({"x": 1})
        r = rune.Raw(b)
        assert rune.msgpack.encode(r) == b
        assert rune.msgpack.encode({"y": r}) == rune.msgpack.encode({"y": {"x": 1}})

    def test_decode_raw_field(self):
        class Test(rune.Struct):
            x: int
            y: rune.Raw

        s = rune.msgpack.encode({"x": 1, "y": [1, 2, 3]})
        res = rune.msgpack.decode(s, type=Test)
        assert res.x == 1
        assert bytes(res.y) == rune.msgpack.encode([1, 2, 3])

    def test_decode_raw_optional_field(self):
        default = rune.Raw()

        class Test(rune.Struct):
            x: int
            y: rune.Raw = default

        s = rune.msgpack.encode({"x": 1, "y": [1, 2, 3]})
        res = rune.msgpack.decode(s, type=Test)
        assert res.x == 1
        assert bytes(res.y) == rune.msgpack.encode([1, 2, 3])

        s = rune.msgpack.encode({"x": 1})
        res = rune.msgpack.decode(s, type=Test)
        assert res.x == 1
        assert res.y is default

    def test_decode_raw_malformed_data(self):
        class Test(rune.Struct):
            x: int
            y: rune.Raw

        s = rune.msgpack.encode({"x": 1, "y": [1, 2]})[:3]
        with pytest.raises(rune.DecodeError):
            rune.msgpack.decode(s, type=Test)

    def test_decode_raw_is_view(self):
        s = rune.msgpack.encode({"x": 1})
        r = rune.msgpack.decode(s, type=rune.Raw)
        assert bytes(r) == s
        assert r.copy() is not r  # actual copy indicates a view

    def test_raw_in_union_works_but_doesnt_change_anything(self):
        class Test(rune.Struct):
            x: int | str | rune.Raw

        s = rune.msgpack.encode({"x": 1})
        r = rune.msgpack.decode(s, type=Test)
        assert r == Test(1)

    def test_raw_can_be_mixed_with_custom_type(self):
        class Test(rune.Struct):
            x: Custom | rune.Raw

        def dec_hook(typ, obj):
            assert typ is Custom
            return typ(*obj)

        s = rune.msgpack.encode({"x": [1, 2]})
        res = rune.msgpack.decode(s, type=Test, dec_hook=dec_hook)
        assert res == Test(Custom(1, 2))
