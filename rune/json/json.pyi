import builtins
from collections.abc import Callable, Iterable
from typing import Any, Generic, Literal, TypeVar, overload

T = TypeVar("T")

def encode(
        obj: Any,
        /,
        *,
        enc_hook: Callable[[Any], Any] | None = None,
        order: Literal["deterministic", "sorted"] | None = None,
) -> bytes: ...
def decode(
        buf: bytes,
        /,
        *,
        type: Any = ...,
        dec_hook: Callable[[Any, Any], Any] | None = None,
        strict: bool = True,
) -> Any: ...
@overload
def format(buf: str, /, *, indent: int = 2) -> str: ...
@overload
def format(buf: bytes | bytearray | memoryview, /, *, indent: int = 2) -> bytes: ...

class Decoder(Generic[T]):
    dec_hook: Callable[[Any, Any], Any] | None

    @overload
    def __init__(
            self: Decoder[Any],
            type: Any = ...,
            *,
            dec_hook: Callable[[Any, Any], Any] | None = None,
            strict: bool = True,
            float_hook: Callable[[str], Any] | None = None,
    ) -> None: ...
    @overload
    def __init__(
            self,
            type: builtins.type[T] | Any,
            *,
            dec_hook: Callable[[Any, Any], Any] | None = None,
            strict: bool = True,
            float_hook: Callable[[str], Any] | None = None,
    ) -> None: ...
    @property
    def type(self) -> Any: ...
    @property
    def float_hook(self) -> Callable[[str], Any] | None: ...
    @property
    def strict(self) -> bool: ...
    def __class_getitem__(cls, item: Any) -> Any: ...
    def decode(self, buf: bytes, /) -> T: ...
    def decode_lines(self, buf: bytes, /) -> list[T]: ...

class Encoder:
    def __init__(
            self,
            *,
            enc_hook: Callable[[Any], Any] | None = None,
            decimal_format: Literal["string", "number"] | Callable[[Any], Any] | None = None,
            order: Literal["deterministic", "sorted"] | None = None,
            uuid_format: Literal["canonical", "hex"] | None = None,
    ) -> None: ...
    @property
    def enc_hook(self) -> Callable[[Any], Any] | None: ...
    @property
    def decimal_format(self) -> Literal["string", "number"] | Callable[[Any], Any]: ...
    @property
    def order(self) -> Literal["deterministic", "sorted"] | None: ...
    @property
    def uuid_format(self) -> Literal["canonical", "hex"]: ...
    def encode(self, obj: Any) -> bytes: ...
    def encode_lines(self, items: Iterable[Any]) -> bytes: ...
    def encode_into(self, obj: Any, buffer: bytearray, offset: int = 0) -> None: ...
