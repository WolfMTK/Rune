# exceptions.rs
class DecodeError:
    pass

class ValidationError:
    pass

# common/raw.rs
class Raw:
    def __init__(self, data: str | bytes | bytearray | memoryview, /) -> None: ...
    def __bytes__(self) -> bytes: ...
    def __len__(self) -> int: ...

# lib.rs
_struct_lookup_cache: dict[frozenset[type], dict[str | int, type]]
