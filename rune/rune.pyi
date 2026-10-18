# exceptions.rs
class DecodeError:
    pass

class ValidationError:
    pass

# lib.rs
_struct_lookup_cache: dict[frozenset[type], dict[str | int, type]]
