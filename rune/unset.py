from typing import ClassVar


class UnsetType:
    __slots__ = ()

    _instance: ClassVar["UnsetType | None"] = None

    def __new__(cls) -> "UnsetType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "UNSET"


UNSET = UnsetType()
