import inspect
from typing import Any, ClassVar


class Struct:
    __struct_fields__: ClassVar[tuple[tuple[str, Any], ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        hints = inspect.get_annotations(cls, eval_str=True)
        cls.__struct_fields__ = tuple(hints.items())

    def __init__(self, **kwargs: Any) -> None:
        for name, _ in self.__struct_fields__:
            if name in kwargs:
                value = kwargs.pop(name)
            elif hasattr(type(self), name):
                value = getattr(type(self), name)
            else:
                raise TypeError(f"Missing required field `{name}`")
            setattr(self, name, value)
        if kwargs:
            unknown = ", ".join(f"`{k}`" for k in kwargs)
            raise TypeError(f"Unexpected field(s): {unknown}")

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return all(
            getattr(self, name) == getattr(other, name) for name, _ in self.__struct_fields__
        )

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={getattr(self, name)!r}" for name, _ in self.__struct_fields__)
        return f"{type(self).__name__}({fields})"
