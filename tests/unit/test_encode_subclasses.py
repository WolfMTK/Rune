from typing import Any

import pytest
import rune


@pytest.fixture(params=["json", "msgpack"])
def proto(request: pytest.FixtureRequest) -> Any:
    if request.param == "json":
        return rune.json
    if request.param == "msgpack":
        return rune.msgpack
    raise AssertionError(f"Library {request.param} not found")


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
