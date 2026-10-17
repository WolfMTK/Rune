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
