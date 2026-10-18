import pytest

from .utils import PY312, max_call_depth

print(PY312)


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
