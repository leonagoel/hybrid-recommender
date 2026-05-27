import numpy as np
from backend.main import _json_scalar


def test_json_scalar_with_primitive_types():
    assert _json_scalar(5) == 5
    assert _json_scalar("test") == "test"
    assert _json_scalar(3.14) == 3.14
    assert _json_scalar(True) is True
    assert _json_scalar(None) is None


def test_json_scalar_with_numpy_types():
    np_int = np.int64(42)
    result_int = _json_scalar(np_int)
    assert result_int == 42
    assert isinstance(result_int, int)

    np_float = np.float32(1.23)
    result_float = _json_scalar(np_float)
    assert abs(result_float - 1.23) < 1e-6
    assert isinstance(result_float, float)


def test_json_scalar_with_complex_types():
    lst = [1, 2, 3]
    assert _json_scalar(lst) == lst

    dct = {"a": 1}
    assert _json_scalar(dct) == dct
