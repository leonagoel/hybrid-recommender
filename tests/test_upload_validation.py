import pytest
from fastapi import HTTPException

from backend.main import _validate_upload_bytes


def test_upload_validation_rejects_empty_files():
    with pytest.raises(HTTPException) as exc:
        _validate_upload_bytes("products.csv", ".csv", b"")

    assert exc.value.status_code == 400
    assert exc.value.detail == "Uploaded file is empty."


def test_upload_validation_rejects_binary_files():
    with pytest.raises(HTTPException) as exc:
        _validate_upload_bytes("products.csv", ".csv", b"title\x00rating\nA,5")

    assert exc.value.status_code == 400
    assert exc.value.detail == "Uploaded file appears to be binary."


def test_upload_validation_rejects_json_named_as_csv():
    with pytest.raises(HTTPException) as exc:
        _validate_upload_bytes("products.csv", ".csv", b'{"title":"Alpha"}')

    assert exc.value.status_code == 400
    assert exc.value.detail == "CSV uploads must contain CSV content."


def test_upload_validation_rejects_csv_named_as_json():
    with pytest.raises(HTTPException) as exc:
        _validate_upload_bytes("products.json", ".json", b"title,rating\nAlpha,5")

    assert exc.value.status_code == 400
    assert exc.value.detail == "JSON uploads must contain JSON content."


def test_upload_validation_accepts_csv_and_json_shapes():
    _validate_upload_bytes("products.csv", ".csv", b"title,rating\nAlpha,5")
    _validate_upload_bytes("products.json", ".json", b'[{"title":"Alpha"}]')
