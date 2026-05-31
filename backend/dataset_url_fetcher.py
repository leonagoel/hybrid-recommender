"""Fetch remote dataset files over HTTP(S) with SSRF protections."""

from __future__ import annotations

import io
import os
import urllib.error
import urllib.request
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request

from backend.url_validation import (
    DISALLOWED_URL_MESSAGE,
    UrlValidationError,
    validate_public_url,
)

DEFAULT_FETCH_TIMEOUT_SECONDS = 15


class _RedirectValidator(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _max_fetch_bytes() -> int:
    return int(os.environ.get("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))


def fetch_dataset_from_url(url: str, max_bytes: int | None = None) -> bytes:
    """
    Download dataset bytes from a public http(s) URL after SSRF validation.
    """
    validate_public_url(url)

    limit = max_bytes if max_bytes is not None else _max_fetch_bytes()
    request = Request(
        url.strip(),
        headers={"User-Agent": "HybridRecommender/1.0"},
        method="GET",
    )
    opener = urllib.request.build_opener(_RedirectValidator())

    try:
        with opener.open(request, timeout=DEFAULT_FETCH_TIMEOUT_SECONDS) as response:
            chunks: list[bytes] = []
            total = 0
            while True:
                block = response.read(min(65536, limit - total + 1))
                if not block:
                    break
                total += len(block)
                if total > limit:
                    raise UrlValidationError(DISALLOWED_URL_MESSAGE)
                chunks.append(block)
    except UrlValidationError:
        raise
    except (urllib.error.URLError, OSError, ValueError):
        raise UrlValidationError(DISALLOWED_URL_MESSAGE) from None

    return b"".join(chunks)


def filename_from_url(url: str) -> str:
    """Infer a dataset filename from the URL path."""
    path = urlparse(url).path.rstrip("/")
    name = path.rsplit("/", 1)[-1] if path else ""
    if name.lower().endswith((".csv", ".json")):
        return name
    return "data.csv"


def dataset_buffer_from_url(url: str, max_bytes: int | None = None) -> tuple[io.BytesIO, str]:
    """Return (buffer, filename) for a validated remote dataset URL."""
    contents = fetch_dataset_from_url(url, max_bytes=max_bytes)
    return io.BytesIO(contents), filename_from_url(url)
