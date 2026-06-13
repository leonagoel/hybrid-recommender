from pathlib import Path


def test_backend_main_creates_single_fastapi_app():
    source = Path("backend/main.py").read_text()

    assert source.count("app = FastAPI(") == 1
    assert source.index("app = FastAPI(") < source.index(
        "app.include_router(recommend.router, prefix=\"/api\")"
    )
