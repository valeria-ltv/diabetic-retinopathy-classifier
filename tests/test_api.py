import asyncio
import base64
import io
import time

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from src.api.main import app

active_calls = 0
max_active_calls = 0


def make_dummy_prediction() -> dict:
    """Return a realistic mock prediction."""
    probs = [0.70, 0.15, 0.10, 0.04, 0.01]
    return {
        "regression_pred": 0.42,
        "classification_probs": probs,
        "class_score": 0.42,
        "ensemble_score": 0.42,
        "predicted_class": 0,
    }


class MockPipeline:
    """Lightweight mock pipeline to simulate model inference without weights."""
    def predict_single(self, image_path):
        global active_calls, max_active_calls
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        # Small delay to simulate inference
        time.sleep(0.1)
        active_calls -= 1
        return make_dummy_prediction()


def mock_generate_explanations(image_path, pipeline):
    """Lightweight mock for Grad-CAM explanation generation."""
    global active_calls, max_active_calls
    active_calls += 1
    max_active_calls = max(max_active_calls, active_calls)
    # Small delay to simulate inference
    time.sleep(0.1)
    active_calls -= 1
    dummy_arr = np.zeros((10, 10, 3), dtype=np.uint8)
    return {
        "prediction": make_dummy_prediction(),
        "preprocessed": dummy_arr,
        "cam_clf": dummy_arr,
        "cam_reg": dummy_arr
    }


@pytest.fixture(autouse=True)
def setup_test_env():
    """Reset call counters and initialize app state without running app lifespan."""
    global active_calls, max_active_calls
    active_calls = 0
    max_active_calls = 0

    app.state.model_lock = asyncio.Lock()
    app.state.pipeline = MockPipeline()
    yield
    app.state.model_lock = None
    app.state.pipeline = None


def create_test_image_bytes():
    """Create an in-memory valid PNG image payload for upload tests."""
    img = Image.new("RGB", (50, 50), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- Basic API tests ---


def test_health_check():
    """Verify /health returns 200 and reports the pipeline as loaded."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_upload_invalid_file():
    """Verify non-image uploads are rejected with 400."""
    client = TestClient(app)
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid image file" in response.json()["detail"]


def test_predict_file_too_large(monkeypatch):
    """Verify files exceeding MAX_FILE_SIZE are rejected with 413."""
    # Temporarily lower the upload limit so a small payload triggers 413
    import src.api.routes as routes_module
    monkeypatch.setattr(routes_module, "MAX_FILE_SIZE", 5)

    client = TestClient(app)
    large_data = b"1234567890"
    response = client.post(
        "/predict",
        files={"file": ("large.png", large_data, "image/png")}
    )
    assert response.status_code == 413
    assert "File is too large" in response.json()["detail"]


def test_internal_server_error_handler(monkeypatch):
    """Verify unexpected inference exceptions return a 500 response."""
    def broken_predict(image_path):
        raise RuntimeError("Unexpected CUDA/PyTorch failure")

    monkeypatch.setattr(app.state.pipeline, "predict_single", broken_predict)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/predict",
        files={"file": ("test.png", create_test_image_bytes(), "image/png")}
    )
    assert response.status_code == 500


# --- Concurrency and lock tests ---


@pytest.mark.asyncio
async def test_predict_lock():
    """Verify concurrent /predict requests run one at a time under model_lock."""
    img_bytes = create_test_image_bytes()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Send 5 requests simultaneously
        tasks = [
            ac.post("/predict", files={"file": ("test.png", img_bytes, "image/png")})
            for _ in range(5)
        ]
        responses = await asyncio.gather(*tasks)

    for r in responses:
        assert r.status_code == 200

    # Ensure no two inference calls overlapped while holding the lock
    assert max_active_calls == 1


@pytest.mark.asyncio
async def test_explain_lock(monkeypatch):
    """Verify concurrent /explain requests run one at a time under model_lock and return PNGs."""
    import src.api.routes as routes_module
    monkeypatch.setattr(routes_module, "generate_explanations", mock_generate_explanations)

    img_bytes = create_test_image_bytes()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        tasks = [
            ac.post("/explain", files={"file": ("test.png", img_bytes, "image/png")})
            for _ in range(5)
        ]
        responses = await asyncio.gather(*tasks)

    for r in responses:
        assert r.status_code == 200
        data = r.json()
        for field in ("preprocessed", "classifier_cam", "regressor_cam"):
            assert base64.b64decode(data[field]).startswith(b"\x89PNG")

    assert max_active_calls == 1


@pytest.mark.asyncio
async def test_lock_released_after_error(monkeypatch):
    """Verify model_lock is released after a failed request so later requests succeed."""
    img_bytes = create_test_image_bytes()
    calls = 0

    def flaky_predict(image_path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("Boom on first request")
        return make_dummy_prediction()

    monkeypatch.setattr(app.state.pipeline, "predict_single", flaky_predict)

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r1 = await ac.post("/predict", files={"file": ("1.png", img_bytes, "image/png")})
        r2 = await ac.post("/predict", files={"file": ("2.png", img_bytes, "image/png")})

    assert r1.status_code == 500
    assert r2.status_code == 200
    assert calls == 2