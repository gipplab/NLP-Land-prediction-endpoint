"""Test the status route."""
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from nlp_land_prediction_endpoint import __version__
from nlp_land_prediction_endpoint.app import app


@pytest.fixture
def client() -> Generator:
    """Get the test client for tests and reuse it.

    Yields:
        Generator: Yields the test client as input argument for each test.
    """
    with TestClient(app) as tc:
        yield tc


@pytest.fixture
def endpoint() -> str:
    """Get the endpoint for tests.

    Returns:
        str: The endpoint including current version.
    """
    return f"/api/v{__version__.split('.')[0]}/status"


def test_backend_status(client: TestClient, endpoint: str) -> None:
    """Test the backend status.

    Args:
        client (TestClient): The current test client.
        endpoint (str): Endpoint prefix.
    """
    response = client.get(endpoint)
    assert response.status_code == 200
    assert response.json() == {
        "message": f"NLP-Land-prediction-endpoint online at version {__version__}."
    }
