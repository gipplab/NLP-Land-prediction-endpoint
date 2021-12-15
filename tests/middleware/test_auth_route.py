"""Test the auth route."""
import os
from typing import Any, Generator

import pytest
import requests
from fastapi.testclient import TestClient
from requests.models import Response

from nlp_land_prediction_endpoint import __version__
from nlp_land_prediction_endpoint.app import app
from nlp_land_prediction_endpoint.middleware.auth import UserModel, create_token


@pytest.fixture
def client() -> Generator:
    """Get the test client for tests and reuse it.

    Yields:
        Generator: Yields the test client as input argument for each test.
    """
    with TestClient(app) as tc:
        yield tc


@pytest.fixture
def refresh_endpoint() -> str:
    """Get the login_endpoint for tests.

    Returns:
        str: The login_endpoint including current version.
    """
    return f"/api/v{__version__.split('.')[0]}/auth/refresh"


@pytest.fixture
def login_endpoint() -> str:
    """Get the login_endpoint for tests.

    Returns:
        str: The login_endpoint including current version.
    """
    return f"/api/v{__version__.split('.')[0]}/auth/login"


@pytest.fixture
def dummy_user() -> UserModel:
    """Create a dummy user.

    Returns:
        str: The login_endpoint including current version.
    """
    example = UserModel.Config.schema_extra.get("example")
    return UserModel(**example)


@pytest.fixture
def dummy_token() -> str:
    """Create a dummy user.

    Returns:
        str: The login_endpoint including current version.
    """
    example = UserModel.Config.schema_extra.get("example")
    user = UserModel(**example)
    token = create_token(user)
    return token


@pytest.fixture
def mock_post(monkeypatch: Any) -> None:
    """Mocks the request.post in auth.py"""

    def mock_post(*agrs, **kwargs):
        # type: (*str, **int) -> Response
        response = Response()
        response.status_code = 200
        response._content = b"{\"email\": \"test@test.de\"}"
        return response

    monkeypatch.setattr(requests, "post", mock_post)


@pytest.fixture
def mock_post_failure(monkeypatch: Any) -> None:
    """Mocks the request.post in auth.py"""

    def mock_post(*agrs, **kwargs):
        # type: (*str, **int) -> Response
        response = Response()
        response.status_code = 401
        response._content = b"{\"message\": \"error\"}"
        return response

    monkeypatch.setattr(requests, "post", mock_post)


def test_dev_auth(client: TestClient, login_endpoint: str, dummy_user: UserModel) -> None:
    """Test the local auth"""
    response = client.post(login_endpoint, json=dummy_user.dict())
    assert response.status_code == 200


def test_simulated_existing_failed_auth(client: TestClient, login_endpoint: str,
                                        dummy_user: UserModel, mock_post_failure: Any) -> None:
    os.environ["AUTH_LOGIN_PROVIDER"] = "http://127.0.0.1"
    os.environ["AUTH_TOKEN_ROUTE"] = login_endpoint
    response = client.post(login_endpoint, json=dummy_user.dict())
    assert response.status_code == 401


def test_simulated_existing_auth(
        client: TestClient, login_endpoint: str,
        dummy_user: UserModel,
        mock_post: Any) -> None:
    os.environ["AUTH_LOGIN_PROVIDER"] = "http://127.0.0.1"
    os.environ["AUTH_TOKEN_ROUTE"] = login_endpoint
    response = client.post(login_endpoint, json=dummy_user.dict())
    assert response.status_code == 200


def test_non_existing_auth(client: TestClient, login_endpoint: str, dummy_user: UserModel) -> None:
    os.environ["AUTH_LOGIN_PROVIDER"] = "http://127.0.0.1"
    os.environ["AUTH_TOKEN_ROUTE"] = login_endpoint
    response = client.post(login_endpoint, json=dummy_user.dict())
    assert response.status_code == 401


def test_refresh_forged(client: TestClient, refresh_endpoint: str, dummy_user: UserModel) -> None:
    response = client.post(refresh_endpoint, headers={"Authorization": "Bearer 123"})
    assert response.status_code == 401


def test_refresh(client: TestClient, refresh_endpoint: str, dummy_token: str) -> None:
    response = client.post(refresh_endpoint, headers={"Authorization": f"Bearer {dummy_token}"})
    assert response.status_code == 200