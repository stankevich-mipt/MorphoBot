#    Copyright 2025, Stankevich Andrey, stankevich.as@phystech.edu

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Pytest configuration and fixtures for the Telegram bot test suite.

This module provides shared fixtures and setup routines to
support asynchronous testing of the Telegram bot microservice.
It manages application initialization, client creation, and
environment configuration for reliable, reusable test runs.

Fixtures:
- telegram app: Async fixture that yields the initialized
Telegram bot application
- async_client: Async HTTP client fixture for simulating
incoming requests to the FastAPI app
- test_settings: fixture for creating mock settings
that are used in further tests
"""


from bot.src.main import app, tg_app
from bot.src.settings import Settings

import pytest


@pytest.fixture(scope='session')
async def telegram_app():
    """Handle logic for tg bot startup/teardown."""
    await tg_app.initialize()
    yield tg_app
    await tg_app.shutdown()


@pytest.fixture(scope='session')
async def async_client(telegram_app):
    """Create an async client for FastAPI request mimicry."""
    from httpx import AsyncClient
    async with AsyncClient(app=app, base_url="http://test.com") as client:
        yield client


@pytest.fixture
def test_settings(monkeypatch):
    """Create a test settings mock."""
    monkeypatch.setenv("BOT_TOKEN", "test-token-42")
    monkeypatch.setenv("POLL_MODE", "true")
    monkeypatch.setenv("WEBHOOK_BASE", "https://test-webhook.local")
    monkeypatch.setenv("SECRET_TOKEN", "test-secret-token-42")
    return Settings()
