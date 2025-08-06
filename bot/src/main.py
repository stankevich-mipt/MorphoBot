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
Telegram Bot with FastAPI Integration.

A Telegram bot application that acts
as a fronted microservice to process user photos,
implemented with python-telegram-bot library
and FastAPI web framework. The bot supports both polling and webhook
deployment modes for flexible hosting options.

Architecture:
    - FastAPI web server for webhook handling and potential API endpoints
    - python-telegram-bot for Telegram Bot API integration
    - Dual deployment modes: polling (development) and webhook (production)
    - Asynchronous operation with concurrent update processing

Bot Handlers:
    - /start: Welcome command handler
    - /help: Help command handler
    - Photo messages: Process uploaded photos for gender-swapping

Deployment Modes:
    1. Polling Mode: Bot actively polls Telegram servers for updates
    2. Webhook Mode: Telegram sends updates to configured webhook URL

Action Sequence:
    1. Initialize FastAPI app and Telegram bot application
    2. Register command and message handlers for bot functionality
    3. On startup: Configure either polling or webhook mode based on settings
    4. Process incoming updates through registered handlers
    5. Handle photo messages via photo_received handler for processing
    6. Gracefully shutdown bot application on termination

Security:
    - Webhook endpoint protected with secret token validation
    - Header-based authentication for incoming webhook requests

Dependencies:
    - fastapi: Web framework for webhook handling
    - python-telegram-bot: Telegram Bot API wrapper
    - Custom modules: handlers (bot logic), settings (configuration)

Usage:
    Run with polling: Set poll_mode=True in settings
    Run with webhook: Set poll_mode=False and configure webhook_base URL
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from bot.src.handlers import help_cmd, photo_received, start_cmd
from bot.src.settings import settings

from fastapi import FastAPI, HTTPException, Header, Request


from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
)


logger = logging.getLogger(__name__)

tg_app: Application = (
    ApplicationBuilder()
    .token(settings.bot_token)
    .concurrent_updates(True)
    .build()
)

# Register handlers
tg_app.add_handler(CommandHandler("start", start_cmd))
tg_app.add_handler(CommandHandler("help", help_cmd))
tg_app.add_handler(
    MessageHandler(filters.PHOTO & ~filters.COMMAND, photo_received)
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager for app startup/shutdown."""
    # ---------- START-UP ---------- #
    if settings.poll_mode:
        asyncio.create_task(run_polling())
    else:
        asyncio.create_task(set_webhook())
        await tg_app.start()

    yield

    # ---------- SHUTDOWN ---------- #
    await tg_app.stop()


app = FastAPI(title="MorphoBot", lifespan=lifespan)


# ---------- POLLING MODE ---------- #
async def run_polling():
    """Asyncio coroutine that launches the updater in polling mode."""
    await tg_app.initialize()
    await tg_app.start()
    logger.info("Polling started ➜ Ctrl-C to stop")
    await tg_app.updater.start_polling()
    await tg_app.updater.idle()


# ---------- WEBHOOK MODE ---------- #
WEBHOOK_PATH = f"/webhook/{settings.secret_token}"


@app.post(WEBHOOK_PATH)
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Annotated[str | None, Header()] = None,
) -> dict:
    """Handle the update if the token is valid.

    Asyncio coroutine that checks whether the request header
    secret token matches the one provided in settings.
    If it does, the request json body is passed through
    process_update method, otherwise, a HTTPException is thrown.

    Attributes:
        request (fastapi.Request): the request sent
        to the exposed webhook point
        x_telegram_bot_api_secret_token

    Returns:
        dict: {"ok": True}
    Raises:
        HTTPException: 403 Forbidden if secret token does not match
    """
    # Verify header
    if x_telegram_bot_api_secret_token != settings.secret_token:
        raise HTTPException(status_code=403, detail="Forbidden")
    update = await request.json()
    await tg_app.process_update(update)
    return {"ok": True}


async def set_webhook():
    """Coroutine that sets the webhook updates."""
    await tg_app.initialize()
    await tg_app.bot.set_webhook(
        url=f"{settings.webhook_base}{WEBHOOK_PATH}",
        secret_token=settings.secret_token,  # header validation[14]
        drop_pending_updates=True,
    )
    logger.info("Webhook registered")
