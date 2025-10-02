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
Configuration settings for Telegram bot derived from pydantic BaseSettings.

This script defines application-wide configuration options
that are loaded from .env file, providing type validation and defaults.
"""

import os

from pydantic import Field

from pydantic_settings import BaseSettings, SettingsConfigDict


# Get the absolute path to the directory containing this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path to the .env file in the same directory as this script
ENV_PATH = os.path.join(BASE_DIR, '.env')


class Settings(BaseSettings):
    """Application settings loaded from .env file.

    Attributes:
        bot_token (str): Secret token used to authenticate the bot
        with the Telegram Bot API. This token is required for all
        bot-to-Telegram communication.
        webhook_base (str, optional): The base HTTPS URL exposed
        for receiving updates from Telegram via webhooks. Must be
        a routable, publicly accessible address using HTTPS.
        Telegram will send POST requests to this URL (or its subpaths)
        when new messages or events occur.
        secret_token (str): Secret token included as part of the webhook
        URL path to authenticate incoming requests for Telegram.
        This token must be kept confidential, must be sufficiently
        random and unguessable to prevent unauthorized access.
        poll_mode (bool, default=True): Whether to use updates
        based on polling or webhook. Former is simpler and
        is recommended for development, as it does not
        require domain configuration and valid SSL certificates.
        Latter is more complex, but more performant, hence
        it will be used in production.

    Environment variables:
        - BOT_TOKEN: Telegram bot authentication token, as issued by BotFather.
        - WEBHOOK_BASE (optional): Public HTTPS endpoint
            (e.g., https://example.com/bot1234)
        - SECRET_TOKEN: Secret token for securing webhook endpoint path.
        - POLL_MODE: If set true, uses polling update mode.

    Configuration is loaded at startup. Fields can be overridden by setting
    the respective environment variables or by creating a .env file
    in the project root.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_PATH, env_file_encoding='utf-8')

    bot_token: str = Field(
        ..., json_schema_extra={"env", "BOT_TOKEN"})
    webhook_base: str | None = Field(
        default=None, json_schema_extra={"env", "WEBHOOK_BASE"})
    secret_token: str = Field(
        ..., json_schema_extra={"env", "SECRET_TOKEN"})
    poll_mode: bool = Field(
        default=False, json_schema_extra={"env", "POLL_MODE"})


settings = Settings()
