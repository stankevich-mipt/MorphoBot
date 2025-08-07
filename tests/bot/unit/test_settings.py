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
Test correctness of settings instance creation.

This scripts checks whether the pydantic-settings
instance created with bot.src.settings.py correctly picks
up options from environmental variables

"""


def test_settings_load_from_env(test_settings):
    """Assert correctness of test_settings attributes."""
    assert test_settings.bot_token == "test-token-42"
    assert test_settings.poll_mode is True
    assert test_settings.webhook_base == "https://test-webhook.local"
    assert test_settings.secret_token == "test-secret-token-42"
