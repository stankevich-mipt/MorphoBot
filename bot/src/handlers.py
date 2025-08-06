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
Module that provides bot interaction logic.

Implements asyncio coroutines that are executed upon
/start and /help command and the event of a user
sending a photo to a bot

"""

import io

from telegram import Update
from telegram.ext import ContextTypes


async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle /start command logic.

    Attributes:
        update (telegram.Update): the telegram update event
        containing the message info
        ctx: callback context
    """
    await update.message.reply_text(
        "👋 Hi! Send me a selfie and I'll do something funny with it."
    )

help_cmd = start_cmd  # alias


async def photo_received(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Route user photo to inference microservice.

    Attributes:
        update (telegram.Update): the telegram update event
        containing the message info
        ctx: callback context

    """
    if not update.message.photo:
        return

    await update.message.reply_chat_action("upload_photo")
    # Grab highest-resolution photo object
    photo = update.message.photo[-1]
    file_info = await ctx.bot.get_file(photo.file_id)
    file_bytes = await file_info.download_as_bytearray()

    file_bytes = io.BytesIO(file_bytes)

    await update.message.reply_photo(
        file_bytes,
        caption="Here you go!  🚀"
    )

    # TODO: POST bytes to image-processor service, get swapped_img
    # swapped_img = await call_processor(file_bytes)
