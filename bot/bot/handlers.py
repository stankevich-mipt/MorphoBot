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

import aiohttp
import asyncio
import io

from telegram import Update
from telegram.constants import ChatAction
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


async def classify_photo(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    
    INFER_URL = "http://localhost:8000/classify"
    
    message = update.effective_message
    if not message or not message.photo:
        await message.reply_text("Please send a photo.")
        return
    
    photo = message.photo[-1]

    try: 
        await context.bot.send_chat_action(
            chat_id=message.chat_id, action=ChatAction.TYPING
        )

        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()

        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            form = aiohttp.FormData()
            form.add_field(
                "file", image_bytes,
                filename="photo.jpg", content_type="image/jpeg"
            )

            async with session.post(INFER_URL, data=form) as resp:
                if resp.status == 400:
                    data = await resp.json()
                    await message.reply_text(
                        f"Could not process: {data.get('error', 'bad_request')}")
                    return
                if resp.status >= 500:
                    await message.reply_text(
                        "Service is unavailable, please try again later"
                    )
                    return
                data = await resp.json()

        if "predicted_class" not in data:
            await message.reply_text("No prediction returned.")
            return
        
        label = data["predicted_class"]
        conf  = data.get("confidence")
        bbox  = data.get("bbox")
        parts = [f"Class: {label}"]
        
        if conf is not None:
            parts.append(f"Confidence: {conf:.3f}")
        if bbox is not None:
            parts.append(f"BBox: {tuple(bbox)}")
        await message.reply_text("\n".join(parts))

    except asyncio.TimeoutError:
        await message.reply_text("Processing timed out, please try again.")
    except Exception:
        await message.reply_text("Unexpected error during classification.")

