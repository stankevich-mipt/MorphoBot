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
import os 

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes


# Configuration
CLASSIFY_URL = os.getenv("CLASSIFY_URL", "http://localhost:8000/classify")
TRANSLATE_URL = os.getenv("TRANSLATE_URL", "http://localhost:8001/translate")


async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle /start command logic.

    Attributes:
        update: Update event
        ctx: callback context
    """
    if not update.message:
        return 

    await update.message.reply_text(
        "👋 Hi! Send me a selfie and I'll do something funny with it."
    )   

help_cmd = start_cmd  # alias


async def photo_received(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Route user photo through classification and translation pipeline.
    
    First classifies the gender of the person in the photo, then uses
    that classification to call the translation service with the correct
    source_gender parameter.
    
    Attributes:
        update (telegram.Update): the telegram update event 
            containing the message info
        ctx: callback context
    """
    if not update.message:
        return    
    if not update.message.photo:
        await update.message.reply_text("❌ Please, provide a photo.")

    
    message = update.message
    photo = message.photo[-1]
    
    try:
        # Step 1: Download photo from Telegram
        await ctx.bot.send_chat_action(
            chat_id=message.chat_id, 
            action=ChatAction.TYPING
        )
        
        file = await ctx.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()
        
        # Step 2: Classify gender
        await message.reply_text("🔍 Detecting gender...")
        
        classify_timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=classify_timeout) as session:
            form = aiohttp.FormData()
            form.add_field(
                "file", 
                image_bytes,
                filename="photo.jpg", 
                content_type="image/jpeg"
            )
            
            async with session.post(CLASSIFY_URL, data=form) as resp:
                if resp.status == 400:
                    data = await resp.json()
                    await message.reply_text(
                        f"❌ Could not classify the photo: "
                        f"{data.get('error', 'bad_request')}"
                    )
                    return
                
                if resp.status >= 500:
                    await message.reply_text(
                        "⚠️ Classification service is unavailable. "
                        "Please try again later."
                    )
                    return
                
                if resp.status != 200:
                    await message.reply_text(
                        f"❌ Classification failed (status {resp.status})"
                    )
                    return
                
                data = await resp.json()
                if "predicted_class" not in data:
                    await message.reply_text("❌ No classification returned.")
                    return
                
                predicted_gender = data["predicted_class"]
                confidence = data.get("confidence", 0.0)
                
                # Validate gender classification
                if predicted_gender not in ["male", "female"]:
                    await message.reply_text(
                        f"❌ Unexpected classification: {predicted_gender}"
                    )
                    return
        
        # Step 3: Translate gender using classification result
        await ctx.bot.send_chat_action(
            chat_id=message.chat_id, 
            action=ChatAction.UPLOAD_PHOTO
        )
        await message.reply_text(
            f"✨ Detected {predicted_gender} (confidence: {confidence:.2%})\n"
            f"🔄 Translating gender..."
        )
        
        translate_timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=translate_timeout) as session:
            form = aiohttp.FormData()
            form.add_field(
                "file", 
                image_bytes,
                filename="photo.jpg", 
                content_type="image/jpeg"
            )

            form.add_field("source_gender", predicted_gender)
            form.add_field("include_debug", "false")
            
            async with session.post(TRANSLATE_URL, data=form) as resp:
                if resp.status == 400:
                    await message.reply_text(
                        "❌ Could not process the photo for translation. "
                        "Please make sure it contains a clear face."
                    )
                    return
                
                if resp.status == 413:
                    await message.reply_text(
                        "❌ Photo is too large. Maximum size is 10 MB."
                    )
                    return
                
                if resp.status >= 500:
                    await message.reply_text(
                        "⚠️ Translation service is temporarily unavailable. "
                        "Please try again later."
                    )
                    return
                
                if resp.status != 200:
                    await message.reply_text(
                        f"❌ Translation failed (status {resp.status})"
                    )
                    return
                

                translated_bytes = await resp.read()
                
                source_gender = resp.headers.get("X-Source-Gender", predicted_gender)
                target_gender = resp.headers.get("X-Target-Gender", "unknown")
                processing_time = resp.headers.get("X-Processing-Time-Ms", "N/A")
                
                caption = (
                    f"✨ Gender translation complete!\n"
                    f"🔍 Detected: {predicted_gender.capitalize()} "
                    f"({confidence:.1%} confidence)\n"
                    f"🔄 Translated: {source_gender.capitalize()} → "
                    f"{target_gender.capitalize()}\n"
                    f"⏱️ Processing time: {processing_time} ms"
                )
                
                translated_img = io.BytesIO(translated_bytes)
                await message.reply_photo(
                    translated_img,
                    caption=caption
                )
    
    except asyncio.TimeoutError:
        await message.reply_text(
            "⏰ Processing timed out. The service might be overloaded. "
            "Please try again."
        )
    except aiohttp.ClientError:
        await message.reply_text(
            "❌ Could not connect to the processing services. "
            "Please check if they are running."
        )
    except Exception as e:
        await message.reply_text(
            "❌ An unexpected error occurred during processing."
        )
