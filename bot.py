import asyncio
import logging
import os
from collections import defaultdict

from dotenv import load_dotenv
from ollama import AsyncClient
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ollama_client import describe_image
from prompts import DEFAULT_PRESET, PRESETS, resolve_preset

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

user_state: dict[int, dict] = defaultdict(
    lambda: {"mode": DEFAULT_PRESET, "last_image": None}
)


# --- Helpers ---


def preset_keyboard(exclude: str | None = None) -> InlineKeyboardMarkup:
    """Build an inline keyboard of preset buttons, 2 per row."""
    buttons = []
    for key, preset in PRESETS.items():
        if key == exclude:
            continue
        buttons.append(InlineKeyboardButton(preset["name"], callback_data=f"redo:{key}"))
    rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
    return InlineKeyboardMarkup(rows)


def mode_keyboard() -> InlineKeyboardMarkup:
    """Build an inline keyboard for /mode selection, 2 per row."""
    buttons = [
        InlineKeyboardButton(p["name"], callback_data=f"mode:{k}")
        for k, p in PRESETS.items()
    ]
    rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
    return InlineKeyboardMarkup(rows)


async def send_typing_loop(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> asyncio.Task:
    """Start a background task that sends typing indicator every 4 seconds."""

    async def _loop():
        try:
            while True:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass

    return asyncio.create_task(_loop())


async def send_long_message(
    update: Update,
    text: str,
    parse_mode: str | None = None,
    reply_markup: InlineKeyboardMarkup | None = None,
):
    """Send a message, splitting if >4096 chars. Keyboard goes on last chunk."""
    max_len = 4096
    if len(text) <= max_len:
        try:
            await update.message.reply_text(
                text, parse_mode=parse_mode, reply_markup=reply_markup
            )
        except Exception:
            # Fallback to plain text if markdown parsing fails
            await update.message.reply_text(text, reply_markup=reply_markup)
        return

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Find a good split point
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    for i, chunk in enumerate(chunks):
        markup = reply_markup if i == len(chunks) - 1 else None
        try:
            await update.message.reply_text(chunk, parse_mode=parse_mode, reply_markup=markup)
        except Exception:
            await update.message.reply_text(chunk, reply_markup=markup)


# --- Handlers ---


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome! Send me an image and I'll describe it using AI.\n\n"
        "By default I use the *Detailed* preset. You can:\n"
        "- Add a caption like `tags` or `sd` to override the mode for one image\n"
        "- Use /mode to change your default preset\n"
        "- Use /help to see all presets and commands",
        parse_mode=ParseMode.MARKDOWN,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    current = user_state[uid]["mode"]
    current_name = PRESETS[current]["name"]

    lines = ["*Available presets:*\n"]
    for key, preset in PRESETS.items():
        marker = " (current)" if key == current else ""
        lines.append(f"  `{key}` \u2014 {preset['name']}{marker}")

    lines.append(f"\n*Current default:* {current_name}")
    lines.append("\n*Commands:*")
    lines.append("  /mode \u2014 Change default preset")
    lines.append("  /help \u2014 Show this message")
    lines.append("\n*Tip:* Send an image with a caption (e.g. `sd` or `tags`) to use that preset once.")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    current = user_state[uid]["mode"]
    await update.message.reply_text(
        f"Current mode: *{PRESETS[current]['name']}*\n\nSelect a new default:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=mode_keyboard(),
    )


async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    ollama_client: AsyncClient = context.bot_data["ollama_client"]
    model_name: str = context.bot_data["model_name"]

    # Get image bytes
    if update.message.photo:
        file = await update.message.photo[-1].get_file()
    elif update.message.document:
        file = await update.message.document.get_file()
    else:
        return

    image_bytes = await file.download_as_bytearray()
    image_bytes = bytes(image_bytes)

    # Determine preset
    caption = update.message.caption
    if caption:
        preset_key = resolve_preset(caption)
    else:
        preset_key = user_state[uid]["mode"]

    preset = PRESETS[preset_key]

    # Store for re-describe
    user_state[uid]["last_image"] = image_bytes

    # Typing indicator
    typing_task = await send_typing_loop(update.effective_chat.id, context)

    try:
        result = await describe_image(
            ollama_client, model_name, preset["prompt"], image_bytes
        )
    except RuntimeError as e:
        typing_task.cancel()
        await update.message.reply_text(f"Error: {e}")
        return
    except Exception as e:
        typing_task.cancel()
        logger.error("describe_image failed: %s", e, exc_info=True)
        await update.message.reply_text("Something went wrong while processing your image.")
        return

    typing_task.cancel()

    parse_mode = ParseMode.MARKDOWN if preset["markdown"] else None
    keyboard = preset_keyboard(exclude=preset_key)

    await send_long_message(update, result, parse_mode=parse_mode, reply_markup=keyboard)


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id
    data = query.data

    if data.startswith("mode:"):
        key = data.split(":", 1)[1]
        if key not in PRESETS:
            return
        user_state[uid]["mode"] = key
        await query.edit_message_text(
            f"Default mode set to *{PRESETS[key]['name']}*.",
            parse_mode=ParseMode.MARKDOWN,
        )

    elif data.startswith("redo:"):
        key = data.split(":", 1)[1]
        if key not in PRESETS:
            return

        image_bytes = user_state[uid].get("last_image")
        if not image_bytes:
            await query.edit_message_text("No image stored. Send a new image first.")
            return

        preset = PRESETS[key]
        ollama_client: AsyncClient = context.bot_data["ollama_client"]
        model_name: str = context.bot_data["model_name"]

        chat_id = query.message.chat_id
        typing_task = asyncio.create_task(_typing_loop(chat_id, context))

        try:
            result = await describe_image(
                ollama_client, model_name, preset["prompt"], image_bytes
            )
        except RuntimeError as e:
            typing_task.cancel()
            await context.bot.send_message(chat_id=chat_id, text=f"Error: {e}")
            return
        except Exception as e:
            typing_task.cancel()
            logger.error("redo describe_image failed: %s", e, exc_info=True)
            await context.bot.send_message(
                chat_id=chat_id, text="Something went wrong while re-describing."
            )
            return

        typing_task.cancel()

        parse_mode = ParseMode.MARKDOWN if preset["markdown"] else None
        keyboard = preset_keyboard(exclude=key)

        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=result,
                parse_mode=parse_mode,
                reply_markup=keyboard,
            )
        except Exception:
            await context.bot.send_message(
                chat_id=chat_id, text=result, reply_markup=keyboard
            )


async def _typing_loop(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    try:
        while True:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


def main():
    load_dotenv()

    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        raise SystemExit("BOT_TOKEN not set in environment")

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model_name = os.getenv("MODEL_NAME", "qwen3-vl:4b")

    app = Application.builder().token(bot_token).build()

    app.bot_data["ollama_client"] = AsyncClient(host=ollama_url)
    app.bot_data["model_name"] = model_name

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("mode", mode_command))
    app.add_handler(
        MessageHandler(filters.PHOTO | filters.Document.IMAGE, image_handler)
    )
    app.add_handler(CallbackQueryHandler(callback_handler))

    logger.info("Bot starting with model %s at %s", model_name, ollama_url)
    app.run_polling()


if __name__ == "__main__":
    main()
