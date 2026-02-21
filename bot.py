import asyncio
import json
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

from ollama_client import chat_with_image, describe_image, modify_description
from prompts import CHAT_SYSTEM_PROMPT, CONVERSATION_PROMPT, DEFAULT_PRESET, PRESETS, resolve_preset

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

STATE_FILE = "user_state.json"

user_state: dict[int, dict] = defaultdict(
    lambda: {
        "mode": DEFAULT_PRESET,
        "last_image": None,
        "last_output": None,
        "conversation_mode": False,
        "chat_history": [],
        "chat_mode": False,
    }
)


def load_user_state():
    """Load persisted user state from disk."""
    logger.info("Looking for state file at: %s", os.path.abspath(STATE_FILE))
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                for uid, state in data.items():
                    user_state[int(uid)] = state
            logger.info("Loaded user state for %d users", len(data))
        except Exception as e:
            logger.warning("Failed to load user state: %s", e)
    else:
        logger.info("No existing state file found, starting fresh")


def save_user_state(uid: int):
    """Persist user state to disk."""
    try:
        data = {str(k): v for k, v in user_state.items()}
        # Remove last_image from persistence (it's binary data)
        for state in data.values():
            state.pop("last_image", None)
        with open(STATE_FILE, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug("Saved user state for %d users", len(data))
    except Exception as e:
        logger.error("Failed to save user state: %s", e, exc_info=True)


def reset_conversation(uid: int):
    """Reset conversation mode for a user."""
    user_state[uid]["conversation_mode"] = False
    user_state[uid]["last_output"] = None


def reset_chat(uid: int):
    """Reset chat mode for a user."""
    user_state[uid]["chat_mode"] = False
    user_state[uid]["chat_history"] = []


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
    uid = update.effective_user.id
    current_mode = user_state[uid]["mode"]
    current_name = PRESETS[current_mode]["name"]
    await update.message.reply_text(
        f"Welcome! Send me an image and I'll describe it using AI.\n\n"
        f"Your current default preset is *{current_name}*.\n\n"
        "I'll remember the last preset you used for each image. You can:\n"
        "- Add a caption like `tags` or `sd` to change your default and process the image\n"
        "- Use /mode to change your default preset without sending an image\n"
        "- Reply with text after an image to modify the description (e.g., 'make it darker')\n"
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
    lines.append("  /stop \u2014 Exit description modification mode")
    lines.append("  /chat \u2014 Enter chat mode for open-ended discussion")
    lines.append("  /help \u2014 Show this message")
    lines.append("\n*Modes:*")
    lines.append("- *Description modification*: Reply with text to edit the description (e.g., 'make it darker')")
    lines.append("- *Chat mode* (use /chat): Ask questions, get story prompts, creative ideas about the image")

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
        # Remember this as the user's new default
        old_mode = user_state[uid]["mode"]
        user_state[uid]["mode"] = preset_key
        logger.info("User %d: changing default from '%s' to '%s' (caption: %s)", uid, old_mode, preset_key, caption)
        save_user_state(uid)
    else:
        preset_key = user_state[uid]["mode"]
        logger.info("User %d: using saved default '%s' (no caption)", uid, preset_key)

    preset = PRESETS[preset_key]

    # Store for re-describe and chat
    user_state[uid]["last_image"] = image_bytes
    user_state[uid]["chat_history"] = []  # Reset chat history for new image

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

    # Store for conversation mode
    user_state[uid]["last_output"] = result
    user_state[uid]["conversation_mode"] = True

    parse_mode = ParseMode.MARKDOWN if preset["markdown"] else None
    keyboard = preset_keyboard(exclude=preset_key)

    await send_long_message(update, result, parse_mode=parse_mode, reply_markup=keyboard)


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Exit conversation mode."""
    uid = update.effective_user.id
    if user_state[uid]["conversation_mode"]:
        reset_conversation(uid)
        await update.message.reply_text("Conversation mode exited. Send a new image anytime.")
    else:
        await update.message.reply_text("Not in conversation mode. Send an image first.")


async def chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enter chat mode for open-ended discussion about the last image."""
    uid = update.effective_user.id
    if not user_state[uid].get("last_image"):
        await update.message.reply_text("Send an image first, then use /chat to discuss it.")
        return
    user_state[uid]["chat_mode"] = True
    user_state[uid]["chat_history"] = []  # Reset history for fresh chat
    await update.message.reply_text(
        "Chat mode enabled! Ask me anything about the image or request creative content like story prompts.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Exit Chat", callback_data="chat:exit")]]),
    )


async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages for conversation mode or chat mode."""
    uid = update.effective_user.id
    ollama_client: AsyncClient = context.bot_data["ollama_client"]
    model_name: str = context.bot_data["model_name"]

    # Check for chat mode first (open-ended conversation)
    if user_state[uid].get("chat_mode"):
        await _handle_chat_message(update, context, uid, ollama_client, model_name)
        return

    # Check for conversation mode (description modification)
    if user_state[uid].get("conversation_mode"):
        await _handle_conversation_message(update, context, uid, ollama_client, model_name)
        return

    # Ignore text when not in any mode
    return


async def _handle_chat_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    uid: int,
    ollama_client: AsyncClient,
    model_name: str,
):
    """Handle open-ended chat about an image."""
    # Check if we have an image
    image_bytes = user_state[uid].get("last_image")
    if not image_bytes:
        reset_chat(uid)
        await update.message.reply_text("No image found. Send an image first.")
        return

    user_message = update.message.text
    chat_history = user_state[uid].get("chat_history", [])

    # Typing indicator
    typing_task = await send_typing_loop(update.effective_chat.id, context)

    try:
        result = await chat_with_image(
            ollama_client, model_name, image_bytes, chat_history, user_message
        )
    except RuntimeError as e:
        typing_task.cancel()
        await update.message.reply_text(f"Error: {e}")
        return
    except Exception as e:
        typing_task.cancel()
        logger.error("chat_with_image failed: %s", e, exc_info=True)
        await update.message.reply_text("Something went wrong while processing your message.")
        return

    typing_task.cancel()

    # Store in chat history (user message + assistant response)
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": result})
    user_state[uid]["chat_history"] = chat_history
    logger.info("User %d: chat message '%s'", uid, user_message[:50])

    # Send response with exit button
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Exit Chat", callback_data="chat:exit")]])
    await send_long_message(update, result, reply_markup=keyboard)


async def _handle_conversation_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    uid: int,
    ollama_client: AsyncClient,
    model_name: str,
):
    """Handle description modification requests."""
    # Check if we have a last_output to modify
    last_output = user_state[uid].get("last_output")
    if not last_output:
        reset_conversation(uid)
        await update.message.reply_text("No image description to modify. Send an image first.")
        return

    modification_request = update.message.text

    # Typing indicator
    typing_task = await send_typing_loop(update.effective_chat.id, context)

    try:
        result = await modify_description(
            ollama_client, model_name, last_output, modification_request
        )
    except RuntimeError as e:
        typing_task.cancel()
        await update.message.reply_text(f"Error: {e}")
        return
    except Exception as e:
        typing_task.cancel()
        logger.error("modify_description failed: %s", e, exc_info=True)
        await update.message.reply_text("Something went wrong while modifying the description.")
        return

    typing_task.cancel()

    # Update state with new output
    user_state[uid]["last_output"] = result
    logger.info("User %d: modified description with '%s'", uid, modification_request[:50])

    # Send response with keyboard to continue or exit
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Exit Conversation", callback_data="conv:exit")]])
    await send_long_message(update, result, reply_markup=keyboard)


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
        save_user_state(uid)
        logger.info("User %d: changed mode to '%s' via keyboard", uid, key)
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

        # Update and save as new default
        user_state[uid]["mode"] = key
        save_user_state(uid)
        logger.info("User %d: changed mode to '%s' via redo button", uid, key)

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

        # Update last_output for conversation mode
        user_state[uid]["last_output"] = result
        user_state[uid]["conversation_mode"] = True

    elif data == "conv:exit":
        reset_conversation(uid)
        await query.edit_message_text("Conversation mode exited.")

    elif data == "chat:exit":
        reset_chat(uid)
        await query.edit_message_text("Chat mode exited.")


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
    model_name = os.getenv("MODEL_NAME", "qwen3-vl:8b")

    # Load persisted user state
    load_user_state()

    app = Application.builder().token(bot_token).build()

    app.bot_data["ollama_client"] = AsyncClient(host=ollama_url)
    app.bot_data["model_name"] = model_name

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("mode", mode_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("chat", chat_command))
    app.add_handler(
        MessageHandler(filters.PHOTO | filters.Document.IMAGE, image_handler)
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))
    app.add_handler(CallbackQueryHandler(callback_handler))

    logger.info("Bot starting with model %s at %s", model_name, ollama_url)
    app.run_polling()


if __name__ == "__main__":
    main()
