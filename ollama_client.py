import re
import time

from ollama import AsyncClient

from prompts import CHAT_SYSTEM_PROMPT, CONVERSATION_PROMPT


async def wait_for_model_slot(client: AsyncClient, model_name: str, timeout: int = 300) -> bool:
    """Wait until no other model is loaded, or our model is already loaded."""
    deadline = time.monotonic() + timeout
    while True:
        response = await client.ps()
        models = response.get("models", [])
        if not models:
            return True
        loaded_names = [m["name"] for m in models]
        if model_name in loaded_names:
            return True
        if time.monotonic() >= deadline:
            return False
        await _async_sleep(5)


async def _async_sleep(seconds: float):
    import asyncio
    await asyncio.sleep(seconds)


async def describe_image(
    client: AsyncClient,
    model_name: str,
    system_prompt: str,
    image_bytes: bytes,
) -> str:
    """Send an image to the model and return the cleaned description."""
    slot = await wait_for_model_slot(client, model_name)
    if not slot:
        raise RuntimeError(
            "Timed out waiting for Ollama model slot. Another model is still loaded."
        )

    response = await client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Describe this image.",
                "images": [image_bytes],
            },
        ],
        keep_alive=0,
        options={"temperature": 0.7, "num_ctx": 8192},
    )

    return clean_output(response["message"]["content"])


async def modify_description(
    client: AsyncClient,
    model_name: str,
    original_description: str,
    modification_request: str,
) -> str:
    """Modify an existing description based on user request."""
    prompt = CONVERSATION_PROMPT.format(
        original_description=original_description,
        modification_request=modification_request,
    )

    response = await client.chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You modify image descriptions based on user requests. Output only the revised description, no explanations.",
            },
            {"role": "user", "content": prompt},
        ],
        keep_alive=0,
        options={"temperature": 0.5, "num_ctx": 8192},
    )

    return clean_output(response["message"]["content"])


async def chat_with_image(
    client: AsyncClient,
    model_name: str,
    image_bytes: bytes,
    chat_history: list[dict],
    user_message: str,
    max_history_turns: int = 10,
) -> str:
    """Multi-turn chat with image context."""
    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": "Analyze this image.", "images": [image_bytes]},
        *chat_history[-max_history_turns:],
        {"role": "user", "content": user_message},
    ]

    response = await client.chat(
        model=model_name,
        messages=messages,
        keep_alive="5m",
        options={"temperature": 0.7, "num_ctx": 8192},
    )

    return clean_output(response["message"]["content"])


def clean_output(text: str) -> str:
    """Strip think blocks, code fences, common preambles, and normalize whitespace."""
    # Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip code fences
    text = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).strip("`").strip(), text)
    # Strip common preambles
    preambles = [
        r"^Here is the description:\s*",
        r"^Here's the description:\s*",
        r"^Sure,?\s*here'?s?\s*(the|a|my)?\s*(detailed\s*)?(description|response|analysis)[\s:]*",
        r"^Certainly[!.]?\s*(Here'?s?\s*)?(the|a|my)?\s*(description)?[\s:]*",
    ]
    for pattern in preambles:
        text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
