# Vision Bot

Telegram bot that describes images using Qwen3-VL running locally via Ollama.

## Setup

```bash
ollama pull qwen3-vl:4b
cp .env.example .env   # then fill in BOT_TOKEN
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python bot.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BOT_TOKEN` | — | Telegram bot token from [@BotFather](https://t.me/BotFather) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `MODEL_NAME` | `qwen3-vl:4b` | Model to use for image description |

## Usage

Send an image to the bot and it replies with an AI-generated description.

### Presets

| Key | Name | Description |
|-----|------|-------------|
| `simple` | Simple | Single concise sentence |
| `detailed` | Detailed | One paragraph, 6-10 sentences (default) |
| `tags` | Tags | Comma-separated visual tags, max 50 |
| `cinematic` | Cinematic | Film-still style paragraph |
| `style` | Style Focus | Art-director analysis of visual style |
| `z_image` | Z-Image | Structured visual description for image generation |
| `sd` | Stable Diffusion | SD/SDXL-optimized tag prompt with weight syntax |

### Commands

- `/start` — Welcome message
- `/help` — List presets and current mode
- `/mode` — Change default preset via inline keyboard

### Quick override

Send an image with a caption like `sd`, `tags`, or `cinematic` to use that preset for one image without changing your default.

### Re-describe

After each response, inline buttons let you re-describe the same image with a different preset.

## Model slot management

The bot shares the Ollama server with other workloads. Before processing, it checks `ollama ps` and waits (up to 5 minutes) if another model is loaded. After each response, the model is unloaded immediately (`keep_alive=0`).

## Prompt sources

- Simple, Detailed, Tags, Cinematic — [ComfyUI-QwenVL](https://github.com/1038lab/ComfyUI-QwenVL)
- Style Focus, Z-Image, Stable Diffusion — [AutoDescribe-Images](https://github.com/hydropix/AutoDescribe-Images)
