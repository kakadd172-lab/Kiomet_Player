
import base64
import requests

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "screenshots"
    / "screen_20261118_220722.png"
)

MODEL = "qwen2.5vl:3b"


def main():
    if not IMAGE_PATH.exists():
        print(f"Screenshot not found: {IMAGE_PATH}")
        return

    print("[Vision LLM] Loading screenshot...")

    image_data = base64.b64encode(
        IMAGE_PATH.read_bytes()
    ).decode("utf-8")

    print("[Vision LLM] Sending image to Ollama...")

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Analyze this screenshot of Kiomet. "
                        "Describe what is visible. "
                        "Identify the screen type, interface "
                        "elements, buttons and any visible "
                        "game information. Do not guess "
                        "information that cannot be seen."
                    ),
                    "images": [image_data]
                }
            ],
            "stream": False
        },
        timeout=300
    )

    response.raise_for_status()

    result = response.json()

    print("\n========== VISION MODEL OUTPUT ==========\n")
    print(result["message"]["content"])
    print("\n=========================================")


if __name__ == "__main__":
    main()
