"""Gemini-powered deity/festival webcam demo with local audio playback."""

from __future__ import annotations

import os
import threading
import time

import cv2
import pygame
from google import genai
from google.genai import types
from PIL import Image

from festival_config import PROJECT_ROOT

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
API_COOLDOWN_SECONDS = 4.0
DEITY_AUDIO = {
    # This is the only matching local asset currently included in the project.
    "diwali": PROJECT_ROOT / "Happy_Diwali.mp3",
}
SYSTEM_PROMPT = (
    "Analyze this image. Reply with exactly one lowercase word from: "
    "ganesha, krishna, shiva, diwali, holi, none. "
    "Use none when none of those are clearly present."
)
VALID_RESPONSES = {*DEITY_AUDIO, "ganesha", "krishna", "shiva", "holi", "none"}
current_playing: str | None = None


def normalize_response(response_text: str) -> str:
    """Accept only an exact allowed classification from the model response."""
    answer = "".join(character for character in response_text.lower() if character.isalpha())
    return answer if answer in VALID_RESPONSES else "none"


def play_audio(keyword: str) -> None:
    global current_playing

    audio_path = DEITY_AUDIO.get(keyword)
    if audio_path is None:
        print(f"Detected {keyword}; no local audio track is configured.")
        return
    if not audio_path.is_file():
        print(f"Configured audio file is missing: {audio_path}")
        return
    if current_playing == keyword and pygame.mixer.music.get_busy():
        return

    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.load(str(audio_path))
        pygame.mixer.music.play(loops=-1)
        current_playing = keyword
        print(f"Playing: {audio_path.name}")
    except pygame.error as exc:
        print(f"Could not play {audio_path.name}: {exc}")
        current_playing = None


def process_frame(client: genai.Client, frame_bgr, processing: threading.Event) -> None:
    """Classify one stable frame without blocking the webcam display loop."""
    try:
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[SYSTEM_PROMPT, image],
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=8),
        )
        keyword = normalize_response(response.text or "")
        print(f"Gemini detected: {keyword}")
        if keyword != "none":
            play_audio(keyword)
    except Exception as exc:
        print(f"Gemini request failed: {exc}")
    finally:
        processing.clear()


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY before running this optional demo.")

    try:
        client = genai.Client(api_key=api_key)
        pygame.mixer.init()
    except Exception as exc:
        raise SystemExit(f"Startup failed: {exc}") from exc

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        pygame.mixer.quit()
        raise SystemExit("Could not open the default webcam.")

    processing = threading.Event()
    last_request_time = 0.0
    print("Gemini deity/festival demo started. Press q in the video window to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a webcam frame.")
                break
            cv2.imshow("Gemini Deity/Festival Detection", frame)

            now = time.monotonic()
            if now - last_request_time >= API_COOLDOWN_SECONDS and not processing.is_set():
                processing.set()
                last_request_time = now
                threading.Thread(
                    target=process_frame,
                    args=(client, frame.copy(), processing),
                    daemon=True,
                ).start()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()


if __name__ == "__main__":
    main()
