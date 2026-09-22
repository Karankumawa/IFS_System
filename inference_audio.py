"""Webcam festival recognition with serialized local-audio playback."""

from __future__ import annotations

import queue
import threading
import time

import cv2
import numpy as np
import pygame
import tensorflow as tf

from festival_config import (
    FESTIVAL_AUDIO,
    IMAGE_SIZE,
    MODEL_PATH,
    load_class_names,
    validate_model_output,
)

CONFIDENCE_THRESHOLD = 0.85
CONSECUTIVE_PREDICTIONS_REQUIRED = 3
FRAME_SKIP = 10


class AudioController:
    """A single audio worker that coalesces stale class changes safely."""

    def __init__(self) -> None:
        self._requests: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._desired_class = "background"
        self._playing_class = "background"
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @property
    def playing_class(self) -> str:
        with self._lock:
            return self._playing_class

    def request(self, class_name: str) -> None:
        """Schedule only the newest audio state, never concurrent mixer access."""
        with self._lock:
            if class_name == self._desired_class:
                return
            self._desired_class = class_name

        try:
            self._requests.get_nowait()
        except queue.Empty:
            pass
        try:
            self._requests.put_nowait(class_name)
        except queue.Full:
            # The worker picked up a request after the drain; it will check the
            # desired state before starting a track, so no action is needed.
            pass

    def _is_current_request(self, class_name: str) -> bool:
        with self._lock:
            return class_name == self._desired_class

    def _set_playing_class(self, class_name: str) -> None:
        with self._lock:
            self._playing_class = class_name

    def _run(self) -> None:
        while True:
            class_name = self._requests.get()
            if class_name is None:
                return
            self._switch_audio(class_name)

    def _switch_audio(self, class_name: str) -> None:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.fadeout(300)
            time.sleep(0.35)
        if not self._is_current_request(class_name):
            return

        audio_path = FESTIVAL_AUDIO.get(class_name)
        if audio_path is None:
            self._set_playing_class("background")
            return
        if not audio_path.is_file():
            print(f"Audio file not found: {audio_path}")
            self._set_playing_class("background")
            return

        try:
            pygame.mixer.music.load(str(audio_path))
            pygame.mixer.music.play(loops=-1, fade_ms=250)
            self._set_playing_class(class_name)
            print(f"Playing: {audio_path.name}")
        except pygame.error as exc:
            print(f"Could not play {audio_path.name}: {exc}")
            self._set_playing_class("background")

    def close(self) -> None:
        self._requests.put(None)
        self._worker.join(timeout=1)
        pygame.mixer.music.stop()


def load_model() -> tuple[tf.keras.Model, tuple[str, ...]]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()
    validate_model_output(model, class_names)
    return model, class_names


def predict(
    frame_bgr: np.ndarray, model: tf.keras.Model, class_names: tuple[str, ...]
) -> tuple[str, float]:
    resized = cv2.resize(frame_bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    model_input = np.expand_dims(image_rgb, axis=0)
    model_input = tf.keras.applications.mobilenet_v2.preprocess_input(model_input)
    probabilities = model.predict(model_input, verbose=0)[0]
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    return (
        class_names[index] if confidence >= CONFIDENCE_THRESHOLD else "background",
        confidence,
    )


def main() -> None:
    try:
        print("Loading model...")
        model, class_names = load_model()
        pygame.mixer.init()
    except Exception as exc:
        raise SystemExit(f"Startup failed: {exc}") from exc

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        pygame.mixer.quit()
        raise SystemExit("Could not open the default webcam.")

    audio = AudioController()
    recent_predictions: list[str] = []
    frame_count = 0
    display_label = "background"
    display_confidence = 0.0
    print("Festival recognition started. Press q in the video window to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a webcam frame.")
                break

            frame_count += 1
            if frame_count % FRAME_SKIP == 0:
                detected_class, display_confidence = predict(frame, model, class_names)
                recent_predictions.append(detected_class)
                recent_predictions = recent_predictions[-CONSECUTIVE_PREDICTIONS_REQUIRED:]

                if len(recent_predictions) == CONSECUTIVE_PREDICTIONS_REQUIRED and len(
                    set(recent_predictions)
                ) == 1:
                    display_label = recent_predictions[0]
                    audio.request(display_label)

            color = (0, 255, 0) if display_label != "background" else (0, 0, 255)
            cv2.putText(
                frame,
                f"Class: {display_label} ({display_confidence:.1%})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"Playing: {audio.playing_class}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            cv2.imshow("Festival Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        audio.close()
        pygame.mixer.quit()


if __name__ == "__main__":
    main()
