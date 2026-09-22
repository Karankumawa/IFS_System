"""Shared paths, labels, and runtime checks for festival recognition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
FESTIVAL_DATASET_DIR = PROJECT_ROOT / "dataset" / "festivals_classification"
TRAIN_DIR = FESTIVAL_DATASET_DIR / "train"
# The downloaded dataset calls this split "test". The training script uses it
# for validation; keep a separate holdout split before publishing metrics.
VALIDATION_DIR = FESTIVAL_DATASET_DIR / "test"

MODEL_PATH = PROJECT_ROOT / "festival_model.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "festival_model.labels.json"
HISTORY_PATH = PROJECT_ROOT / "training_history.png"
AUDIO_MAPPING_PATH = PROJECT_ROOT / "audio_mapping.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# This fallback makes the supplied model usable. New training runs write the
# authoritative order to festival_model.labels.json.
DEFAULT_CLASS_NAMES = (
    "CHITHIRAI THIRUVIZHA",
    "DIWALI",
    "HOLI",
    "JALLIKATU",
    "KARTHIGAI DEEPAM",
    "PONGAL",
    "RAKSHA BANDHAN",
    "THAIPUSAM",
    "THIRUVAIYARU",
)

# Only map audio that is actually included in the project. The desktop app lets
# the user configure additional local audio tracks in audio_mapping.json.
DEFAULT_FESTIVAL_AUDIO = {
    "DIWALI": PROJECT_ROOT / "Happy_Diwali.mp3",
    "HOLI": PROJECT_ROOT / "static/audio/holi.mp3",
    "RAKSHA BANDHAN": PROJECT_ROOT / "static/audio/rakhi.mp3",
}


def load_class_names() -> tuple[str, ...]:
    """Load the label order saved with the model, or use the bundled fallback."""
    if not CLASS_NAMES_PATH.exists():
        return DEFAULT_CLASS_NAMES

    try:
        data = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read label metadata: {CLASS_NAMES_PATH}") from exc

    if not isinstance(data, list) or not data or not all(
        isinstance(label, str) and label.strip() for label in data
    ):
        raise RuntimeError(f"Invalid label metadata: {CLASS_NAMES_PATH}")
    return tuple(data)


def save_class_names(class_names: Sequence[str]) -> None:
    """Persist the model-output order for the app and webcam inference."""
    labels = list(class_names)
    if not labels or not all(isinstance(label, str) and label.strip() for label in labels):
        raise ValueError("Class names must be a non-empty sequence of strings.")
    CLASS_NAMES_PATH.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_festival_audio() -> dict[str, Path]:
    """Load user-selected local audio tracks while retaining bundled defaults."""
    audio_mapping = DEFAULT_FESTIVAL_AUDIO.copy()
    if not AUDIO_MAPPING_PATH.exists():
        return audio_mapping

    try:
        data = json.loads(AUDIO_MAPPING_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read audio mapping: {AUDIO_MAPPING_PATH}") from exc
    if not isinstance(data, dict) or not all(
        isinstance(class_name, str) and isinstance(path, str)
        for class_name, path in data.items()
    ):
        raise RuntimeError(f"Invalid audio mapping: {AUDIO_MAPPING_PATH}")

    for class_name, stored_path in data.items():
        path = Path(stored_path).expanduser()
        audio_mapping[class_name] = path if path.is_absolute() else PROJECT_ROOT / path
    return audio_mapping


def save_festival_audio(audio_mapping: dict[str, Path]) -> None:
    """Save paths relative to the project where possible for portability."""
    serializable_mapping: dict[str, str] = {}
    for class_name, path in audio_mapping.items():
        resolved_path = path.expanduser().resolve(strict=False)
        try:
            serializable_mapping[class_name] = resolved_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            serializable_mapping[class_name] = str(resolved_path)
    AUDIO_MAPPING_PATH.write_text(
        json.dumps(serializable_mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_model_output(model: object, class_names: Sequence[str]) -> None:
    """Fail early when a model and its label metadata do not belong together."""
    output_shape = getattr(model, "output_shape", None)
    if not output_shape or output_shape[-1] is None:
        raise RuntimeError("The loaded model does not expose a single classification output.")
    if int(output_shape[-1]) != len(class_names):
        raise RuntimeError(
            f"Model has {output_shape[-1]} output classes but label metadata has "
            f"{len(class_names)} entries. Retrain the model or restore matching metadata."
        )
