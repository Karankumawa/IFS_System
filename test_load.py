"""Local smoke test for the supplied festival model and label metadata."""

from __future__ import annotations

import tensorflow as tf

from festival_config import IMAGE_SIZE, MODEL_PATH, load_class_names, validate_model_output


def main() -> None:
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Model file not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()
    validate_model_output(model, class_names)

    output = model(tf.zeros((1, *IMAGE_SIZE, 3)), training=False)
    if output.shape != (1, len(class_names)):
        raise RuntimeError(f"Unexpected model output shape: {output.shape}")
    print(f"SUCCESS: model loaded with {len(class_names)} classes.")


if __name__ == "__main__":
    main()
