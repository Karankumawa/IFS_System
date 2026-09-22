import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TRAIN_DIR = r"e:\IFS_System\dataset\festivals_classification\train"
VAL_DIR = r"e:\IFS_System\dataset\festivals_classification\test"

IMG_SIZE = (224, 224) 
BATCH_SIZE = 32
EPOCHS = 20
MODEL_SAVE_PATH = "festival_model.keras"
LABELS_SAVE_PATH = "festival_model.labels.json"

# ==============================================================================
# DATA PREPARATION & AUGMENTATION
# ==============================================================================
def create_datasets():
    print("Loading datasets...")
    
    # Robust Data Augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
    ])

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode='categorical'
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        shuffle=False,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode='categorical'
    )
    
    class_names = train_dataset.class_names
    
    # Save class names to a JSON file so the backend can map indices to names
    with open(LABELS_SAVE_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"Saved {len(class_names)} class labels to {LABELS_SAVE_PATH}")

    # Apply augmentation to training dataset
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), 
                                      num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, class_names

# ==============================================================================
# MODEL BUILDING (TRANSFER LEARNING)
# ==============================================================================
def build_model(num_classes):
    print("Building Robust Transfer Learning model (MobileNetV2)...")
    
    # Load base model with pre-trained ImageNet weights
    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    
    # Freeze the base layers
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    
    # Preprocess input strictly inside the model to scale pixels to [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x) # 30% dropout for better regularization
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ==============================================================================
# TRAINING & EVALUATION
# ==============================================================================
def plot_history(history):
    """Plots training/validation accuracy and loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    
    plt.savefig("training_history.png")
    print("Saved training_history.png")

def main():
    train_dataset, val_dataset, class_names = create_datasets()
    
    model = build_model(len(class_names))
    model.summary()
    
    # Callbacks for robust training
    early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
    checkpoint = ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    
    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )
    
    print(f"\nTraining completed. Best model saved safely as {MODEL_SAVE_PATH}")
    plot_history(history)

if __name__ == "__main__":
    main()
