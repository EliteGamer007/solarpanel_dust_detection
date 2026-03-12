import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Performance / device setup
print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
USE_GPU = len(gpus) > 0
if USE_GPU:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("GPUs:", gpus)
else:
    print("GPUs: none (TensorFlow will use CPU).")

USE_JIT = USE_GPU

# ---- Config ----
DATASET_PATH = "Detect_solar_dust"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 1337
AUTOTUNE = tf.data.AUTOTUNE
WEIGHTS_PATH = "best_resnet_solar.keras"
FULL_MODEL_PATH = "resnet50_solar_full.keras"

# ---- Data Augmentation ----
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(factor=0.2, fill_mode="nearest"),
        tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(height_factor=(-0.3, 0.3), fill_mode="nearest"),
        tf.keras.layers.RandomBrightness(factor=0.2),
        tf.keras.layers.RandomContrast(factor=0.2),
    ],
    name="data_augmentation",
)

# ---- Load Dataset ----
train_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels="inferred",
    label_mode="binary",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels="inferred",
    label_mode="binary",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Verify class balance
print("\nVerifying Validation Split Balance...")
y_val = []
for _, labels in val_raw:
    y_val.extend(labels.numpy().flatten())
unique, counts = np.unique(y_val, return_counts=True)
print(f"Validation Class Distribution: {dict(zip(unique, counts))}")
print(f"Class Names: {train_raw.class_names}\n")

# Compute class weights to handle imbalance
y_train = []
for _, labels in train_raw:
    y_train.extend(labels.numpy().flatten())
y_train = np.array(y_train)
n_samples = len(y_train)
n_classes = 2
class_counts = np.bincount(y_train.astype(int))
class_weight = {i: n_samples / (n_classes * count) for i, count in enumerate(class_counts)}
print(f"Class Weights: {class_weight}\n")


def train_preprocess(images, labels):
    images = data_augmentation(images, training=True)
    images = preprocess_input(tf.cast(images, tf.float32))
    return images, labels


def val_preprocess(images, labels):
    images = preprocess_input(tf.cast(images, tf.float32))
    return images, labels


# NO caching for train - forces fresh augmentation every epoch (critical for small data)
train_data = (
    train_raw
    .map(train_preprocess, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

val_data = (
    val_raw
    .map(val_preprocess, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)


# ---- Build Model ----
def build_resnet_model():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid", dtype="float32")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model


model = build_resnet_model()

model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
    jit_compile=USE_JIT,
)

# ---- Callbacks ----
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        WEIGHTS_PATH, monitor="val_accuracy",
        save_best_only=True, mode="max", verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=3, min_lr=1e-6, verbose=1
    ),
]

# ---- Phase 1: Train head only ----
print("=" * 60)
print("Phase 1: Training classification head (base frozen)")
print("=" * 60)
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=callbacks,
    class_weight=class_weight,
)

# ---- Phase 2: Fine-tune last 50 layers ----
print("=" * 60)
print("Phase 2: Fine-tuning last 50 layers")
print("=" * 60)

base_model = model.layers[1] if hasattr(model.layers[1], 'layers') else None
# Get the ResNet50 base by name
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "resnet" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    # Fallback: the input model wraps ResNet50 as a functional sub-graph
    # Unfreeze the last 50 layers of the overall model
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-50:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
else:
    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=5e-6),
    loss="binary_crossentropy",
    metrics=["accuracy"],
    jit_compile=USE_JIT,
)

print("Trainable variables:", len(model.trainable_variables))

fine_tune_history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=callbacks,
    class_weight=class_weight,
)

# ---- Evaluate & Save ----
val_loss, val_acc = model.evaluate(val_data)
print(f"\nFinal Validation Accuracy: {val_acc:.4f}")
print(f"Final Validation Loss:     {val_loss:.4f}")

model.save(FULL_MODEL_PATH)
print(f"\nFull model saved to {FULL_MODEL_PATH}")
print(f"Best weights saved to {WEIGHTS_PATH}")
