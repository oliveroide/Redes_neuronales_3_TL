import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FE = 10
EPOCHS_FT = 10
FINE_TUNE_AT = 100
AUTOTUNE = tf.data.AUTOTUNE
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "resultados")
TFDS_DATA_DIR = os.environ.get("TFDS_DATA_DIR")

os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

print(f"TensorFlow: {tf.__version__}")
print(f"TensorFlow Datasets: {tfds.__version__}")
print(f"GPUs: {[gpu.name for gpu in gpus]}")
print(f"TFDS_DATA_DIR: {TFDS_DATA_DIR}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")

(train_raw, val_raw, test_raw), info = tfds.load(
    "plant_leaves",
    split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
    as_supervised=True,
    with_info=True,
    data_dir=TFDS_DATA_DIR,
)

n_classes = info.features["label"].num_classes
class_names = list(info.features["label"].names)

print(f"Clases: {n_classes}")
print(f"Total de imágenes: {info.splits['train'].num_examples}")

def format_example(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    return image, label

train_ds = (
    train_raw.map(format_example, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(1000, seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
val_ds = (
    val_raw.map(format_example, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTOTUNE)
)
test_ds = (
    test_raw.map(format_example, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTOTUNE)
)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(
    n_classes,
    activation="softmax",
    kernel_initializer="glorot_uniform",
)(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

def make_callbacks(phase):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, f"mejor_{phase}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(OUTPUT_DIR, f"log_{phase}.csv")
        ),
    ]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FE,
    callbacks=make_callbacks("fase1"),
)

base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=len(history.epoch) + EPOCHS_FT,
    initial_epoch=len(history.epoch),
    callbacks=make_callbacks("fase2"),
)

best_path = os.path.join(OUTPUT_DIR, "mejor_fase2.keras")
if os.path.exists(best_path):
    model = tf.keras.models.load_model(best_path)

acc = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]
loss = history.history["loss"] + history_fine.history["loss"]
val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(acc, label="Train")
axes[0].plot(val_acc, label="Val")
axes[0].axvline(
    len(history.history["accuracy"]) - 1,
    linestyle="--",
    label="Fine-Tuning",
)
axes[0].set_title("Exactitud")
axes[0].legend(loc="lower right")

axes[1].plot(loss, label="Train")
axes[1].plot(val_loss, label="Val")
axes[1].axvline(
    len(history.history["loss"]) - 1,
    linestyle="--",
    label="Fine-Tuning",
)
axes[1].set_title("Pérdida")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "curvas_entrenamiento.png"), dpi=150)
plt.close(fig)

test_loss, test_acc = model.evaluate(test_ds)
print(f"Pérdida en Test: {test_loss:.4f}")
print(f"Exactitud en Test: {test_acc:.4f}")

y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)

cm = confusion_matrix(y_true, y_pred)
print(cm)

fig, ax = plt.subplots(figsize=(12, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=90)
ax.set_title("Matriz de Confusión - Plant Leaves (22 clases)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "matriz_confusion.png"), dpi=150)
plt.close(fig)

model.save(os.path.join(OUTPUT_DIR, "modelo_plant_leaves.keras"))

with open(os.path.join(OUTPUT_DIR, "clases.json"), "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

historial = {}
for key in history.history:
    valores = history.history[key] + history_fine.history.get(key, [])
    historial[key] = [float(v) for v in valores]
with open(os.path.join(OUTPUT_DIR, "historial.json"), "w", encoding="utf-8") as f:
    json.dump(historial, f, ensure_ascii=False, indent=2)

np.save(os.path.join(OUTPUT_DIR, "y_true.npy"), y_true)
np.save(os.path.join(OUTPUT_DIR, "y_pred.npy"), y_pred)

print("Entrenamiento finalizado")
print(f"Resultados en: {OUTPUT_DIR}")
