import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger,
    TensorBoard, LearningRateScheduler, Callback
)
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Dropout
from keras.utils import get_custom_objects
import math
import csv
import copyreg
import types

# ===============================
# 0. Reproducibility & config
# ===============================
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Prevent deepcopy error when saving checkpoints with generators
def _pickle_module(m):
    return m.__name__
copyreg.pickle(types.ModuleType, _pickle_module)

# ✅ Register Swish so old .h5 models load correctly
def swish(x):
    return K.sigmoid(x) * x
get_custom_objects().update({'swish': Activation(swish)})

# ✅ Register FixedDropout (EfficientNet dropout layer)
class FixedDropout(Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)
get_custom_objects().update({'FixedDropout': FixedDropout})

# Paths
train_dir = r"C:\xamppnew\htdocs\EmoGame\EmoGame\data\train"
val_dir   = r"C:\xamppnew\htdocs\EmoGame\EmoGame\data\test"
resume_model_path = "models/emotion_model_full_local.h5"
best_model_path   = "models/best_model.h5"
summary_log_path  = "models/val_accuracy_summary.csv"
train_log_path    = "models/training_log.csv"

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
FINETUNE_LEARNING_RATE = 1e-5
START_EPOCH = 119
END_EPOCH = 230

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ===============================
# 1. Data generators
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    brightness_range=(0.7, 1.3),
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print("Class mapping:", train_generator.class_indices)

# ===============================
# 2. MixUp generator
# ===============================
def mixup_generator(generator, alpha=0.2):
    while True:
        x, y = next(generator)
        lam = np.random.beta(alpha, alpha, x.shape[0])
        lam_x = lam.reshape(x.shape[0], 1, 1, 1)
        lam_y = lam.reshape(x.shape[0], 1)
        index = np.random.permutation(x.shape[0])
        x_mix = x*lam_x + x[index]*(1-lam_x)
        y_mix = y*lam_y + y[index]*(1-lam_y)
        yield x_mix, y_mix

train_gen = mixup_generator(train_generator)

# ===============================
# 3. Load model
# ===============================
initial_epoch = START_EPOCH
if os.path.exists(resume_model_path):
    model = load_model(resume_model_path, compile=False)  # custom objects fixed
    print(f"✅ Loaded model from {resume_model_path}, resuming training at epoch {START_EPOCH}...")
else:
    raise FileNotFoundError(f"{resume_model_path} not found! Cannot resume training.")

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
model.compile(
    optimizer=Adam(learning_rate=FINETUNE_LEARNING_RATE),
    loss=loss_fn,
    metrics=['accuracy']
)

# ===============================
# 4. Callbacks
# ===============================
checkpoint_cb = ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, mode="max", verbose=1)
csv_logger = CSVLogger(train_log_path, append=True)
reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
earlystop_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
tensorboard_cb = TensorBoard(log_dir='logs', histogram_freq=1)

# Cosine decay for resumed training
def resumed_cosine_decay(epoch):
    total_epochs = END_EPOCH - START_EPOCH
    current_epoch = epoch - START_EPOCH
    lr = 0.5 * FINETUNE_LEARNING_RATE * (1 + math.cos(math.pi * current_epoch / total_epochs))
    return lr

cosine_cb = LearningRateScheduler(resumed_cosine_decay)

# ===============================
# 5. Custom callback to log val_accuracy
# ===============================
class ValAccuracyLogger(Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "val_accuracy", "is_new_best"])
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get("val_accuracy", 0.0)
        is_best = val_acc > self.best_val_acc
        if is_best:
            self.best_val_acc = val_acc
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, val_acc, is_best])
        print(f"📌 Epoch {epoch+1} — val_accuracy: {val_acc:.4f} {'(NEW BEST!)' if is_best else ''}")

val_logger_cb = ValAccuracyLogger(summary_log_path)

# ===============================
# 6. Training
# ===============================
steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
validation_steps = max(1, val_generator.samples // BATCH_SIZE)

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    initial_epoch=initial_epoch,
    epochs=END_EPOCH,
    callbacks=[checkpoint_cb, csv_logger, reduce_lr_cb, earlystop_cb, tensorboard_cb, cosine_cb, val_logger_cb]
)

# ===============================
# 7. Save final model & evaluate
# ===============================
final_model_path = "models/emotion_model_full_local.h5"
model.save(final_model_path)
print(f"✅ Saved final model to: {final_model_path}")

loss, acc = model.evaluate(val_generator, steps=validation_steps, verbose=1)
print(f"📊 Final evaluation — loss: {loss:.4f}, val_accuracy: {acc:.4f}")
