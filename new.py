import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0

# -----------------------------
# Set random seeds for reproducibility
# -----------------------------
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# -----------------------------
# Data directories
# -----------------------------
train_dir = 'data/train'
val_dir = 'data/test'

# -----------------------------
# Data generators
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# -----------------------------
# Model architecture (EfficientNetB0 base)
# -----------------------------
input_tensor = Input(shape=(48,48,1))
# Convert grayscale to 3 channels
x = Lambda(lambda y: tf.tile(y, [1,1,1,3]))(input_tensor)

base_model = EfficientNetB0(
    include_top=False,
    weights=None,   # Set 'imagenet' if you want pretrained weights
    input_tensor=x,
    pooling='avg'
)

x = Dense(512, activation='relu')(base_model.output)
x = Dropout(0.5)(x)
output = Dense(8, activation='softmax')(x)  # 8 classes (FER+RAF)

emotion_model = Model(inputs=input_tensor, outputs=output)

# -----------------------------
# Compile model
# -----------------------------
emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# -----------------------------
# Callbacks
# -----------------------------
checkpoint_cb = ModelCheckpoint(
    'best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)
tensorboard_cb = TensorBoard(log_dir='logs', histogram_freq=1)
csv_logger = CSVLogger('training_log.csv', append=True)
earlystop_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

# -----------------------------
# Train model
# -----------------------------
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[checkpoint_cb, tensorboard_cb, csv_logger, earlystop_cb, reduce_lr_cb],
    verbose=2
)

# -----------------------------
# Save final model & weights
# -----------------------------
emotion_model.save('emotion_model_full.keras')  # native Keras format
emotion_model.save_weights('model.weights.h5')

print("\nTraining complete. Model and weights saved.")

# -----------------------------
# Evaluate final model
# -----------------------------
final_train_acc = emotion_model_info.history['accuracy'][-1]
final_val_acc = emotion_model_info.history['val_accuracy'][-1]

print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")

# -----------------------------
# Evaluate best saved model
# -----------------------------
best_model = load_model('best_model.h5')
best_val_loss, best_val_acc = best_model.evaluate(val_generator, verbose=0)
print(f"Best Model Validation Accuracy (from checkpoint): {best_val_acc:.4f}")

print("\nTensorBoard logs saved in ./logs. Run 'tensorboard --logdir=logs' to visualize.")
