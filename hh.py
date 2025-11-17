import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
import efficientnet.tfkeras as efn
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# 1. Paths & settings
# ===============================
checkpoint_path = 'models/emotion_model_full_local.h5'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

val_dir = r"C:\xamppnew\htdocs\EmoGame\EmoGame\data\test"

# ===============================
# 2. Validation generator
# ===============================
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(val_generator.class_indices)

# ===============================
# 3. Build the same model
# ===============================
base_model = efn.EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.6)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

# ===============================
# 4. Load weights safely
# ===============================
model.load_weights(checkpoint_path, by_name=True)

# ===============================
# 5. Compile & evaluate
# ===============================
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
steps = val_generator.samples // BATCH_SIZE
loss, acc = model.evaluate(val_generator, steps=steps, verbose=1)
print(f"Best model accuracy on validation set: {acc:.4f}")
