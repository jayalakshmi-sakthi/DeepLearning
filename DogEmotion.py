# ----------------------------
# Dog Emotion Classifier (Colab Full Pipeline)
# ----------------------------

# 1️⃣ Install Kaggle
!pip install kaggle

# 2️⃣ Upload your Kaggle API key
from google.colab import files
files.upload()  # upload kaggle.json here

import os
os.environ['KAGGLE_CONFIG_DIR'] = "/content"

# 3️⃣ Download the dataset from Kaggle
!kaggle datasets download -d danielshanbalico/dog-emotion --force

# 4️⃣ Unzip only selected emotion folders
import zipfile

with zipfile.ZipFile("dog-emotion.zip","r") as zip_ref:
    all_files = zip_ref.namelist()
    # Choose only a subset of emotions to reduce memory usage
    selected_files = [f for f in all_files if f.startswith("Dog Emotion/happy/")
                                     or f.startswith("Dog Emotion/angry/")
                                     or f.startswith("Dog Emotion/sad/")]
    for f in selected_files:
        zip_ref.extract(f, "dog-emotion")

# Check extracted folders
print("Folders inside dog-emotion/Dog Emotion:", os.listdir("dog-emotion/Dog Emotion"))

# ----------------------------
# 5️⃣ Imports & Setup
# ----------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from google.colab import files

# ----------------------------
# 6️⃣ Data Preprocessing
# ----------------------------
img_size = 128
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    "dog-emotion/Dog Emotion",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    "dog-emotion/Dog Emotion",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Ensure dataset has more than 1 class
if train_generator.num_classes <= 1:
    raise ValueError("Dataset must have at least 2 classes. Check folder structure.")

print("Detected classes:", train_generator.class_indices)

# ----------------------------
# 7️⃣ Build CNN Model
# ----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ----------------------------
# 8️⃣ Train Model
# ----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5  # increase epochs for better accuracy
)

# ----------------------------
# 9️⃣ Save Model
# ----------------------------
model.save("dog_emotion_classifier.h5")
print("✅ Model trained & saved as dog_emotion_classifier.h5")

# ----------------------------
# 🔟 Predict Uploaded Image
# ----------------------------
uploaded = files.upload()  # upload any dog image

class_labels = list(train_generator.class_indices.keys())
model = load_model("dog_emotion_classifier.h5")

for fn in uploaded.keys():
    print(f"Uploaded file: {fn}")
    img = load_img(fn, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    emotion = class_labels[predicted_class]

    print(f"🐶 Predicted Emotion: {emotion}")
