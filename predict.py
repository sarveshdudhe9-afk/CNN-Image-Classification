import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load trained model
model = tf.keras.models.load_model("cnn_model.keras")

# CIFAR-10 classes
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Image path
image_path = "test.jpg"

# Check whether the file exists
print("File exists:", os.path.exists(image_path))
print("File size:", os.path.getsize(image_path), "bytes")

# Open image
image = Image.open(image_path)

# Convert to RGB
image = image.convert("RGB")

# Resize
image = image.resize((32, 32))

# Convert to array
image = np.array(image)

# Normalize
image = image.astype("float32") / 255.0

# Add batch dimension
image = np.expand_dims(image, axis=0)

# Predict
prediction = model.predict(image)

# Get class
predicted_class = np.argmax(prediction[0])

# Get confidence
confidence = prediction[0][predicted_class] * 100

print("Predicted Class:", class_names[predicted_class])
print("Confidence:", round(confidence, 2), "%")