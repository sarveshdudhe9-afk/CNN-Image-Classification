import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("cnn_model.keras")

# CIFAR-10 class names
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

# Load image
image_path = "test.jpg"
image = Image.open(image_path).convert("RGB")

# Resize image
image_resized = image.resize((32, 32))

# Convert to array
image_array = np.array(image_resized).astype("float32") / 255.0

# Add batch dimension
input_image = np.expand_dims(image_array, axis=0)

# Find last convolutional layer
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer
        break

print("Last convolutional layer:", last_conv_layer.name)

# Create a model for the convolutional layer
conv_model = tf.keras.Model(
    model.inputs,
    last_conv_layer.output
)

# Get convolution output
with tf.GradientTape() as tape:

    conv_outputs = conv_model(input_image)

    tape.watch(conv_outputs)

    # Continue through remaining layers
    x = conv_outputs

    start_layer = False

    for layer in model.layers:

        if layer == last_conv_layer:
            start_layer = True
            continue

        if start_layer:
            x = layer(x)

    predictions = x

    predicted_class = tf.argmax(predictions[0])

    class_score = predictions[:, predicted_class]

# Calculate gradients
gradients = tape.gradient(class_score, conv_outputs)

# Average gradients
pooled_gradients = tf.reduce_mean(
    gradients,
    axis=(0, 1, 2)
)

# Create heatmap
conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]

heatmap = tf.squeeze(heatmap)

# Remove negative values
heatmap = tf.maximum(heatmap, 0)

# Normalize
heatmap = heatmap / (
    tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
)

heatmap = heatmap.numpy()

# Show original image
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title(
    "Prediction: "
    + class_names[int(predicted_class)]
)
plt.axis("off")
plt.show()

# Show Grad-CAM
plt.figure(figsize=(6, 6))
plt.imshow(heatmap, cmap="jet")
plt.title("Grad-CAM - Important Areas")
plt.colorbar()
plt.axis("off")
plt.show()

# Print prediction
print(
    "Predicted Class:",
    class_names[int(predicted_class)]
)

print(
    "Confidence:",
    round(
        float(predictions[0][predicted_class]) * 100,
        2
    ),
    "%"
)