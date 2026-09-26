import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build CNN model
model = tf.keras.Sequential([
    
    # Convolutional Layer 1
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu",
        input_shape=(32, 32, 3)
    ),
    
    # Pooling Layer 1
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convolutional Layer 2
    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),

    # Pooling Layer 2
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten
    tf.keras.layers.Flatten(),

    # Fully Connected Layer
    tf.keras.layers.Dense(128, activation="relu"),

    # Output Layer
    tf.keras.layers.Dense(10, activation="softmax")
])

# Display model architecture
model.summary()

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Accuracy Percentage:", test_accuracy * 100)

# Plot training and validation accuracy
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("CNN Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()


# Plot training and validation loss
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("CNN Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

# ==============================
# STEP 10 - CONFUSION MATRIX
# ==============================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Make predictions on test images
y_pred = model.predict(x_test)

# Convert predicted probabilities into class numbers
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert actual one-hot labels into class numbers
y_true_classes = np.argmax(y_test, axis=1)

# Create confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Class names
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

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(xticks_rotation=45)

plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ==============================
# STEP 11 - SAVE THE MODEL
# ==============================

# Save the trained CNN model
model.save("cnn_model.keras")

print("CNN model saved successfully!")