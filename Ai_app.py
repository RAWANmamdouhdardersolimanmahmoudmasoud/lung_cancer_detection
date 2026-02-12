import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import deque

# Define dataset path
dataset_path = r'C:\Users\elkhe\Desktop\archive (1)\lung_colon_image_set\lung_image_sets'
train_dir = dataset_path + r'\train_images'
test_dir = dataset_path + r'\test_images'

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Preprocessing for test images (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Use ResNet50 pre-trained model
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Build the model with additional custom layers
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Set callbacks for early stopping and learning rate reduction
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6)
]

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    callbacks=callbacks
)

# Save the model
model.save('lung_cancer_model_resnet50.keras')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")

# Function to plot accuracy and loss history
def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.show()

# Plot the accuracy and loss
plot_history(history)

# Function to visualize test images with their labels
def plot_sample_images(generator, num_images=5):
    images, labels = next(generator)
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i])
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.show()

# Function to visualize test images with their cancer percentages
def plot_sample_images_with_percentage(generator, model, num_images=5, threshold=0.5):
    images, labels = next(generator)
    predictions = model.predict(images)

    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i])
        pred_prob = predictions[i][0]  # Get the predicted probability
        pred_label = 'Cancer' if pred_prob > threshold else 'No Cancer'
        plt.title(f"{pred_label}\n({pred_prob:.2f})")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Function to implement BFS for threshold exploration
def bfs_predict(predictions, threshold_range=(0.1, 0.9), step=0.1):
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    queue = deque(thresholds)
    results = {}

    while queue:
        threshold = queue.popleft()
        classified = (predictions > threshold).astype("int32")
        results[threshold] = classified

    return results

# Function to implement DFS for threshold exploration
def dfs_predict(predictions, threshold_range=(0.1, 0.9), step=0.1):
    results = {}
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)

    def dfs(threshold_index):
        if threshold_index >= len(thresholds):
            return
        threshold = thresholds[threshold_index]
        classified = (predictions > threshold).astype("int32")
        results[threshold] = classified
        dfs(threshold_index + 1)

    dfs(0)
    return results

# Predict on the test set
test_images, test_labels = next(test_generator)
predictions = model.predict(test_images)

# Explore thresholds using BFS and DFS
bfs_results = bfs_predict(predictions, threshold_range=(0.1, 0.9), step=0.1)
dfs_results = dfs_predict(predictions, threshold_range=(0.1, 0.9), step=0.1)

# Analyze BFS and DFS results
for threshold, classified in bfs_results.items():
    print(f"BFS - Threshold: {threshold}")
    print(confusion_matrix(test_labels, classified))
    print(classification_report(test_labels, classified, target_names=['No Cancer', 'Cancer']))

for threshold, classified in dfs_results.items():
    print(f"DFS - Threshold: {threshold}")
    print(confusion_matrix(test_labels, classified))
    print(classification_report(test_labels, classified, target_names=['No Cancer', 'Cancer']))

# Visualize predictions using optimal threshold
optimal_threshold = 0.5  # Adjust based on BFS/DFS results
classified_optimal = (predictions > optimal_threshold).astype("int32")

# Plot sample predictions
plot_sample_images_with_percentage(test_generator, model, num_images=5, threshold=optimal_threshold)

# Confusion Matrix
cm = confusion_matrix(test_labels, classified_optimal)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Cancer', 'Cancer'], yticklabels=['No Cancer', 'Cancer'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(test_labels, classified_optimal, target_names=['No Cancer', 'Cancer']))
