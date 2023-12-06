# Import necessary libraries
import os
import tensorflow as tf
from tf.keras import layers, models
from tf.keras.preprocessing.image import ImageDataGenerator
from tf.keras.callbacks import EarlyStopping

# Define your data directories
train_dir = '/satalite-crop-dataset/dataset/NAIP/train'
validation_dir = '/satalite-crop-dataset/dataset/NAIP/val'
test_dir = '/satalite-crop-dataset/dataset/NAIP/test'

# Define your image size and batch size
img_size = (128, 128)
batch_size = 32

# Create an ImageDataGenerator for training data with rescaling
train_datagen = ImageDataGenerator(rescale=1./255)
# Create an ImageDataGenerator for validation data with rescaling
validation_datagen = ImageDataGenerator(rescale=1./255)

# Generate training data batches from directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Generate validation data batches from directory
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the CNN model
model = models.Sequential()
# Add a convolutional layer with 32 filters, 3x3 kernel, ReLU activation, and input shape
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
# Add a max pooling layer with a 2x2 pool size
model.add(layers.MaxPooling2D((2, 2)))
# Add another convolutional layer with 64 filters and 3x3 kernel
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Add another max pooling layer with a 2x2 pool size
model.add(layers.MaxPooling2D((2, 2)))
# Add another convolutional layer with 128 filters and 3x3 kernel
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# Add another max pooling layer with a 2x2 pool size
model.add(layers.MaxPooling2D((2, 2)))
# Flatten the output to feed into densely connected layers
model.add(layers.Flatten())
# Add a dense layer with 512 units and ReLU activation
model.add(layers.Dense(512, activation='relu'))
# Add the output layer with softmax activation for multi-class classification
model.add(layers.Dense(6, activation='softmax'))  # Change 6 to the number of classes you have

# Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy metric
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Define early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping]
)

# Save the model
model.save('satellite_crop_classification_model.h5')