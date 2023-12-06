import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
loaded_model = tf.keras.models.load_model('satellite_crop_classification_model.h5')

# Load a single image for prediction
img_path = '' #give path of a single image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Normalize the image
img_array /= 255.0

# Make predictions
predictions = loaded_model.predict(img_array)

# Get the predicted class
predicted_class = np.argmax(predictions)

# Print the predicted class
print(f'The predicted class is: {predicted_class}')
