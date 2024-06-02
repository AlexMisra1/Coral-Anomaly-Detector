import tensorflow as tf
from tensorflow import keras

# Load your Keras model
model = keras.models.load_model('coral_anomaly_detector.h5')

# Save the model in TensorFlow SavedModel format
model.export('saved_model')



