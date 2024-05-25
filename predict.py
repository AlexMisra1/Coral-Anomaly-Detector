from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('coral_anomaly_detector.h5')

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize pixel values

    # Make prediction
    prediction = model.predict(img_array)

    # Interpret prediction
    if prediction < 0.5:
        return "Healthy Coral"
    else:
        return "Bleached Coral"

# Example usage
if __name__ == "__main__":
    img_path = 'dataset/Validation/Healthy Coral (raw)/images577.jpg'
    result = predict_image(img_path)
    print("Prediction:", result)
