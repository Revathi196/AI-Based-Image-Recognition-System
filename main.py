# main.py

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

class ImageRecognitionSystem:
    def __init__(self):
        # Load pre-trained MobileNetV2 model
        self.model = MobileNetV2(weights='imagenet')

    def load_and_preprocess_image(self, img_path):
        """Load an image and preprocess it for prediction."""
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def predict_image(self, img_path):
        """Predict the class of the image."""
        img_array = self.load_and_preprocess_image(img_path)
        predictions = self.model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions

    def display_predictions(self, img_path):
        """Display top 3 predictions for the image."""
        print(f"Predictions for the image {img_path}:")
        predictions = self.predict_image(img_path)
        for idx, (imagenet_id, label, score) in enumerate(predictions):
            print(f"{idx + 1}. {label}: {score * 100:.2f}%")

def main():
    recognition_system = ImageRecognitionSystem()

    while True:
        print("\nAI-Based Image Recognition System")
        print("1. Upload Image and Predict")
        print("2. Exit")
        choice = input("Enter your choice (1-2): ").strip()

        if choice == '1':
            img_path = input("Enter the image file path: ").strip()
            if os.path.exists(img_path):
                recognition_system.display_predictions(img_path)
            else:
                print("Invalid image path. Please try again.")
        elif choice == '2':
            print("Exiting the system. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
