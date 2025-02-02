# AI-Based Image Recognition System

## Overview
This is an AI-based image recognition system built using TensorFlow and Keras. It uses a pre-trained MobileNetV2 model to classify images. Users can upload an image, and the system will predict the top 3 possible objects in the image based on the model's capabilities.

## Features
- **Image Upload**: Allows users to upload an image.
- **Image Prediction**: Classifies the image into one of the objects that the model recognizes (from ImageNet classes).
- **Top 3 Predictions**: Shows the top 3 most likely predictions with confidence percentages.

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- PIL (Pillow)
  
### Install dependencies:
```bash
pip install tensorflow numpy pillow
