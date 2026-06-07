import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from collections import Counter

st.title("FashionInsightAI")

st.write("Welcome to FashionInsightAI!")
st.write("AI features are currently under development.")

@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

# Color Analysis Function
def get_dominant_color(image):
    small_image = image.resize((50, 50))

    pixels = list(small_image.getdata())

    rounded_pixels = [
        (r // 25 * 25, g // 25 * 25, b // 25 * 25)
        for r, g, b in pixels
    ]

    most_common = Counter(rounded_pixels).most_common(1)[0][0]

    return most_common

def get_color_name(rgb):
    r, g, b = rgb

    if r > 200 and g > 200 and b > 200:
        return "White"

    elif r < 60 and g < 60 and b < 60:
        return "Black"

    elif r > 180 and g < 100 and b < 100:
        return "Red"

    elif r < 100 and g > 180 and b < 100:
        return "Green"

    elif r < 100 and g < 100 and b > 180:
        return "Blue"

    elif r > 180 and g > 180 and b < 120:
        return "Yellow"

    elif r > 150 and g > 100 and b < 80:
        return "Brown"

    elif r > 150 and g > 150 and b > 150:
        return "Light Gray"

    else:
        return "Unknown"

uploaded_file = st.file_uploader(
    "Upload a fashion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    resized_image = image.resize((224, 224))
    image_array = np.array(resized_image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    st.subheader("AI Analysis Result")

    for _, label, confidence in decoded_predictions:
        st.write(f"Prediction: {label}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
        st.write("---")

    dominant_color = get_dominant_color(image)

    color_name = get_color_name(dominant_color)

    st.subheader("Dominant Color")

    st.write(f"Color: {color_name}")
    st.write(f"RGB: {dominant_color}")      


    