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
from ultralytics import YOLO
import cv2
import math

st.title("FashionInsightAI")

st.write("Welcome to FashionInsightAI!")
st.write("AI features are currently under development.")

@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")
@st.cache_resource
def load_detection_model():
    return YOLO("yolov8n.pt")

model = load_model()
detection_model = load_detection_model()

# Color Analysis Function
def get_dominant_color(image):
    small_image = image.resize((80, 80))
    pixels = list(small_image.getdata())

    filtered_pixels = []

    for r, g, b in pixels:
        # white background ignore
        if r > 220 and g > 220 and b > 220:
            continue

        # black shadow/background ignore
        if r < 25 and g < 25 and b < 25:
            continue

        filtered_pixels.append((r, g, b))

    if len(filtered_pixels) == 0:
        filtered_pixels = pixels

    rounded_pixels = [
        (r // 25 * 25, g // 25 * 25, b // 25 * 25)
        for r, g, b in filtered_pixels
    ]

    most_common = Counter(rounded_pixels).most_common(1)[0][0]

    return most_common

COLOR_MAP = {
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Gray": (128, 128, 128),
    "Red": (255, 0, 0),
    "Green": (0, 128, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Orange": (255, 165, 0),
    "Brown": (139, 69, 19),
    "Dark Brown": (101, 67, 33),
    "Beige": (245, 245, 220),
    "Pink": (255, 192, 203),
    "Purple": (128, 0, 128),
    "Navy": (0, 0, 128),
    "Olive": (128, 128, 0)
}

def get_color_name(rgb):
    min_distance = float("inf")
    closest_color = None

    for color_name, color_rgb in COLOR_MAP.items():
        distance = math.sqrt(
            (rgb[0] - color_rgb[0]) ** 2 +
            (rgb[1] - color_rgb[1]) ** 2 +
            (rgb[2] - color_rgb[2]) ** 2
        )

        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    return closest_color

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

 
def detect_clothing_area(image):
    results = detection_model(image)
    result = results[0]

    best_box = None
    best_confidence = 0
    best_name = None

    # target_objects = ["person", "handbag", "tie", "backpack", "suitcase", "shoes", "dress", "skirt", "pants", "shirt", "jacket", "sne"
    # "boots", "sandals", "socks"]

    target_objects = list(detection_model.names.values())

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        name = detection_model.names[class_id]

        if name in target_objects and confidence > best_confidence:
            best_confidence = confidence
            best_box = box.xyxy[0].tolist()
            best_name = name

    if best_box is None:
        return image, None, None, None

    x1, y1, x2, y2 = map(int, best_box)

    clothing_area = image.crop((x1, y1, x2, y2))

    image_with_box = np.array(image).copy()

    cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return Image.fromarray(image_with_box), clothing_area, best_name, best_confidence

uploaded_file = st.file_uploader(
    "Upload a fashion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_with_box, clothing_area, detected_name, detection_confidence = detect_clothing_area(image)

    st.subheader("Detected Clothing Area")
    st.image(image_with_box, caption="Detected Area", use_container_width=True)

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

    if clothing_area is not None:
        dominant_color = get_dominant_color(clothing_area)
        # st.write(f"Detected object: {detected_name}")  sometimes it can show wrong detection, so I hide it for now
        # st.write(f"Detection confidence: {detection_confidence * 100:.2f}%")
    else:
        st.warning("No clothing area detected. Full image was used for color analysis.")
        dominant_color = get_dominant_color(image)

    color_name = get_color_name(dominant_color)

    st.subheader("Dominant Color")

    hex_color = rgb_to_hex(dominant_color)

    st.write(f"Color: {color_name}")

    st.markdown(
        f"""
        <div style="
            width: 120px;
            height: 60px;
            background-color: {hex_color};
            border: 2px solid #333;
            border-radius: 8px;
            margin-top: 10px;
        "></div>
        """,
        unsafe_allow_html=True
    )     


    