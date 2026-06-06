import streamlit as st
from PIL import Image

st.title("FashionInsightAI")

st.write("Welcome to FashionInsightAI!")

uploaded_file = st.file_uploader(
    "Upload a fashion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.success("Image uploaded successfully!")

    st.write("Category: Coming Soon")
    st.write("Confidence: Coming Soon")