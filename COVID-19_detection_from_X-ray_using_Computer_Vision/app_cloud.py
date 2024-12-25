import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import requests
import base64
import os

# Function to cache the model file
@st.cache_resource
def download_and_load_model():
    """
    Downloads the model file from a given URL, saves it locally, and loads it.
    Returns the loaded model.
    """
    model_url = "https://drive.google.com/uc?export=download&id=1VApI7olygDhgRMJzWgVRjH8BYMxMd9Lu"  # Direct download link
    model_path = "covid19_model.h5"

    # Check if the model file already exists locally
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            response = requests.get(model_url, stream=True)
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

    # Load and return the model
    return load_model(model_path)

# Inject CSS to set a background image and effects
def set_background(image_file):
    """
    Adds a background image to the Streamlit app using CSS.
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{image_file}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Read and encode the background image
with open("24740.jpg", "rb") as bg_file:  # Replace with your image file
    bg_image = bg_file.read()
    encoded_bg_image = base64.b64encode(bg_image).decode()

# Set the background image
set_background(encoded_bg_image)

# Title and description
st.markdown('<h1>COVID-19 X-ray Classifier</h1>', unsafe_allow_html=True)
st.markdown(
    '<p>This app predicts whether an X-ray image shows normal lungs, pneumonia, or COVID-19.</p>',
    unsafe_allow_html=True,
)

# File uploader
st.markdown('<p>Upload an X-ray image:</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# Preprocess the image
def preprocess_image(image):
    img = load_img(image, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load the model
model = download_and_load_model()

# Process uploaded image
if uploaded_file is not None:
    try:
        # Preprocess the uploaded image
        img_array = preprocess_image(uploaded_file)

        # Predict the class
        predictions = model.predict(img_array)
        confidence_scores = predictions[0]
        predicted_class = np.argmax(confidence_scores)
        class_names = ['COVID', 'NORMAL', 'PNEUMONIA']

        # Display confidence scores and prediction
        st.markdown(
            f"<p><strong>Confidence Scores:</strong> COVID: {confidence_scores[0]*100:.2f}%, "
            f"NORMAL: {confidence_scores[1]*100:.2f}%, PNEUMONIA: {confidence_scores[2]*100:.2f}%</p>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<p><strong>Predicted Class:</strong> {class_names[predicted_class]}</p>", unsafe_allow_html=True)

        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)

    except Exception as e:
        st.error(f"Error processing the image: {e}")
