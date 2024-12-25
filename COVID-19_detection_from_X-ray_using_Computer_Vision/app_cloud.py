import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import requests
import base64
import os
import tempfile

# Function to cache the model file
def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

@st.cache_resource
def download_and_load_model():
    model_id = "1VApI7olygDhgRMJzWgVRjH8BYMxMd9Lu"
    try:
        # Temporary file to store the model
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
            tmp_file_path = tmp_file.name
            
            # Download the model
            download_file_from_google_drive(model_id, tmp_file_path)
            st.success("Model downloaded successfully.")
            
            # Load the model
            model = load_model(tmp_file_path)
            st.success("Model loaded successfully.")
            return model
    except Exception as e:
        st.error(f"Error occurred during model download or loading: {e}")
        return None

# Load the model
model = download_and_load_model()

if model is None:
    st.error("Unable to load the model. Please check the logs and try again.")
else:
    st.success("Model is ready for predictions.")

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
with open("COVID-19_detection_from_X-ray_using_Computer_Vision/24740.jpg", "rb") as bg_file:  # Replace with your image file
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
st.markdown(
    """
    <style>
    .file-uploader-label {
        color: black;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<p class="file-uploader-label">Upload an X-ray image for prediction:</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"], label_visibility="visible")

if uploaded_file:
    st.success("File uploaded successfully!")

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
