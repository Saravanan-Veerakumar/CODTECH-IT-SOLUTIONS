import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import base64

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
        .fade-in-out {{
            animation: fadeInOut 1s infinite;
        }}
        @keyframes fadeInOut {{
            0% {{ opacity: 0; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0; }}
        }}
        .boom-in-out {{
            animation: boomInOut 1s ease-in-out;
        }}
        @keyframes boomInOut {{
            0% {{ transform: scale(0.5); opacity: 0; }}
            50% {{ transform: scale(1.2); opacity: 1; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
        .fade-in-out-text {{
            animation: fadeInOutText 1s infinite;
        }}
        @keyframes fadeInOutText {{
            0% {{ opacity: 0; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0; }}
        }}
        .normal-class {{
            color: darkgreen;
            font-weight: bold;
        }}
        .pneumonia-class {{
            color: orange;
            font-weight: bold;
        }}
        .covid-class {{
            color: red;
            font-weight: bold;
        }}
        .image-preview {{
            width: 23%;
            margin-left: 0;
            margin-right: auto;
            display: block;
        }}
        .image-container {{
            text-align: right;
        }}
        .prediction-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load your trained model
model = load_model("covid19_model.h5")

# Read and encode the background image
with open("24740.jpg", "rb") as bg_file:  # Replace with your image file
    bg_image = bg_file.read()
    encoded_bg_image = base64.b64encode(bg_image).decode()

# Set the background image
set_background(encoded_bg_image)

# Streamlit file uploader and text with left alignment
st.markdown(
    """
    <style>
    .title-text {
        color: #000000;
        font-size: 36px;
        text-align: left;
        font-weight: bold;
    }
    .description-text {
        color: #000000;
        font-size: 18px;
        text-align: left;
    }
    .file-uploader-text {
        color: #000000;
        font-size: 18px;
        text-align: left;
    }
    .stFileUploader {
        display: block;
        text-align: left;
    }
    .engineered-text {
        color: #000000;
        font-size: 16px;
        text-align: left;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom title and description with left alignment
st.markdown('<p class="title-text">COVID-19 X-ray Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="description-text">This is a simple app to predict whether a X-ray image shows normal lungs, pneumonia, or COVID.</p>', unsafe_allow_html=True)

# Engineered by text placed below the title
st.markdown('<p class="engineered-text">**Engineered by SARAVANAN VEERAKUMAR as a part of internship program with CODTECH IT Solutions.</p>', unsafe_allow_html=True)

# Add left-aligned file uploader label
st.markdown('<p class="file-uploader-text">Choose an image...</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:
    try:
        # Preprocess the uploaded image
        img_array = preprocess_image(uploaded_file)

        # Predict the class and get confidence scores
        predictions = model.predict(img_array)
        confidence_scores = predictions[0]  # Extract confidence scores for all classes
        predicted_class = np.argmax(confidence_scores)
        class_names = ['COVID', 'NORMAL', 'PNEUMONIA']  # Replace with your actual class names

        # Create two merged columns and one separate column
        col1_2, col3 = st.columns([3, 1])

        # First merged column: Display confidence scores and predicted class
        confidence_str = f"COVID: {confidence_scores[0]*100:.2f}%, NORMAL: {confidence_scores[1]*100:.2f}%, PNEUMONIA: {confidence_scores[2]*100:.2f}%"
        col1_2.markdown(
            f"""
            <style>
            .confidence-text {{
                font-size: 16px;  /* Slightly larger font for visibility */
                color: black;
                font-family: Arial, sans-serif;
            }}
            .prediction-text {{
                font-size: 20px;
                font-weight: bold;
                margin-top: 20px;
            }}
            </style>
            <p class="confidence-text"><strong>Confidence Scores:</strong></p>
            <p class="confidence-text">{confidence_str}</p>
            """,
            unsafe_allow_html=True
        )

        if class_names[predicted_class] == 'NORMAL':
            col1_2.markdown(
                '<p class="fade-in-out-text normal-class prediction-text">Predicted Class: NORMAL</p>',
                unsafe_allow_html=True
            )
        elif class_names[predicted_class] == 'PNEUMONIA':
            col1_2.markdown(
                '<p class="fade-in-out-text pneumonia-class prediction-text">Predicted Class: PNEUMONIA</p>',
                unsafe_allow_html=True
            )
        elif class_names[predicted_class] == 'COVID':
            col1_2.markdown(
                '<p class="fade-in-out-text covid-class prediction-text">Predicted Class: COVID</p>',
                unsafe_allow_html=True
            )

        # Third column: Display the image preview
        
        image_url = uploaded_file.getvalue()
        col3.markdown(
            f"""
            <style>
            .image-container {{
                text-align: center;
            }}
            .image-preview {{
                width: 30%;  /* Adjust width to fit the column */
                margin-top: 20px;
                border: 2px solid #ccc;
                border-radius: 8px;
            }}
            </style>
            <div class="image-container">
                <img src="data:image/png;base64,{base64.b64encode(image_url).decode()}" class="image-preview" />
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Error processing the image: {e}")
