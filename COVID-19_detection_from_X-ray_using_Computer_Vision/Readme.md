# COVID-19 Detection from X-ray Using Computer Vision

A computer vision-based project leveraging deep learning techniques to detect COVID-19 from chest X-ray images. The project demonstrates the potential of AI in healthcare diagnostics.

---

## 📂 Project Structure

- **`app_cloud.py`**: The main application file for deployment in a cloud platform such as **Streamlit Cloud**.
- **`app_local.py`**: The main application file for deployment in a local machine using **Anaconda Prompt & Navigator**.
- **`model.h5`**: Pre-trained model for COVID-19 detection.
- **`Dataset`**: X-ray images categorized into `COVID`,`NORMAL` and `PNEUMONIA`.

---

## 📥 Downloads

### Pre-trained Model File
Download the pre-trained model file from the link below:  
[**Model File - Google Drive**](https://drive.google.com/file/d/1VApI7olygDhgRMJzWgVRjH8BYMxMd9Lu/view?usp=sharing)

### Dataset for Custom Training
Download the dataset for further training from the following link:  
[**Dataset - Google Drive**](https://drive.google.com/file/d/1tbxLmhSt5lJm_gIrb6SQ4KH4cBe9_cI7/view?usp=sharing)

---

## 🚀 How to Run the Application

# Steps to Run

## Download the Pre-trained Model:
- Save the downloaded `covid19_model.h5` file in the root directory of the project.  
- Ensure the file path in `app_local.py` matches the location of the model file.  

## Run the Application:
1. Execute the following command to start the application:  
   ```bash
   streamlit run app_local.py


### Prerequisites
1. Install Python 3.8 or later.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
3. Anaconda Navigator and Anaconda Prompt (**Optional )

The app will open in your default web browser.

---

# 🔧 Planned Improvements

## Hosting
- Host the application on **Streamlit Cloud** or **Heroku** for easy access.  
- Current bandwidth limitations in my GitHub repository prevent hosting directly.

## Model Optimization
- **Compression**: Use HDF5 compression features to reduce the model file size.  
- **Quantization**: Convert model weights from 32-bit floats to 16-bit floats for efficient deployment.

## External Hosting
- Host the optimized model file on external storage for faster downloads and seamless integration.

---

# 📈 Features

## COVID-19 Detection
- Classifies chest X-ray images into `COVID`, `NORMAL` and `PNEUMONIA` categories.  

## Custom Training
- Provides a dataset for further training and improvement.  

## User Interface
- Easy-to-use web interface built using **Streamlit**.

---
## 📸Example output (Screenshots):

![image](https://github.com/user-attachments/assets/6f479793-dbe7-47a6-9384-b1e2c2ba6c01)

![image](https://github.com/user-attachments/assets/d3825d29-9e98-429c-a0d1-d0626e04f64c)

![image](https://github.com/user-attachments/assets/18fc7ec2-6a0a-4efa-bf8f-0d931d78d753)



### 🎥App Demo (Video)▶️ :

📹 Watch the demo of the app in action, Click the below thumbnail to see the live demo.
👇🏻👇🏻👇🏻

[![Watch the demo](https://img.youtube.com/vi/woGspTurIEc/0.jpg)](https://youtu.be/woGspTurIEc)


---

# 🛠️ Customization

## Training
- Use the provided dataset to train the model on additional data.  
- Modify the architecture in `app_local.py` for specific use cases.

## Feature Enhancements
- Add visualizations for model predictions.  
- Implement explainability features like **Grad-CAM** or **SHAP** to provide insights into model decisions.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---
# 🛑 Known Issues and Notes

- **Model File**: Ensure the downloaded `model.h5` file matches the required format and file path.  
- **Dataset Structure**: Organize the dataset into subfolders for **training and testing** contains the classes: `COVID`, `NORMAL`, `PNEUMONIA` for training.

---

# 💻 Technologies Used
- **Python**: Programming language.  
- **Streamlit**: Framework for building the user interface.  
- **TensorFlow/Keras**: Libraries for deep learning.  
- **OpenCV**: Image processing library.

---

# 📧 Contact
**Saravanan Veerakumar**
- GitHub : https://github.com/Saravanan-Veerakumar
- LinkedIn : https://www.linkedin.com/in/saravanan-veerakumar/

---

# ⚠️ Disclaimer
This project is for educational purposes only. It is not intended for clinical use and should not replace medical diagnosis.

