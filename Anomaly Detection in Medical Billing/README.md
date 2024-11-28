# Anomaly Detection in Billing Data using GANs

This project leverages Generative Adversarial Networks (GANs) to detect anomalies in medical billing data. It includes a training pipeline for building GAN models and an interactive Streamlit-based web application for anomaly detection.

---

## 🚀 Features
- **GAN-based Anomaly Detection**: Uses a trained discriminator to identify anomalies in billing data.
- **Streamlit Web App**: An interactive app for uploading billing data, setting thresholds, and detecting anomalies.
- **Customizable Training**: Flexible GAN training script for synthetic or real billing datasets.
- **Data Preprocessing**: Automatically scales and encodes input data for compatibility with models.

---

## 📂 Project Structure

anomaly_detection_billing 

├── data 

    └── example_data.csv # Example dataset for testing 
  
├── models 

    ├── gan_generator.h5 # Pre-trained generator model 
  
    └── gan_discriminator.h5 # Pre-trained discriminator model 
  
├── scripts 

    └── train_gan.py # Python script for training GANs 
  
├── streamlit_app 

    └── anomaly_detection_app.py # Streamlit app for anomaly detection
  
├── README.md # Project documentation 

├── requirements.txt # Python dependencies 

└── LICENSE # License file



---

## 🛠 Installation

### Prerequisites
Ensure you have Python 3.8+ installed. Use a virtual environment to manage dependencies.

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/saravanan-veerakumar/anomaly-detection-billing.git
   cd anomaly-detection-billing
   
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Set up models:**

- If training from scratch, use the training script (scripts/train_gan.py).
  
- Alternatively, download pre-trained models from the models/ directory.


📊 Usage
**1. Train the GAN Models**
Train the GAN using synthetic or custom billing data:

```bash
python scripts/train_gan.py
```
This generates gan_generator.h5 and gan_discriminator.h5 in the models/ folder.

**2. Launch the Streamlit App**

Run the Streamlit application to detect anomalies:

```bash
streamlit run streamlit_app/anomaly_detection_app.py
```
**App Features**
- **Upload CSV File:** Upload a billing dataset with columns like ClaimAmount, ICD10_Code, CPT_Code, etc.
- **Set Threshold:** Adjust the score threshold for anomaly detection.
- **View Results:** Identify and analyze anomalies in the uploaded data.
  
**3. Example Dataset**
Use the example dataset provided in the data/ folder (example_data.csv) to test the application.

**📈 GAN Architecture**

**Generator**
-  Fully connected neural network.
-  Input: Noise vector of size 100.
-  Output: Synthetic billing data with scaled features.

**Discriminator**

-  Fully connected neural network.
-  Input: Billing data (real or synthetic).
-  Output: Probability of being real (score between 0 and 1).

**Loss Functions**

-  Discriminator: Binary Crossentropy.
-  GAN: Binary Crossentropy (via discriminator feedback).

**📊 Anomaly Detection**

**How It Works**

1.  The discriminator assigns scores to each data point.
2.  Data points with scores below the threshold are flagged as anomalies.

**Visualization**
The app provides a histogram of discriminator scores to help set appropriate thresholds.

**📷 Screenshots**

**Streamlit Web App**

![image](https://github.com/user-attachments/assets/e08f0ccd-6b11-480d-a204-0c999a3dd007)

![image](https://github.com/user-attachments/assets/8e7186b6-1b6e-473a-8fa9-4e6bebd76249)

![image](https://github.com/user-attachments/assets/704ebe29-6399-4f9c-bee9-db3370b2f8c4)

**📜 License**

This project is licensed under the MIT License.

**🤝 Contributing**

Contributions are welcome! To contribute:

  1. Fork the repository.
  2. Create a feature branch (git checkout -b feature-name).
  3. Commit changes (git commit -m "Add feature").
  4. Push to the branch (git push origin feature-name).
  5. Open a pull request.

**✉️ Contact**

For questions or feedback, contact:

-  Your Name: SARAVANAN VEERAKUMAR
-  Email: saravananv1925@gmail.com
-  GitHub Profile : https://github.com/Saravanan-Veerakumar

**🌟 Acknowledgements**
-  TensorFlow for model development.
-  Streamlit for building the web application.
-  Scikit-learn for data preprocessing utilities.




   


