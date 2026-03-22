# Celebrity Face Recognition System

## Overview
This project identifies celebrities from images using image processing and machine learning techniques. It combines OpenCV for face detection, Wavelet Transform for feature extraction, and SVM for classification.

### Categorized Celebrities
The system can classify the following 19 individuals across different sports:
* **Cricket 🏏**: Virat Kohli, MS Dhoni, Rohit Sharma, Hardik Pandya, KL Rahul, Jasprit Bumrah, Ravindra Jadeja, Bhuvneshwar Kumar, Shikhar Dhawan, Dinesh Karthik, Kedar Jadhav, Kuldeep Yadav, Mohammed Shami, Yuzvendra Chahal, Vijay Shankar
* **Tennis 🎾**: Roger Federer, Serena Williams, Maria Sharapova
* **Football ⚽**: Lionel Messi

## Features
* 📷 Upload image for prediction
* 👁️ Face detection using OpenCV
* 🧠 Feature extraction using Wavelet Transform
* 🤖 Celebrity classification using SVM
* 📊 Confidence score display
* 🌐 Simple and interactive web UI

## 🛠️ Tech Stack
* **Programming Language:** Python
* **Web Framework:** Flask
* **UI Framework:** Gradio
* **Face Detection:** OpenCV
* **Feature Extraction:** PyWavelets (Wavelet Transform)
* **Model Classification:** Scikit-learn (Support Vector Machine / SVM)
* **Data Handling:** NumPy & Joblib
* **Frontend:** HTML, CSS, JavaScript

## Project Structure
* `app.py`: The main Flask application file representing the web UI.
* `server/`: Contains backend utility files (like `util.py`) for processing images, face detection, and serving predictions.
* `server/artifacts/`: Contains the pre-computed dictionaries and the saved SVM model.
* `model/`: Jupyter Notebooks (`data_cleaning.ipynb`, `sports_celebrity_classification.ipynb`) for dataset preprocessing and model training.
* `images_dataset/`: The original curated image dataset used to build the recognition model.
* `requirements.txt`: List of Python library dependencies.

## How It Works
1. Upload an image
2. System detects face using OpenCV
3. Extract features using Wavelet Transform
4. Model predicts celebrity using SVM
5. Result is displayed with confidence score

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/shahjainam11/CelebrityFaceRecognition.git
   ```

2. **Navigate to project folder**
   ```bash
   cd CelebrityFaceRecognition
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask server**
   ```bash
   python app.py
   ```

5. **Open browser**
   Navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## 📊 Dataset
* Custom dataset of celebrity images
* Each folder represents one celebrity
* Images are preprocessed and cleaned

## ⚠️ Limitations
* Works only on trained celebrities
* Accuracy depends on dataset quality
* Sensitive to lighting and pose variations

## 🔮 Future Improvements
* Add Deep Learning (CNN) model
* Real-time face detection using webcam
* Expand dataset with more celebrities
* Improve UI with animations

## ⭐ Acknowledgment
Inspired by machine learning and computer vision projects for educational purposes.
