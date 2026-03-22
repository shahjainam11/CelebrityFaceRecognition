# Celebrity Face Recognition System

## Overview
This project identifies celebrities from images using deep learning and vector matching techniques. It leverages MTCNN for precise face detection, FaceNet (via DeepFace) for feature extraction (creating 128-dimensional embeddings), and Spotify's Annoy Index for lightning-fast facial similarity matching and classification.

### Categorized Celebrities
The system can classify the following 19 individuals across different sports:
* **Cricket 🏏**: Virat Kohli, MS Dhoni, Rohit Sharma, Hardik Pandya, KL Rahul, Jasprit Bumrah, Ravindra Jadeja, Bhuvneshwar Kumar, Shikhar Dhawan, Dinesh Karthik, Kedar Jadhav, Kuldeep Yadav, Mohammed Shami, Yuzvendra Chahal, Vijay Shankar
* **Tennis 🎾**: Roger Federer, Serena Williams, Maria Sharapova
* **Football ⚽**: Lionel Messi

## Features
* 📷 Live webcam capture and image upload support
* 👁️ Precise face detection and cropping using MTCNN
* 🧠 Feature extraction using FaceNet (DeepFace)
* 🤖 Fast celebrity vector-matching using Spotify Annoy
* 📊 Multi-model confidence score display and probability comparison
* 🌐 Modern, glassmorphism-styled interactive web UI using Gradio

## 🛠️ Tech Stack
* **Programming Language:** Python 3
* **Web/UI Framework:** Gradio (Acts as the HTTP Web Server & Interactive Dashboard)
* **Face Analytics / Detection:** DeepFace (MTCNN Model)
* **Feature Extraction:** FaceNet (128-Dimensional Embeddings)
* **Model Classification:** Scikit-learn (SVM / Random Forest)
* **Image Processing:** OpenCV & PIL (Image resizing, cropping, and color conversion)
* **Data Handling & Math:** NumPy & Joblib
* **Data Visualization:** Matplotlib & Seaborn
* **Development IDEs:** Jupyter Notebook, VS Code, & PyCharm

## Project Structure
* `app.py`: The main application code running the Gradio web server and UI.
* `server/`: Contains backend utility files (like `util.py`) for processing images, face detection, and serving predictions.
* `server/artifacts/`: Contains the pre-computed dictionaries and the saved SVM model.
* `model/`: Jupyter Notebooks (`data_cleaning.ipynb`, `sports_celebrity_classification.ipynb`) for dataset preprocessing and model training.
* `images_dataset/`: The original curated image dataset used to build the recognition model.
* `requirements.txt`: List of Python library dependencies.

## How It Works
1. Upload an image or capture via live webcam.
2. The system locates and crops the face using **MTCNN**.
3. It extracts a 128-dimensional facial embedding using the **FaceNet CNN**.
4. The **Annoy Index** calculates the angular distance against the database of known celebrities to find the closest match.
5. The result is displayed alongside the celebrity's details, sport icons, and visualization of the matching confidence.

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

4. **Run the application server**
   ```bash
   python app.py
   ```

5. **Open browser**
   Navigate to:
   ```
   http://127.0.0.1:7860/
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
* Expand dataset with more sports and international celebrities
* Optimize database size and scaling for edge devices
* Add real-time video stream continuous tracking
* Support for multiple face detection in a single group photo

## ⭐ Acknowledgment
Inspired by machine learning and computer vision projects for educational purposes.
