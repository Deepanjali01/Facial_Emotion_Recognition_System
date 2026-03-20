# Facial Emotion Recognition and Mental State Estimation 🎭🧠

This project uses **Deep Learning (MobileNetV2)** to recognize human facial emotions in real time and estimate their corresponding mental state (e.g., relaxed, stressed, anxious).  
It integrates emotion recognition with rule-based stress analysis for lightweight inference.

---

## 📚 Features
- Real-time emotion detection using a webcam  
- Mental state estimation derived from detected emotion  
- Model trained on the **FER2013** dataset  
- Optimized with **MobileNetV2** for better speed and accuracy  
- Evaluation and visualization modules included  
- Modular and reusable code structure  

---

## 🧠 Project Structure
~~~bash
Facial Emotion Recognition System/
│
├── train_model_optimized.py        # Trains and fine-tunes the MobileNetV2 model
├── realtime_demo_optimized.py      # Real-time webcam emotion detection demo
├── mental_state_module.py          # Maps emotion → mental state and stress score
├── requirements.txt                # Required Python dependencies
├── mobilenetv2_fer_final_optimized.h5  # Trained model weights
└── Figure_1.png                    # Example output visualization

~~~

---

## ⚙️ Installation

### 1️⃣ Clone the repository
~~~bash
git clone https://github.com/Deepanjali01/FacialEmotion_MentalState.git
cd FacialEmotion_MentalState
~~~
### 2️⃣ Create a virtual environment (optional but recommended)
~~~bash
python -m venv venv
venv\Scripts\activate   # On Windows
~~~

### 3️⃣ Install dependencies
~~~bash
pip install -r requirements.txt
~~~

### 📊 Dataset — FER-2013

The model is trained and validated on the FER-2013 (Facial Expression Recognition) dataset from Kaggle.
It contains 35,887 grayscale images (48×48 pixels) categorized into 7 emotions:
~~~
Angry 😠

Disgust 🤢

Fear 😨

Happy 😀

Sad 😢

Surprise 😲

Neutral 😐
~~~
Each image was preprocessed and augmented during training to improve generalization.


### 🧩 Train the Model

Ensure the FER-2013 dataset is extracted properly and update dataset paths inside 
~~~
train_model_optimized.py.

~~~
Then run:
~~~bash
python train_model_optimized.py
~~~
### 🎥 Run the Real-Time Demo

Connect your webcam and run:

~~~bash
python realtime_demo_optimized.py
~~~

Press Q to exit the window.

### 🧮 Mental State Logic
~~~
| Emotion            | Inferred Mental State | Stress Level |
| ------------------ | --------------------- | ------------ |
| Happy, Surprise    | Relaxed / Positive    | Low          |
| Sad, Fear, Disgust | Stressed / Anxious    | High         |
| Angry              | Possible Stress       | Moderate     |
| Neutral            | Neutral / Low Mood    | Moderate     |
~~~

### 📈 Results and Performance

Model Architecture: MobileNetV2 (fine-tuned on FER-2013)

Training Accuracy: ~90%

Validation Accuracy: ~75–80% (depending on augmentation and class balance)

Inference Speed: Real-time (25–30 FPS on GTX 1650 GPU)

Loss Function: Categorical Crossentropy

Optimizer: Adam

### Example detection output:
~~~bash
Emotion: Happy (97.2%) | Mental State: Relaxed / Positive | Stress: Low
~~~
### 📦 Requirements

See requirements.txt
~~~
Main dependencies:

TensorFlow 2.9.1

OpenCV 4.7

NumPy

Matplotlib

Pandas

scikit-learn
~~~
### 🔮Future Improvements

Integration of transformer-based models or Vision Transformers (ViT) for improved accuracy.

Development of a user dashboard to display emotion trends and stress variations over time.

Deployment on web or mobile platforms using Streamlit or Flask.

Multi-face recognition in group environments.

Incorporation of multimodal data (voice tone, heart rate, posture) for more robust mental state estimation.

Fine-tuning using AffectNet or RAF-DB for better generalization across demographics.

### 🧑‍💻Author

Deepanjali Singh
GitHub: @Deepanjali01

### 📜License

This project is open-source and may be used for research, learning, and non-commercial purposes.


---

### ✅ To add it:
1. In VS Code → **New File → name it `README.md`**
2. Paste this content
3. Save it  
4. Then in terminal:
   ```bash
  
   
   
