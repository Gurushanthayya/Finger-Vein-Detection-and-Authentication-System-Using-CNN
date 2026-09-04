# 🩸 VeinGuard — Finger Vein Detection Authentication System

**VeinGuard** is an intelligent, high-security finger vein recognition system. It scans unique vascular patterns beneath the skin to verify identity in real time using Deep Learning and Computer Vision, complete with an easy-to-use web dashboard.

---

## 📌 Project Overview & What It Does

Finger vein patterns are internal biometrics, making them virtually impossible to forge, copy, or spoof. **VeinGuard** provides a complete solution to capture, enhance, extract, and match finger vein patterns.

### Key Highlights:
- 🧠 **Dual AI Engine:** Uses a Deep Convolutional Neural Network (CNN) for fast feature matching, with a Gabor + HOG computer vision fallback system.
- 🔬 **Visual Preprocessing:** Automatically cleans and enhances infrared vein images using contrast adjustment (CLAHE) and vein structure extraction.
- 💻 **Interactive Web Dashboard:** A sleek, modern web UI to register users, test vein verification, and monitor authentication logs live.
- 📁 **Instant Demo Data:** Comes pre-loaded with demo dataset subjects so you can test authentication right away without setup hassle.

---

## 🛠️ Step-by-Step Setup & Installation

### Prerequisites
- **Python 3.9 or higher** installed on your computer.

### Step 1: Open the Project Directory
Open your terminal or PowerShell and navigate into the project folder:
```bash
cd Finger_Vein_Authentication
```

### Step 2: Create & Activate a Virtual Environment
- **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  
### Step 3: Install Required Packages
Install all project dependencies with one command:
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment (Optional)
Create your local environment file:
- **On Windows:** `copy .env.example .env`
---

## 🚀 How to Run and Walk Through the Project

### 1️⃣ Launching the Web Application
Run the main Flask web server:
```bash
python app.py
```

Once running, open your web browser and go to:
👉 **`http://localhost:5000`**

---

### 2️⃣ Walkthrough: Using the Web Dashboard

When you open the web application in your browser, you can explore 3 main features:

1. **🔒 Instant Authentication:**
   - Click **"Select Demo Sample"** to pick a test subject from the dataset.
   - Click **"Verify Identity"** to run instant vein pattern matching.
   - View matching confidence score (%), processing speed (ms), and visual previews of the processed vein images!

2. **👤 User Registration:**
   - Enter a user name and unique User ID.
   - Upload one or more finger vein images (`.png`, `.jpg`, `.bmp`).
   - Click **"Register User"** to extract biometric templates and save the user into the database.

3. **📊 Live System Stats & Auth Logs:**
   - View real-time security statistics (Total Enrolled Users, Verifications, Acceptance Rate).
   - Inspect recent authentication logs with timestamps, matched IDs, and confidence scores.
---

## 📁 Project Structure Quick Map

```
Finger_Vein_Authentication/
├── app.py                      # Main Web Application server & API backend
├── config.py                   # System configuration & threshold settings
├── requirements.txt            # Project dependencies list
├── templates/
│   └── index.html              # Interactive Web Dashboard UI
├── authentication/
│   └── verify.py               # Biometric feature extraction & matching engine
├── preprocessing/
│   ├── preprocess.py           # Image cleaning, CLAHE, & Gabor filtering
│   └── visualize.py            # Preprocessing stage visualizer script
├── database/
│   ├── db.py                   # Database Manager (SQLite / MySQL)
│   └── veinid_db.sqlite        # Local SQLite database
├── models/
│   ├── architecture.py         # CNN Deep Learning architecture definitions
│   └── vein_cnn_base.keras     # Trained AI feature embedding model
├── training/
│   ├── train.py                # Model training script
│   ├── evaluate.py             # Accuracy evaluation script
│   └── plot_graphs.py          # Performance curve plotter (FAR/FRR/EER)
└── docs/                       # Generated charts and visual artifacts
```

---

## ⚡ Tech Stack

- **Backend:** Python 3.9+, Flask
- **AI & Deep Learning:** TensorFlow / Keras, Scikit-Learn
- **Computer Vision:** OpenCV, Pillow, NumPy
- **Frontend UI:** HTML5, Modern CSS3 (Glassmorphism design), Vanilla JavaScript
- **Database:** SQLite3 (Default) / MySQL

