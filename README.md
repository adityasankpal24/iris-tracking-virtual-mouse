# Iris Tracking Virtual Mouse 👁️🖱️

## 🚀 Overview

This project implements a **real-time contactless virtual mouse system** using computer vision techniques.
The system enables users to control the cursor using **eye movements (iris tracking)** and perform actions like clicks using **blink detection**, eliminating the need for traditional input devices.

---

## 🎯 Key Features

* 👁️ **Iris-Based Cursor Control** – Move cursor using eye gaze
* 👁️‍🗨️ **Blink Detection (EAR Method)** – Perform left/right click actions
* 🎯 **Head Movement Compensation** – Improves tracking stability
* ⚡ **Real-Time Performance** – ~120–180 ms response latency
* 💻 **Hardware-Free System** – Works using a standard webcam

---

## 🛠 Tech Stack

* **Python**
* **OpenCV**
* **MediaPipe (Face Mesh)**
* **NumPy**
* **PyAutoGUI**

---

## ⚙️ System Architecture

1. Capture real-time video using webcam
2. Detect facial landmarks using MediaPipe Face Mesh
3. Track iris position for cursor movement
4. Calculate Eye Aspect Ratio (EAR) for blink detection
5. Map eye movement to screen coordinates
6. Apply smoothing to reduce jitter

---

## 🧠 Core Concepts Used

* Computer Vision
* Facial Landmark Detection
* Gaze Estimation
* Eye Aspect Ratio (EAR)
* Real-Time Image Processing
* Human-Computer Interaction (HCI)

---

## 📷 Output

👉 Add screenshots or demo images here
*(Example: cursor movement, blink detection, etc.)*

---

## 📊 Performance

* Accuracy: ~84%
* Latency: 120–180 ms
* Frame Rate: ~20–25 FPS

---

## 🔧 How to Run

1. Clone the repository

```bash
git clone https://github.com/your-username/iris-tracking-virtual-mouse.git
```

2. Install dependencies

```bash
pip install opencv-python mediapipe numpy pyautogui
```

3. Run the program

```bash
python main.py
```

---

## 🔮 Future Improvements

* Improve accuracy using deep learning models
* Adaptive calibration for different users
* Better performance under varying lighting conditions
* Multi-monitor support

---

## 📌 Applications

* Assistive technology for physically disabled users
* Touchless interfaces (public systems, healthcare)
* AR/VR interaction systems

---

## 👨‍💻 Author

Aditya Sankpal

---
