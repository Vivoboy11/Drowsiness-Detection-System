# 🚗 Driver Drowsiness & Yawn Detection System

A real-time computer vision application designed to detect driver fatigue and prevent accidents. This system uses the cutting-edge **MediaPipe Tasks API** to track facial landmarks, calculating specific geometric ratios to detect both microsleeps (blinking/closed eyes) and early fatigue signs (frequent yawning).

## ✨ Features
* **Real-Time Eye Tracking:** Calculates the Eye Aspect Ratio (EAR) to detect prolonged eye closure.
* **Predictive Yawn Detection:** Calculates the Mouth Aspect Ratio (MAR) to warn the driver of early fatigue.
* **Asynchronous Audio Alarms:** Uses Python threading to trigger a high-pitched Windows audio alarm without dropping video frames.
* **High Performance:** Utilizes MediaPipe's lightweight `face_landmarker` model for CPU-efficient processing.

---

## 🧠 The Science (How it Works)

The system relies on 468 3D facial landmarks mapped in real-time. 

### 1. Eye Aspect Ratio (EAR)
When the driver's eyes close, the vertical distance between the eyelid landmarks drops to near zero, while the horizontal distance remains constant.

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||}$$

* If $EAR < 0.22$ for 20 consecutive frames, the drowsiness alarm triggers.

### 2. Mouth Aspect Ratio (MAR)
When a person yawns, the vertical distance between the inner lips increases dramatically compared to the horizontal width of the mouth.

$$MAR = \frac{||p_{top} - p_{bottom}||}{||p_{left} - p_{right}||}$$

* If $MAR > 0.50$ for 15 consecutive frames, a yawn warning is displayed.

---

## 🛠️ Tech Stack
* **Python 3.x**
* **OpenCV** (`opencv-python`): For webcam video streaming and image rendering.
* **MediaPipe** (`mediapipe`): Google's framework for high-fidelity facial landmark detection.
* **NumPy** (`numpy`): For fast, vector-based Euclidean distance calculations.

---

## 💻 Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/Vivoboy11/Drowsiness-Detection-System.git](https://github.com/Vivoboy11/Drowsiness-Detection-System.git)
cd Drowsiness-Detection-System

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install opencv-python mediapipe numpy  #dependecies
python main.py  #for running the code
⚠️ Limitations
Requires decent lighting; standard webcams struggle in pitch-black car cabins at night.

Heavy glares on thick-rimmed glasses may occasionally disrupt eye landmark detection.


