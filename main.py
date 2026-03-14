import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import threading
import winsound

# 1. Constants and Indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Inner lip points: Left(78), Right(308), Top(13), Bottom(14)
INNER_LIPS = [78, 308, 13, 14] 

# Thresholds
EAR_THRESHOLD = 0.22  
CONSEC_FRAMES = 20    
COUNTER = 0           
ALARM_ON = False      

MAR_THRESHOLD = 0.5   # If ratio > 0.5, mouth is wide open
YAWN_FRAMES = 15      # Frames required to confirm a yawn
YAWN_COUNTER = 0

# 2. Audio Alarm Thread
def sound_alarm():
    global ALARM_ON
    while ALARM_ON:
        winsound.Beep(2500, 500)

# 3. Math Functions (EAR and MAR)
def calculate_ear(eye_indices, landmarks):
    v1 = np.linalg.norm([landmarks[eye_indices[1]].x - landmarks[eye_indices[5]].x, 
                         landmarks[eye_indices[1]].y - landmarks[eye_indices[5]].y])
    v2 = np.linalg.norm([landmarks[eye_indices[2]].x - landmarks[eye_indices[4]].x, 
                         landmarks[eye_indices[2]].y - landmarks[eye_indices[4]].y])
    h = np.linalg.norm([landmarks[eye_indices[0]].x - landmarks[eye_indices[3]].x, 
                        landmarks[eye_indices[0]].y - landmarks[eye_indices[3]].y])
    return (v1 + v2) / (2.0 * h)

def calculate_mar(mouth_indices, landmarks):
    # Vertical distance (Top lip to Bottom lip)
    v = np.linalg.norm([landmarks[mouth_indices[2]].x - landmarks[mouth_indices[3]].x, 
                        landmarks[mouth_indices[2]].y - landmarks[mouth_indices[3]].y])
    # Horizontal distance (Left corner to Right corner)
    h = np.linalg.norm([landmarks[mouth_indices[0]].x - landmarks[mouth_indices[1]].x, 
                        landmarks[mouth_indices[0]].y - landmarks[mouth_indices[1]].y])
    return v / h

# 4. Initialize MediaPipe
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# 5. Start Video Loop
cap = cv2.VideoCapture(0)
print("System Starting... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    if detection_result.face_landmarks:
        face_landmarks = detection_result.face_landmarks[0]
        
        # --- EYE LOGIC (Drowsiness) ---
        left_ear = calculate_ear(LEFT_EYE, face_landmarks)
        right_ear = calculate_ear(RIGHT_EYE, face_landmarks)
        avg_ear = (left_ear + right_ear) / 2.0
        
        if avg_ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    t = threading.Thread(target=sound_alarm)
                    t.daemon = True
                    t.start()
                cv2.putText(frame, "!!! WAKE UP !!!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            COUNTER = 0
            ALARM_ON = False
            
        # --- MOUTH LOGIC (Yawning) ---
        mar = calculate_mar(INNER_LIPS, face_landmarks)
        if mar > MAR_THRESHOLD:
            YAWN_COUNTER += 1
            if YAWN_COUNTER >= YAWN_FRAMES:
                cv2.putText(frame, "- YAWN WARNING: TAKE A BREAK -", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            YAWN_COUNTER = 0

        # Display Stats
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
    cv2.imshow('Drowsiness Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()