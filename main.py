import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# 1. Constants and Indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.22  
CONSEC_FRAMES = 20    
COUNTER = 0           

# 2. EAR Math Function
def calculate_ear(eye_indices, landmarks):
    # Vertical distances
    v1 = np.linalg.norm([landmarks[eye_indices[1]].x - landmarks[eye_indices[5]].x, 
                         landmarks[eye_indices[1]].y - landmarks[eye_indices[5]].y])
    v2 = np.linalg.norm([landmarks[eye_indices[2]].x - landmarks[eye_indices[4]].x, 
                         landmarks[eye_indices[2]].y - landmarks[eye_indices[4]].y])
    # Horizontal distance
    h = np.linalg.norm([landmarks[eye_indices[0]].x - landmarks[eye_indices[3]].x, 
                        landmarks[eye_indices[0]].y - landmarks[eye_indices[3]].y])
    return (v1 + v2) / (2.0 * h)

# 3. Initialize the NEW MediaPipe Tasks API
# Make sure 'face_landmarker.task' is in the same folder!
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# 4. Start Video
cap = cv2.VideoCapture(0)
print("System Starting... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    # Mirror image and convert color
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe's specific Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect faces using the new detector
    detection_result = detector.detect(mp_image)
    
    if detection_result.face_landmarks:
        # Get the first face detected
        face_landmarks = detection_result.face_landmarks[0]
        
        # Calculate EAR
        left_ear = calculate_ear(LEFT_EYE, face_landmarks)
        right_ear = calculate_ear(RIGHT_EYE, face_landmarks)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Alert Logic
        if avg_ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= CONSEC_FRAMES:
                cv2.putText(frame, "!!! DROWSY !!!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            COUNTER = 0
            
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
    cv2.imshow('Drowsiness Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()