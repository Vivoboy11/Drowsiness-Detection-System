import cv2
import mediapipe as mp
from scipy.spatial import distance as dist

def calculate_ear(eye_points, landmarks):
    # Vertical distances
    p2_p6 = dist.euclidean(landmarks[eye_points[1]], landmarks[eye_points[5]])
    p3_p5 = dist.euclidean(landmarks[eye_points[2]], landmarks[eye_points[4]])
    # Horizontal distance
    p1_p4 = dist.euclidean(landmarks[eye_points[0]], landmarks[eye_points[3]])
    
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

# Setup Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    # Logic goes here... (we will fill the EAR check next)
    
    cv2.imshow('Drowsiness Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()