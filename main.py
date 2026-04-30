"""
Iris Tracking Virtual Mouse

A real-time contactless cursor control system using iris tracking and blink detection.
Built using OpenCV, MediaPipe, and PyAutoGUI.

Features:
- Cursor control using eye movement
- Left/Right click using blink detection
- Pause/Resume using long eye closure
- Head movement compensation
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# Disable PyAutoGUI failsafe (prevents crash at screen corner)
pyautogui.FAILSAFE = False

# ================= USER PARAMETERS =================
LONG_BLINK_TIME = 0.5      # Right click threshold
PAUSE_TIME = 2.0           # Pause tracking threshold
BLINK_COOLDOWN = 0.7       # Prevent repeated clicks

SENSITIVITY = 1200         # Cursor movement sensitivity
DEAD_ZONE = 0.01           # Ignore small movements
SMOOTH_ALPHA = 0.25        # Smoothing factor
DEBUG = True
# ==================================================

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices
R_EYE = {"l": 33, "r": 133, "t": 159, "b": 145, "iris": [468, 469, 470, 471]}
L_EYE = {"l": 362, "r": 263, "t": 386, "b": 374, "iris": [473, 474, 475, 476]}
NOSE_TIP = 1

# Screen size
SCREEN_W, SCREEN_H = pyautogui.size()

# ================= HELPER FUNCTIONS =================

def get_point(lm, w, h):
    """Convert normalized landmark to pixel coordinates"""
    return np.array([lm.x * w, lm.y * h])

def compute_ear(lms, eye, w, h):
    """Compute Eye Aspect Ratio (EAR)"""
    vertical = np.linalg.norm(get_point(lms[eye["t"]], w, h) - get_point(lms[eye["b"]], w, h))
    horizontal = np.linalg.norm(get_point(lms[eye["l"]], w, h) - get_point(lms[eye["r"]], w, h))
    return vertical / horizontal if horizontal > 0 else 0

def get_iris_center(lms, ids, w, h):
    """Compute center of iris"""
    return np.mean([get_point(lms[i], w, h) for i in ids], axis=0)

# ================= STATE VARIABLES =================
blink_start = None
last_click = 0
tracking = True

calib_center = None
ear_threshold = None

prev_nose = None
prev_x, prev_y = pyautogui.position()

# Initialize camera
cap = cv2.VideoCapture(0)

print("\nInstructions:")
print("• Look center → press 'c' to calibrate")
print("• Short blink → Left Click")
print("• Long blink → Right Click")
print("• Both eyes closed (2s) → Pause/Resume")
print("• Press 'q' to quit\n")

# ================= EAR CALIBRATION =================
print("Calibrating EAR... Keep eyes open")
ear_samples = []
start_time = time.time()

while time.time() - start_time < 3:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if result.multi_face_landmarks:
        lms = result.multi_face_landmarks[0].landmark
        ear_val = (compute_ear(lms, R_EYE, w, h) + compute_ear(lms, L_EYE, w, h)) / 2
        ear_samples.append(ear_val)

ear_threshold = 0.75 * np.mean(ear_samples)
print(f"EAR Threshold: {ear_threshold:.3f}")

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.multi_face_landmarks:
        lms = result.multi_face_landmarks[0].landmark

        # Compute EAR
        ear_val = (compute_ear(lms, R_EYE, w, h) + compute_ear(lms, L_EYE, w, h)) / 2

        # Compute iris position
        iris = (get_iris_center(lms, R_EYE["iris"], w, h) +
                get_iris_center(lms, L_EYE["iris"], w, h)) / 2

        # Nose position for compensation
        nose = get_point(lms[NOSE_TIP], w, h)

        # -------- BLINK DETECTION --------
        if ear_val < ear_threshold:
            if blink_start is None:
                blink_start = time.time()
        else:
            if blink_start:
                duration = time.time() - blink_start

                if time.time() - last_click > BLINK_COOLDOWN:
                    if duration > PAUSE_TIME:
                        tracking = not tracking
                        print("Tracking:", "ON" if tracking else "PAUSED")

                    elif duration > LONG_BLINK_TIME:
                        pyautogui.click(button="right")
                        print("Right Click")

                    else:
                        pyautogui.click()
                        print("Left Click")

                    last_click = time.time()

                blink_start = None

        # -------- CURSOR MOVEMENT --------
        if tracking and calib_center:
            nx, ny = iris[0] / w, iris[1] / h
            dx = nx - calib_center[0]
            dy = ny - calib_center[1]

            # Dead zone filtering
            if abs(dx) < DEAD_ZONE: dx = 0
            if abs(dy) < DEAD_ZONE: dy = 0

            # Head movement compensation
            if prev_nose is not None:
                dx -= (nose[0] - prev_nose[0]) / w
                dy -= (nose[1] - prev_nose[1]) / h

            # Map to screen
            move_x = prev_x + dx * SENSITIVITY
            move_y = prev_y + dy * SENSITIVITY

            # Smoothing
            sm_x = int(SMOOTH_ALPHA * move_x + (1 - SMOOTH_ALPHA) * prev_x)
            sm_y = int(SMOOTH_ALPHA * move_y + (1 - SMOOTH_ALPHA) * prev_y)

            pyautogui.moveTo(sm_x, sm_y, duration=0)
            prev_x, prev_y = sm_x, sm_y

        prev_nose = nose

        # -------- DEBUG DISPLAY --------
        if DEBUG:
            cv2.circle(frame, tuple(iris.astype(int)), 4, (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"EAR: {ear_val:.2f} | {'TRACKING' if tracking else 'PAUSED'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if tracking else (0, 0, 255),
                2
            )

    cv2.imshow("Eye Controlled Mouse", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('c') and result.multi_face_landmarks:
        iris = (get_iris_center(lms, R_EYE["iris"], w, h) +
                get_iris_center(lms, L_EYE["iris"], w, h)) / 2
        calib_center = (iris[0] / w, iris[1] / h)
        print("Calibrated Center:", calib_center)

# Cleanup
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
