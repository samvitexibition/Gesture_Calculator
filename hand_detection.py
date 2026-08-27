import cv2
import csv
import os
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from normalize import normalize_landmarks_from_mediapipe

# ─── Gesture classes matching gesture_landmarks.csv ─────────────────────────
GESTURE_CLASSES = [
    'open', 'close', 'peace', 'point', 'rock', 'thumb',
    'peace_horizontal', '4_fingers',
    'open_inverted', 'close_inverted',
    'point_inverted', 'rock_inverted', 'thumb_inverted', 'l_shape_inverted'
]

CSV_FILE = 'dataset/landmarks.csv'

def get_camera():
    """Try opening camera across multiple indices and backends on Windows."""
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]
    for index in range(4):
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Successfully opened camera index {index}")
                    return cap
                cap.release()
    return None

# Load Hand Landmarker Model
base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = get_camera()
if cap is None:
    print("[ERROR] No webcam detected. Cannot collect data without a camera.")
    exit()

print("=== Hand Gesture Data Collector ===")
print(f"Gesture classes: {GESTURE_CLASSES}")
print("Press the corresponding number key (shown on screen) to record samples.")
print("Press 'q' to quit.\n")

# Ensure dataset directory exists
os.makedirs("dataset", exist_ok=True)

# Create CSV with header if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        headers = []
        for i in range(21):
            headers.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
        headers.append('gesture_label')
        writer.writerow(headers)

current_gesture_idx = 0
samples_recorded = 0

while True:
    success, frame = cap.read()
    if not success:
        print("[WARNING] Failed to grab frame.")
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Process frame with MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = detector.detect(mp_image)

    current_landmarks = None
    if detection_result.hand_landmarks:
        hand = detection_result.hand_landmarks[0]

        # Normalize landmarks (63 features)
        current_landmarks = normalize_landmarks_from_mediapipe(hand)

        # Draw hand
        connections = [
            (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
            (13,17),(0,17),(17,18),(18,19),(19,20)
        ]
        for c in connections:
            p1 = (int(hand[c[0]].x * w), int(hand[c[0]].y * h))
            p2 = (int(hand[c[1]].x * w), int(hand[c[1]].y * h))
            cv2.line(frame, p1, p2, (255, 100, 50), 2)
        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # UI: Show current gesture target and key mappings
    gesture_name = GESTURE_CLASSES[current_gesture_idx]
    cv2.putText(frame, f"Current: {gesture_name}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press SPACE to record | N/P to switch gesture | Q to quit", (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Show list of gestures
    for i, g in enumerate(GESTURE_CLASSES):
        color = (0, 255, 0) if i == current_gesture_idx else (150, 150, 150)
        cv2.putText(frame, f"{i}: {g}", (w - 200, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Recorded: {samples_recorded}", (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Gesture Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):  # SPACE to record
        if current_landmarks is not None:
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(current_landmarks + [gesture_name])
            samples_recorded += 1
            print(f"Saved sample for '{gesture_name}' (total: {samples_recorded})")
        else:
            print("No hand detected — cannot record.")
    elif key == ord("n"):  # Next gesture
        current_gesture_idx = (current_gesture_idx + 1) % len(GESTURE_CLASSES)
        print(f"Switched to gesture: {GESTURE_CLASSES[current_gesture_idx]}")
    elif key == ord("p"):  # Previous gesture
        current_gesture_idx = (current_gesture_idx - 1) % len(GESTURE_CLASSES)
        print(f"Switched to gesture: {GESTURE_CLASSES[current_gesture_idx]}")

cap.release()
cv2.destroyAllWindows()
print(f"\nDone! Total samples recorded this session: {samples_recorded}")