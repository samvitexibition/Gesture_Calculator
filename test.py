import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def test_hand_landmarker_live():
    # 1. Initialize MediaPipe detector
    base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=10,
        # Lowered slightly to 0.5 for better initial webcam responsiveness
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    detector = vision.HandLandmarker.create_from_options(options)

    # 2. Open webcam (0 is usually the default built-in camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    print("✅ Camera started! Press 'q' on the video window to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip horizontally for a natural mirror display
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert OpenCV BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 3. Detect hand landmarks
        detection_result = detector.detect(mp_image)

        # 4. Draw landmarks if detected
        if detection_result.hand_landmarks:
            for hand in detection_result.hand_landmarks:
                for landmark in hand:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # 5. Display frame
        cv2.imshow("MediaPipe Live Hand Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_hand_landmarker_live()