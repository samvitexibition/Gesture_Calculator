# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import pickle
import time
import math
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import mediapipe as mp
# pyrefly: ignore [missing-import]
from mediapipe.tasks import python
# pyrefly: ignore [missing-import]
from mediapipe.tasks.python import vision
from normalize import normalize_landmarks_from_mediapipe

# ─── Gesture → Calculator Mapping (0 to 10 Numbers + Operators) ──────────────
GESTURE_MAP = {
    # Digits 0 to 10
    'close':          {'symbol': '0',     'display': 'Fist → 0'},
    'point':          {'symbol': '1',     'display': '1 Finger → 1'},
    'peace_horizontal': {'symbol': '2',     'display': 'V-Sign → 2'},
    'rock':           {'symbol': '3',     'display': '3 Fingers → 3'},
    '4_fingers':      {'symbol': '4',     'display': '4 Fingers → 4'},
    'open':           {'symbol': '5',     'display': '5 Fingers → 5'},
    'count_6':        {'symbol': '6',     'display': '6 Fingers → 6'},
    'count_7':        {'symbol': '7',     'display': '7 Fingers → 7'},
    'count_8':        {'symbol': '8',     'display': '8 Fingers → 8'},
    'count_9':        {'symbol': '9',     'display': '9 Fingers → 9'},
    'count_10':       {'symbol': '10',    'display': '10 Fingers → 10'},

    # Operators
    'thumb':          {'symbol': '+',     'display': 'Thumbs Up → +'},
    'thumb_inverted': {'symbol': '-',     'display': 'Thumbs Down → -'},
    'rock_inverted':  {'symbol': '*',     'display': 'Rock Inv → x'},
    'peace_inverted': {'symbol': '/',     'display': 'Peace Inv → ÷'},

    # Actions
    'point_inverted': {'symbol': '=',     'display': 'Point Down → ='},
    'l_shape_inverted': {'symbol': '=',   'display': 'Point Down → ='},
    'open_inverted':  {'symbol': 'Clear', 'display': 'Open Inv → Clear'},
    '4_fingers_inverted': {'symbol': 'Clear', 'display': 'Open Inv → Clear'},
    'close_inverted': {'symbol': 'Back',  'display': 'Fist Inv → Back'},
}

# Color palette for HUD UI
COLOR_BG_DARK    = (30, 30, 30)
COLOR_BG_PANEL   = (45, 45, 50)
COLOR_GREEN      = (0, 220, 100)
COLOR_CYAN       = (220, 200, 0)
COLOR_ORANGE     = (0, 165, 255)
COLOR_WHITE      = (240, 240, 240)
COLOR_YELLOW     = (0, 255, 255)
COLOR_RED        = (80, 80, 255)
COLOR_LANDMARK   = (0, 255, 0)
COLOR_LANDMARK2  = (255, 255, 0)
COLOR_BONE       = (255, 100, 50)
COLOR_BONE2      = (50, 180, 255)
COLOR_PROGRESS_BG= (80, 80, 80)


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


def draw_hud_panel(frame, x, y, w, h, alpha=0.6):
    """Draw a semi-transparent dark panel as HUD background."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_BG_PANEL, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def dist_3d(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)


def count_extended_fingers(flat_coords):
    """
    Counts extended fingers (0-5) using 3D physical joint distance geometry.
    In 3D space:
    - An extended finger has its TIP farther from the wrist than its PIP knuckle.
    - A folded finger curls back toward the palm, bringing TIP closer to wrist than PIP knuckle.
    """
    pts = [(flat_coords[i*3], flat_coords[i*3+1], flat_coords[i*3+2]) for i in range(21)]
    wrist = pts[0]
    mcp_mid = pts[9]

    # Inverted check: in MediaPipe normalized coords, positive Y relative to wrist means pointing DOWN
    # Check middle finger MCP or middle finger TIP below wrist
    inverted = (mcp_mid[1] > 0.0) or (pts[12][1] > 0.0)

    # 3D Euclidean distance helper
    d = lambda i, j: math.sqrt((pts[i][0]-pts[j][0])**2 + (pts[i][1]-pts[j][1])**2 + (pts[i][2]-pts[j][2])**2)

    # 1. Index (MCP 5, PIP 6, DIP 7, TIP 8)
    ext_index = (d(0, 8) > d(0, 6)) and (d(0, 8) > 1.1 * d(0, 5))

    # 2. Middle (MCP 9, PIP 10, DIP 11, TIP 12)
    ext_middle = (d(0, 12) > d(0, 10)) and (d(0, 12) > 1.1 * d(0, 9))

    # 3. Ring (MCP 13, PIP 14, DIP 15, TIP 16)
    ext_ring = (d(0, 16) > d(0, 14)) and (d(0, 16) > 1.1 * d(0, 13))

    # 4. Pinky (MCP 17, PIP 18, DIP 19, TIP 20)
    ext_pinky = (d(0, 20) > d(0, 18)) and (d(0, 20) > 1.1 * d(0, 17))

    # 5. Thumb (CMC 1, MCP 2, IP 3, TIP 4)
    # Extended thumb points away from index MCP (5) and wrist (0)
    # Tightened thresholds to prevent fist (0) from triggering as thumb (+)
    # Curl-ratio: thumb tip must be farther from index PIP (6) than thumb IP (3) is,
    # ensuring the thumb is truly sticking out, not just wrapped over the fingers.
    thumb_curl_ratio = d(4, 6) / max(d(3, 6), 1e-6)
    ext_thumb = (
        (d(0, 4) > 1.4 * d(0, 2)) and
        (d(5, 4) > 0.65) and
        (d(9, 4) > 0.60) and
        (thumb_curl_ratio > 1.3)
    )

    # Check for horizontal peace sign (V-sign)
    # Tip of index and middle are far apart, and far from ring finger MCP
    is_v_sign = (d(8, 12) > 0.3) and (d(8, 13) > 0.3) and (d(12, 13) > 0.3)
    # Check wrist-to-MCP line angle for horizontal orientation
    is_horizontal = abs(pts[9][0] - pts[0][0]) > abs(pts[9][1] - pts[0][1]) * 1.5

    count = sum([ext_index, ext_middle, ext_ring, ext_pinky, ext_thumb])
    return count, inverted, (ext_thumb, ext_index, ext_middle, ext_ring, ext_pinky), is_v_sign and is_horizontal


def classify_geometry(flat_coords):
    """Rule-based geometric gesture classifier for high-reliability fallback."""
    cnt, inverted, (ext_thumb, ext_index, ext_middle, ext_ring, ext_pinky), is_h_peace = count_extended_fingers(flat_coords)

    if ext_index and ext_middle and ext_ring and ext_pinky and ext_thumb:
        g = 'open'
    elif ext_index and ext_middle and ext_ring and ext_pinky and not ext_thumb:
        g = '4_fingers'
    elif not ext_index and not ext_middle and not ext_ring and not ext_pinky and not ext_thumb:
        g = 'close'
    elif is_h_peace and ext_index and ext_middle and not ext_ring and not ext_pinky and not ext_thumb:
        g = 'peace_horizontal'
    elif ext_index and ext_middle and not ext_ring and not ext_pinky and not ext_thumb:
        g = 'peace'
    elif ext_index and not ext_middle and not ext_ring and not ext_pinky and not ext_thumb:
        g = 'point'
    elif ext_index and not ext_middle and not ext_ring and not ext_pinky and ext_thumb:
        g = 'l_shape'
    elif ext_thumb and ext_index and not ext_middle and not ext_ring and not ext_pinky:
        g = 'l_shape'
    elif ext_index and ext_pinky and not ext_middle and not ext_ring:
        g = 'rock'
    elif ext_thumb and not ext_index and not ext_middle and not ext_ring and not ext_pinky:
        g = 'thumb'
    else:
        if cnt == 5: g = 'open'
        elif cnt == 4: g = '4_fingers'
        elif cnt == 3: g = 'rock'
        elif cnt == 2: g = 'peace'
        elif cnt == 1: g = 'point'
        else: g = 'close'

    if inverted:
        g = g + '_inverted'
    return g


def evaluate_hand_gesture(flat_coords, model):
    """Evaluates a single hand feature vector using ML model + geometry."""
    cnt, inverted, _, is_h_peace = count_extended_fingers(flat_coords)
    geom_pred = classify_geometry(flat_coords)

    ml_pred = None
    ml_conf = 0.0
    if model is not None:
        try:
            ml_pred = model.predict([flat_coords])[0]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([flat_coords])[0]
                ml_conf = float(np.max(probs) * 100)
        except Exception:
            pass

    # Check for operator gestures (Thumbs, Inverted actions, etc.)
    ops = ['thumb', 'thumb_inverted', 'rock_inverted', 'peace_inverted', 'point_inverted',
           'open_inverted', 'close_inverted', 'l_shape', 'l_shape_inverted', '4_fingers_inverted']

    # Give strong preference to geometric operators, as they are distinct.
    if geom_pred in ops:
        # If ML model confidently predicts a DIFFERENT operator (e.g., point_inverted instead of l_shape_inverted), trust the ML model.
        if ml_pred in ops and ml_pred != geom_pred and ml_conf > 75:
            return ml_pred, ml_conf, cnt, True
            
        # Only override with ML if it's a VERY highly confident non-operator prediction.
        # Raised from 85→95 to prevent ML from overriding valid inverted gestures like point_inverted (=).
        if ml_pred and ml_pred not in ops and ml_conf > 95:
            return ml_pred, ml_conf, cnt, False
        return geom_pred, max(ml_conf, 92.0), cnt, True

    # For digits, blend geometric count with ML prediction
    digit_map = {0: 'close', 1: 'point', 2: 'peace_horizontal', 3: 'rock', 4: '4_fingers', 5: 'open'}
    geom_digit_pred = digit_map.get(cnt)

    # Prioritize geometric digit prediction for 0-5, as it's often more robust for simple counts.
    # Only use ML digit prediction if it's highly confident AND geometric prediction is absent or less confident.
    if geom_digit_pred:
        # If ML model is very confident and matches a digit, and its confidence is significantly higher
        if ml_pred in digit_map.values() and ml_conf >= 80.0 and ml_conf > max(85.0, 90.0): # 90.0 is geom_conf
            return ml_pred, ml_conf, cnt, False
        return geom_digit_pred, max(ml_conf, 90.0), cnt, False
    elif ml_pred in digit_map.values() and ml_conf >= 55.0: # Fallback to ML if no geometric digit
        return ml_pred, ml_conf, cnt, False

    # Fallback for unrecognized gestures
    return ml_pred if ml_pred else 'close', ml_conf, cnt, False


# Landmark EMA cache for temporal smoothing
smoothed_hand_cache = {}

# ─── Load Model ─────────────────────────────────────────────────────────────
try:
    with open("models/sign_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("models/sign_model.pkl not found! Run train_model.py first.")
    model = None

# ─── Load Hand Landmarker (Support 2 Hands!) ────────────────────────────────
base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,  # Support up to 2 hands simultaneously!
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

# ─── Camera ─────────────────────────────────────────────────────────────────
cap = get_camera()
if cap is None:
    print("[ERROR] No webcam detected. Cannot run prediction without a camera.")
    exit()
else:
    print("Webcam initialized for TWO-HAND detection. Press 'q' to quit.")

# ─── Calculator State ───────────────────────────────────────────────────────
calc_expression = ""
calc_result = ""
consecutive_pred = None
consecutive_count = 0
cooldown_frames = 0
CONFIRM_THRESHOLD = 27  # Increased from 12 to 27 (~0.5s slower lock-in at ~30fps)
COOLDOWN_MAX = 30       # Increased from 22 to 30 for cleaner gap between gestures

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20)
]

# ─── Main Loop ──────────────────────────────────────────────────────────────
while True:
    success, frame = cap.read()
    if not success:
        print("[WARNING] Failed to grab frame.")
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = detector.detect(mp_image)

    prediction = None
    confidence = 0.0
    num_hands_detected = 0

    if detection_result.hand_landmarks:
        num_hands_detected = len(detection_result.hand_landmarks)
        hand_evals = []

        for idx, hand in enumerate(detection_result.hand_landmarks):
            bone_color = COLOR_BONE if idx == 0 else COLOR_BONE2
            landmark_color = COLOR_LANDMARK if idx == 0 else COLOR_LANDMARK2

            # Extract raw normalized features
            raw_feats = normalize_landmarks_from_mediapipe(hand)

            # Apply temporal EMA smoothing to features to eliminate camera jitter
            alpha = 0.65
            if idx in smoothed_hand_cache:
                prev_feats = smoothed_hand_cache[idx]
                norm_feats = [alpha * curr + (1.0 - alpha) * prev for curr, prev in zip(raw_feats, prev_feats)]
            else:
                norm_feats = raw_feats
            smoothed_hand_cache[idx] = norm_feats

            for c in CONNECTIONS:
                p1 = (int(hand[c[0]].x * w), int(hand[c[0]].y * h))
                p2 = (int(hand[c[1]].x * w), int(hand[c[1]].y * h))
                cv2.line(frame, p1, p2, bone_color, 2, cv2.LINE_AA)
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, landmark_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 2, COLOR_WHITE, -1, cv2.LINE_AA)

            pred_g, conf, cnt, is_op = evaluate_hand_gesture(norm_feats, model)
            hand_evals.append((pred_g, conf, cnt, is_op))

        # Clear cache for missing hands
        if num_hands_detected < len(smoothed_hand_cache):
            smoothed_hand_cache = {i: smoothed_hand_cache[i] for i in range(num_hands_detected)}

        if num_hands_detected == 1:
            prediction = hand_evals[0][0]
            confidence = hand_evals[0][1]
        elif num_hands_detected >= 2:
            # 2 Hands logic: check if any hand is showing an operator
            op_hand = next((h for h in hand_evals if h[3]), None)
            if op_hand:
                prediction = op_hand[0]
                confidence = op_hand[1]
            else:
                # Specific two-hand digit logic
                counts = sorted([h[2] for h in hand_evals]) # e.g. [2, 5]
                total_cnt = sum(counts)

                # Special case for 7: one hand '5' and other hand '2'
                if counts == [2, 5]:
                    total_cnt = 7

                total_cnt = min(total_cnt, 10)
                count_map = {
                    0: 'close', 1: 'point', 2: 'peace_horizontal', 3: 'rock', 4: '4_fingers', 5: 'open',
                    6: 'count_6', 7: 'count_7', 8: 'count_8', 9: 'count_9', 10: 'count_10'
                }
                prediction = count_map.get(total_cnt)
                confidence = (hand_evals[0][1] + hand_evals[1][1]) / 2.0

    # ─── Top HUD Panel: Gesture + Confidence ────────────────────────────
    draw_hud_panel(frame, 5, 5, 380, 100)

    if prediction and prediction in GESTURE_MAP:
        gesture_info = GESTURE_MAP[prediction]
        display_text = gesture_info['display']
        if num_hands_detected >= 2 and '→' in display_text:
            display_text = f"2-Hands: {display_text}"

        color = COLOR_GREEN if confidence > 70 else COLOR_ORANGE
        cv2.putText(frame, display_text, (15, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        # Confidence bar
        bar_w = int(2.5 * min(99.99, confidence))
        cv2.rectangle(frame, (15, 50), (15 + 250, 62), COLOR_PROGRESS_BG, -1)
        bar_color = COLOR_GREEN if confidence > 70 else COLOR_ORANGE if confidence > 40 else COLOR_RED
        cv2.rectangle(frame, (15, 50), (15 + bar_w, 62), bar_color, -1)
        cv2.putText(frame, f"{confidence:.0f}%", (270, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)

        # Lock-in progress bar
        if cooldown_frames > 0:
            cooldown_frames -= 1
            cv2.putText(frame, "Cooldown...", (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ORANGE, 1, cv2.LINE_AA)
        else:
            symbol = gesture_info['symbol']
            is_operator = symbol in ['+', '-', '*', '/', '=']
            
            if is_operator and confidence < 70.0:
                consecutive_count = 0
            else:
                if prediction == consecutive_pred:
                    consecutive_count += 1
                else:
                    consecutive_pred = prediction
                    consecutive_count = 1

            progress = min(1.0, consecutive_count / CONFIRM_THRESHOLD)
            cv2.rectangle(frame, (15, 72), (15 + 250, 84), COLOR_PROGRESS_BG, -1)
            cv2.rectangle(frame, (15, 72), (15 + int(250 * progress), 84), COLOR_CYAN, -1)
            cv2.putText(frame, "LOCK-IN", (270, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1, cv2.LINE_AA)

            if consecutive_count >= CONFIRM_THRESHOLD:
                symbol = gesture_info['symbol']

                if symbol == 'Clear':
                    calc_expression = ""
                    calc_result = ""
                elif symbol == 'Back':
                    calc_expression = calc_expression[:-1]
                    calc_result = ""
                elif symbol == '=':
                    try:
                        if calc_expression:
                            calc_result = str(eval(calc_expression))
                    except Exception:
                        calc_result = "Error"
                else:
                    calc_expression += symbol
                    calc_result = ""

                cooldown_frames = COOLDOWN_MAX
                consecutive_count = 0
    else:
        cv2.putText(frame, "Show 1 or 2 hands...", (15, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1, cv2.LINE_AA)
        consecutive_count = 0

    # ─── Bottom HUD Panel: Calculator ────────────────────────────────────
    panel_y = h - 90
    draw_hud_panel(frame, 5, panel_y, w - 10, 82)

    expr_display = calc_expression if calc_expression else "..."
    cv2.putText(frame, f"Calc: {expr_display}", (15, panel_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_WHITE, 2, cv2.LINE_AA)

    if calc_result:
        result_color = COLOR_RED if calc_result == "Error" else COLOR_YELLOW
        cv2.putText(frame, f"= {calc_result}", (15, panel_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, result_color, 2, cv2.LINE_AA)

    # ─── Right side: Gesture legend (compact) ────────────────────────────
    legend_x = w - 185
    draw_hud_panel(frame, legend_x - 5, 5, 185, 310)
    cv2.putText(frame, "GESTURES (0-10)", (legend_x, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CYAN, 1, cv2.LINE_AA)
    legend_items = [
        ("0,1,3,4,5 (1-Hand)", "0,1,3,4,5"),
        ("2 (V-Sign), 7 (5+2)", "2, 7"),
        ("Thumbs Up", "+"),
        ("Thumbs Down", "-"),
        ("Rock Inverted", "×"),
        ("Peace Inverted", "/"),
        ("Point Down", "="),
        ("Open Inverted", "CLR"),
        ("Fist Inverted", "DEL"),
    ]
    for i, (name, sym) in enumerate(legend_items):
        ly = 45 + i * 26
        cv2.putText(frame, f"{name}: {sym}", (legend_x, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)

    # ─── Show frame ──────────────────────────────────────────────────────
    cv2.imshow("Hand Gesture Calculator (2-Hand Supported)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()