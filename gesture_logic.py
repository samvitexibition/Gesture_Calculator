import math
import numpy as np

def count_extended_fingers(flat_coords):
    """
    Counts extended fingers (0-5) using 3D physical joint distance geometry.
    """
    pts = [(flat_coords[i*3], flat_coords[i*3+1], flat_coords[i*3+2]) for i in range(21)]
    wrist = pts[0]
    mcp_mid = pts[9]

    # Inverted check: in MediaPipe normalized coords, positive Y relative to wrist means pointing DOWN
    inverted = mcp_mid[1] > 0.0

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
    ext_thumb = (d(0, 4) > 1.1 * d(0, 2)) and (d(5, 4) > 0.40)

    # Check for horizontal peace sign (V-sign)
    is_v_sign = (d(8, 12) > 0.3) and (d(8, 13) > 0.3) and (d(12, 13) > 0.3)
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
    elif ext_thumb and ext_index and not ext_middle and not ext_ring and not ext_pinky:
        g = 'l_shape'
    elif ext_index and ext_pinky and not ext_middle and not ext_ring:
        g = 'rock'
    elif not ext_index and ext_middle and ext_ring and ext_pinky and not ext_thumb:
        g = '3_fingers'
    elif ext_thumb and not ext_index and not ext_middle and not ext_ring and not ext_pinky:
        g = 'thumb'
    else:
        if cnt == 5: g = 'open'
        elif cnt == 4: g = '4_fingers'
        elif cnt == 3: g = '3_fingers'
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
    ops = ['thumb', 'thumb_inverted', 'rock_inverted', 'l_shape_inverted', 'point_inverted', 'open_inverted', 'close_inverted',
           'rock', 'l_shape', 'peace_inverted', '4_fingers_inverted']

    if geom_pred in ops:
        if ml_pred and ml_pred not in ops and ml_conf > 85:
            return ml_pred, ml_conf, cnt, False
        return geom_pred, max(ml_conf, 92.0), cnt, True

    # For digits, blend geometric count with ML prediction
    digit_map = {0: 'close', 1: 'point', 2: 'peace_horizontal', 3: '3_fingers', 4: '4_fingers', 5: 'open'}
    geom_digit_pred = digit_map.get(cnt)

    if geom_digit_pred:
        if ml_pred in digit_map.values() and ml_conf >= 80.0 and ml_conf > max(85.0, 90.0):
            return ml_pred, ml_conf, cnt, False
        return geom_digit_pred, max(ml_conf, 90.0), cnt, False
    elif ml_pred in digit_map.values() and ml_conf >= 55.0:
        return ml_pred, ml_conf, cnt, False

    return ml_pred if ml_pred else 'close', ml_conf, cnt, False
