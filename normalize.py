"""
Shared normalization utilities for hand landmark features.
All scripts (collection, training, prediction) must use the same
feature extraction to ensure consistency.
"""
import math


def normalize_landmarks_from_list(raw_coords):
    """
    Normalize a flat list of 63 raw coordinates [x0, y0, z0, x1, y1, z1, ...].
    Used by train_model.py and generate_synthetic_data.py (which work with flat lists).

    Steps:
      1. Position normalization: subtract wrist (landmark 0) from all landmarks.
      2. Scale normalization: divide by distance between wrist (0) and middle finger MCP (9).

    Returns a flat list of 63 normalized values.
    """
    # Wrist is the first landmark (indices 0, 1, 2)
    wrist_x = raw_coords[0]
    wrist_y = raw_coords[1]
    wrist_z = raw_coords[2]

    # Subtract wrist position from all landmarks
    relative = []
    for i in range(21):
        relative.append(raw_coords[i * 3 + 0] - wrist_x)
        relative.append(raw_coords[i * 3 + 1] - wrist_y)
        relative.append(raw_coords[i * 3 + 2] - wrist_z)

    # Hand size = distance from wrist (0) to middle finger MCP (9)
    mcp_x = relative[9 * 3 + 0]
    mcp_y = relative[9 * 3 + 1]
    mcp_z = relative[9 * 3 + 2]
    hand_size = math.sqrt(mcp_x ** 2 + mcp_y ** 2 + mcp_z ** 2)

    if hand_size < 1e-6:
        hand_size = 1.0

    normalized = [val / hand_size for val in relative]
    return normalized


def normalize_landmarks_from_mediapipe(hand_landmarks):
    """
    Normalize directly from MediaPipe hand landmark objects.
    Used by hand_detection.py and predict.py (which have MediaPipe landmark objects).

    Steps:
      1. Position normalization: subtract wrist (landmark 0) from all landmarks.
      2. Scale normalization: divide by distance between wrist (0) and middle finger MCP (9).

    Returns a flat list of 63 normalized values.
    """
    wrist = hand_landmarks[0]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

    relative = []
    for lm in hand_landmarks:
        relative.extend([
            lm.x - wrist_x,
            lm.y - wrist_y,
            lm.z - wrist_z
        ])

    # Hand size = distance from wrist (0) to middle finger MCP (9)
    mcp = hand_landmarks[9]
    hand_size = math.sqrt(
        (mcp.x - wrist_x) ** 2 +
        (mcp.y - wrist_y) ** 2 +
        (mcp.z - wrist_z) ** 2
    )

    if hand_size < 1e-6:
        hand_size = 1.0

    normalized = [val / hand_size for val in relative]
    return normalized
