import csv
import os
import random
import math
from normalize import normalize_landmarks_from_list

# ─── 12 gesture classes matching predict.py's GESTURE_MAP ────────────────────
# Each gesture maps to a finger-state bitmask:
#   bit 0 = thumb, bit 1 = index, bit 2 = middle, bit 3 = ring, bit 4 = pinky
GESTURE_CONFIGS = {
    'open':           {'code': 0b11111, 'inverted': False},  # 5 fingers
    '4_fingers':      {'code': 0b11110, 'inverted': False},  # 4 fingers
    'close':          {'code': 0b00000, 'inverted': False},  # Fist
    'peace':          {'code': 0b00110, 'inverted': False},  # Index + middle
    'point':          {'code': 0b00010, 'inverted': False},  # Index only
    'peace_horizontal': {'code': 0b00110, 'inverted': False, 'roll': 1.57}, # Peace sign rotated 90 degrees
    'rock':           {'code': 0b10010, 'inverted': False},  # Index + pinky (()
    '3_fingers':      {'code': 0b11100, 'inverted': False},  # Middle + ring + pinky (digit 3)
    'thumb':          {'code': 0b00001, 'inverted': False},  # Thumb only
    'l_shape':        {'code': 0b00011, 'inverted': False},  # Thumb + Index upright ())
    'open_inverted':  {'code': 0b11111, 'inverted': True},
    'close_inverted': {'code': 0b00000, 'inverted': True},
    'point_inverted': {'code': 0b00010, 'inverted': True},
    'rock_inverted':  {'code': 0b10010, 'inverted': True},
    'thumb_inverted': {'code': 0b00001, 'inverted': True},
    'l_shape_inverted': {'code': 0b00011, 'inverted': True}, # Thumb + Index inverted
    'peace_inverted': {'code': 0b00110, 'inverted': True},   # Index + middle inverted (÷)
    '4_fingers_inverted': {'code': 0b11110, 'inverted': True}, # 4 fingers inverted (.)
}


def get_realistic_hand(code, inverted=False, gesture_name=""):
    """
    Generates realistic 21-landmark 3D hand coordinates matching MediaPipe structure.
    Args:
        code: Finger-state bitmask (bit 0=thumb, bit 1=index, ..., bit 4=pinky).
        inverted: If True, flip Y coordinates to simulate an inverted hand pose.

    Returns:
        Flat list of 63 coordinates [x0, y0, z0, x1, y1, z1, ...].
    """
    # MCP base coordinates relative to wrist (0,0,0) for upright hand (negative Y is up)
    mcp_bases = [
        [-0.28, -0.25,  0.05],  # 1: Thumb CMC
        [-0.22, -0.78,  0.00],  # 5: Index MCP
        [ 0.00, -0.85,  0.00],  # 9: Middle MCP
        [ 0.20, -0.80,  0.00],  # 13: Ring MCP
        [ 0.36, -0.70,  0.00],  # 17: Pinky MCP
    ]

    # Anatomical finger segment lengths: [seg1, seg2, seg3]
    finger_segs = [
        [0.22, 0.20, 0.18],  # Thumb
        [0.32, 0.22, 0.16],  # Index
        [0.36, 0.24, 0.17],  # Middle
        [0.32, 0.22, 0.16],  # Ring
        [0.26, 0.17, 0.14],  # Pinky
    ]

    # Finger spread angles relative to middle finger (radians)
    finger_spread = [-0.65, -0.18, 0.0, 0.16, 0.38]

    # Allow gesture configs to override base roll
    roll = random.gauss(GESTURE_CONFIGS.get(code, {}).get('roll', 0), 0.15)
    
    if gesture_name in ['peace_inverted', 'point_inverted', 'peace', 'point', 'peace_horizontal']:
        scale = random.uniform(0.75, 1.25)
        noise_scale = 0.02
    else:
        scale = random.uniform(0.85, 1.15)
        noise_scale = 0.012

    # Add more variation to finger segment lengths for each sample
    segs = [[s * random.uniform(0.95, 1.05) for s in f_segs] for f_segs in finger_segs]


    landmarks = [(0.0, 0.0, 0.0)]  # 0: Wrist

    for f in range(5):
        is_ext = (code & (1 << f)) != 0
        base_x, base_y, base_z = mcp_bases[f]
        spread = finger_spread[f] + random.gauss(0, 0.04)

        if f == 0:
            # Thumb
            p1 = (base_x + random.gauss(0, 0.01), base_y + random.gauss(0, 0.01), base_z + random.gauss(0, 0.01))
            landmarks.append(p1)

            if is_ext:
                ang_x = -0.5 + spread + random.gauss(0, 0.05)
                ang_y = -0.6 + random.gauss(0, 0.05)
                z_dir = 0.1
            else:
                ang_x = 0.2 + random.gauss(0, 0.05)
                ang_y = -0.1 + random.gauss(0, 0.05)
                z_dir = -0.15

            cur_x, cur_y, cur_z = p1
            for seg in segs[0]:
                cur_x += seg * math.sin(ang_x) * scale
                cur_y += seg * math.cos(ang_y) * scale
                cur_z += z_dir * seg * scale
                landmarks.append((
                    cur_x + random.gauss(0, noise_scale),
                    cur_y + random.gauss(0, noise_scale),
                    cur_z + random.gauss(0, noise_scale)
                ))
        else:
            # Index, Middle, Ring, Pinky
            p_mcp = (base_x * scale, base_y * scale, base_z * scale)
            landmarks.append(p_mcp)

            if is_ext:
                dx = math.sin(spread)
                dy = -math.cos(spread)
                dz = random.gauss(0, 0.05)

                cur_x, cur_y, cur_z = p_mcp
                for seg in segs[f]:
                    cur_x += dx * seg * scale
                    cur_y += dy * seg * scale
                    cur_z += dz * seg * scale
                    landmarks.append((
                        cur_x + random.gauss(0, noise_scale),
                        cur_y + random.gauss(0, noise_scale),
                        cur_z + random.gauss(0, noise_scale)
                    ))
            else:
                # Folded finger curls back toward palm
                pip_x = p_mcp[0] + math.sin(spread) * segs[f][0] * 0.7 * scale
                pip_y = p_mcp[1] - math.cos(spread) * segs[f][0] * 0.7 * scale
                pip_z = p_mcp[2] - 0.15 * scale
                landmarks.append((
                    pip_x + random.gauss(0, noise_scale),
                    pip_y + random.gauss(0, noise_scale),
                    pip_z + random.gauss(0, noise_scale)
                ))

                dip_x = pip_x - math.sin(spread) * segs[f][1] * 0.3 * scale
                dip_y = pip_y + segs[f][1] * 0.8 * scale
                dip_z = pip_z + 0.1 * scale
                landmarks.append((
                    dip_x + random.gauss(0, noise_scale),
                    dip_y + random.gauss(0, noise_scale),
                    dip_z + random.gauss(0, noise_scale)
                ))

                tip_x = dip_x - math.sin(spread) * segs[f][2] * 0.1 * scale
                tip_y = dip_y + segs[f][2] * 0.6 * scale
                tip_z = dip_z + 0.05 * scale
                landmarks.append((
                    tip_x + random.gauss(0, noise_scale),
                    tip_y + random.gauss(0, noise_scale),
                    tip_z + random.gauss(0, noise_scale)
                ))

    # Flatten & rotate
    flat = []
    cos_r, sin_r = math.cos(roll), math.sin(roll)

    for (x, y, z) in landmarks:
        rx = x * cos_r - y * sin_r
        ry = x * sin_r + y * cos_r
        rz = z
        if inverted:
            ry = -ry
        flat.extend([rx, ry, rz])

    return flat


def generate_mock_data(samples_per_class=100):
    os.makedirs("dataset", exist_ok=True)
    filepath = "dataset/landmarks.csv"

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        headers = []
        for i in range(21):
            headers.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
        headers.append('gesture_label')
        writer.writerow(headers)

        for gesture_name, config in GESTURE_CONFIGS.items():
            code = config['code']
            inverted = config['inverted']
            
            samples = 500 if gesture_name in ['peace_inverted', 'point_inverted', 'peace', 'point', 'peace_horizontal'] else samples_per_class

            for _ in range(samples):
                raw_coords = get_realistic_hand(code, inverted, gesture_name)
                normalized = normalize_landmarks_from_list(raw_coords)
                writer.writerow(normalized + [gesture_name])

    total_samples = len(GESTURE_CONFIGS) * samples_per_class
    print(f"Successfully generated realistic normalized synthetic dataset at: {filepath}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Classes: {len(GESTURE_CONFIGS)}")
    print(f"Total samples: {total_samples}")
    print(f"Features per sample: 63 (21 landmarks x 3 coords)")


if __name__ == "__main__":
    generate_mock_data()