import json
import pickle
from typing import List, Dict, Optional
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

from normalize import normalize_landmarks_from_list
from gesture_logic import evaluate_hand_gesture

app = FastAPI()

# Mount static files & models
os.makedirs("static", exist_ok=True)
os.makedirs("models", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/models", StaticFiles(directory="models"), name="models")

# Load model
try:
    with open("models/sign_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("models/sign_model.pkl not found! Please ensure it is present.")
    model = None

GESTURE_MAP = {
    'close':          {'symbol': '0',     'display': 'Fist → 0'},
    'point':          {'symbol': '1',     'display': '1 Finger → 1'},
    'peace_horizontal': {'symbol': '2',     'display': 'V-Sign → 2'},
    '3_fingers':      {'symbol': '3',     'display': '3 Fingers → 3'},
    '4_fingers':      {'symbol': '4',     'display': '4 Fingers → 4'},
    'open':           {'symbol': '5',     'display': '5 Fingers → 5'},
    'count_6':        {'symbol': '6',     'display': '6 Fingers → 6'},
    'count_7':        {'symbol': '7',     'display': '7 Fingers → 7'},
    'count_8':        {'symbol': '8',     'display': '8 Fingers → 8'},
    'count_9':        {'symbol': '9',     'display': '9 Fingers → 9'},
    'count_10':       {'symbol': '10',    'display': '10 Fingers → 10'},
    'thumb':          {'symbol': '+',     'display': 'Thumbs Up → +'},
    'thumb_inverted': {'symbol': '-',     'display': 'Thumbs Down → -'},
    'rock_inverted':  {'symbol': '*',     'display': 'Rock Inv → ×'},
    'rock':           {'symbol': '(',     'display': 'Rock → ('},
    'peace_inverted': {'symbol': '/',     'display': 'Peace Inv → ÷'},
    'l_shape':        {'symbol': ')',     'display': 'L-Shape → )'},
    '4_fingers_inverted': {'symbol': '.',  'display': '4-Fingers Inv → .'},
    'point_inverted': {'symbol': '=',     'display': 'Point Down → ='},
    'open_inverted':  {'symbol': 'Clear', 'display': 'Open Inv → Clear'},
    'close_inverted': {'symbol': 'Back',  'display': 'Fist Inv → Back'},
}

@app.get("/")
async def get():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())

class StateModel(BaseModel):
    calc_expression: str = ""
    calc_result: str = ""
    consecutive_pred: Optional[str] = None
    consecutive_count: int = 0
    cooldown_frames: int = 0
    smoothed_hand_cache: Dict[str, List[float]] = {}

class PredictRequest(BaseModel):
    hands: List[List[float]]
    state: StateModel

@app.post("/predict")
async def predict_endpoint(req: PredictRequest):
    # Retrieve state
    calc_expression = req.state.calc_expression
    calc_result = req.state.calc_result
    consecutive_pred = req.state.consecutive_pred
    consecutive_count = req.state.consecutive_count
    cooldown_frames = req.state.cooldown_frames
    # FastAPI handles str keys in dict, convert back to int for local logic
    smoothed_hand_cache = {int(k): v for k, v in req.state.smoothed_hand_cache.items()}
    
    CONFIRM_THRESHOLD = 12
    COOLDOWN_MAX = 22

    hands = req.hands
    prediction = None
    confidence = 0.0
    display_text = "Show 1 or 2 hands..."
    num_hands_detected = len(hands)

    if num_hands_detected > 0:
        hand_evals = []
        for idx, hand_coords in enumerate(hands):
            raw_feats = normalize_landmarks_from_list(hand_coords)
            
            # Temporal smoothing
            alpha = 0.65
            if idx in smoothed_hand_cache:
                prev_feats = smoothed_hand_cache[idx]
                norm_feats = [alpha * curr + (1.0 - alpha) * prev for curr, prev in zip(raw_feats, prev_feats)]
            else:
                norm_feats = raw_feats
            smoothed_hand_cache[idx] = norm_feats
            
            pred_g, conf, cnt, is_op = evaluate_hand_gesture(norm_feats, model)
            hand_evals.append((pred_g, conf, cnt, is_op))
        
        # Clear cache for missing hands
        if num_hands_detected < len(smoothed_hand_cache):
            smoothed_hand_cache = {i: smoothed_hand_cache[i] for i in range(num_hands_detected)}

        if num_hands_detected == 1:
            prediction = hand_evals[0][0]
            confidence = hand_evals[0][1]
        elif num_hands_detected >= 2:
            op_hand = next((h for h in hand_evals if h[3]), None)
            if op_hand:
                prediction = op_hand[0]
                confidence = op_hand[1]
            else:
                counts = sorted([h[2] for h in hand_evals])
                total_cnt = sum(counts)
                if counts == [2, 5]:
                    total_cnt = 7
                total_cnt = min(total_cnt, 10)
                count_map = {
                    0: 'close', 1: 'point', 2: 'peace_horizontal', 3: '3_fingers', 4: '4_fingers', 5: 'open',
                    6: 'count_6', 7: 'count_7', 8: 'count_8', 9: 'count_9', 10: 'count_10'
                }
                prediction = count_map.get(total_cnt)
                confidence = (hand_evals[0][1] + hand_evals[1][1]) / 2.0

    # Logic for lock-in
    progress = 0.0
    if prediction and prediction in GESTURE_MAP:
        gesture_info = GESTURE_MAP[prediction]
        display_text = gesture_info['display']
        if num_hands_detected >= 2 and '→' in display_text:
            display_text = f"2-Hands: {display_text}"
        
        if cooldown_frames > 0:
            cooldown_frames -= 1
            display_text = "Cooldown..."
        else:
            symbol = gesture_info['symbol']
            is_operator = symbol in ['+', '-', '*', '/', '=']
            
            if is_operator and confidence < 75.0:
                consecutive_count = 0
            else:
                if prediction == consecutive_pred:
                    consecutive_count += 1
                else:
                    consecutive_pred = prediction
                    consecutive_count = 1

            progress = min(1.0, consecutive_count / CONFIRM_THRESHOLD)
            
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
                    operators = {'+', '-', '*', '/'}
                    if symbol in operators and calc_expression and calc_expression[-1] in operators:
                        # Allow * and / after another operator; block + and -
                        if symbol in ('*', '/'):
                            calc_expression += symbol
                        # else: skip, don't add consecutive +/-
                    elif symbol == '.' and '.' in calc_expression.split('+')[-1].split('-')[-1].split('*')[-1].split('/')[-1].split('(')[-1]:
                        pass  # Skip duplicate decimal in current number
                    else:
                        calc_expression += symbol
                    calc_result = ""
                
                cooldown_frames = COOLDOWN_MAX
                consecutive_count = 0
    else:
        consecutive_count = 0

    # Pack updated state
    updated_state = {
        "calc_expression": calc_expression,
        "calc_result": calc_result,
        "consecutive_pred": consecutive_pred,
        "consecutive_count": consecutive_count,
        "cooldown_frames": cooldown_frames,
        "smoothed_hand_cache": {str(k): v for k, v in smoothed_hand_cache.items()}
    }

    return {
        "prediction": prediction,
        "confidence": confidence,
        "display_text": display_text,
        "progress": progress,
        "calc_expression": calc_expression,
        "calc_result": calc_result,
        "cooldown": cooldown_frames > 0,
        "state": updated_state
    }
