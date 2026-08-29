import json
# pyrefly: ignore [missing-import]
import pickle
import asyncio
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
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
    '4_fingers_inverted': {'symbol': 'Clear', 'display': 'Open Inv → Clear'},
    'point_inverted': {'symbol': '=',     'display': 'Point Down → ='},
    'l_shape_inverted': {'symbol': '=',   'display': 'Point Down → ='},
    'open_inverted':  {'symbol': 'Clear', 'display': 'Open Inv → Clear'},
    'close_inverted': {'symbol': 'Back',  'display': 'Fist Inv → Back'},
}

@app.get("/")
async def get():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Calculator state per connection
    calc_expression = ""
    calc_result = ""
    consecutive_pred = None
    consecutive_count = 0
    cooldown_frames = 0
    CONFIRM_THRESHOLD = 55  # ~1.8s - 2.0s hold required before lock-in
    COOLDOWN_MAX = 45       # ~1.5s cooldown after confirmation
    smoothed_hand_cache = {}
    prediction_history = []

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            hands = payload.get("hands", [])
            
            prediction = None
            confidence = 0.0
            display_text = "Show 1 or 2 hands..."
            num_hands_detected = len(hands)

            if num_hands_detected > 0:
                hand_evals = []
                for idx, hand_coords in enumerate(hands):
                    # hand_coords is a flat list of 63 floats from JS
                    raw_feats = normalize_landmarks_from_list(hand_coords)
                    
                    # Temporal smoothing
                    alpha = 0.3
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

                if prediction:
                    prediction_history.append(prediction)
                    if len(prediction_history) > 7:
                        prediction_history.pop(0)
                    prediction = max(set(prediction_history), key=prediction_history.count)
                else:
                    prediction_history.clear()
            else:
                prediction_history.clear()

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

            response = {
                "prediction": prediction,
                "confidence": confidence,
                "display_text": display_text,
                "progress": progress,
                "calc_expression": calc_expression,
                "calc_result": calc_result,
                "cooldown": cooldown_frames > 0
            }
            await websocket.send_json(response)
    
    except WebSocketDisconnect:
        print("Client disconnected")
