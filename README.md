# Gesture Calculator

**Gesture Calculator** is a hands-free calculator that turns hand gestures into numbers, operators, and calculator controls. Instead of reaching for a keyboard or touchscreen, you can use your webcam and simple gestures to build and evaluate expressions in real time.

The project combines a browser-based camera experience with MediaPipe hand tracking and a Python/FastAPI backend. Hand landmarks are normalized, smoothed, and classified before the resulting gesture is added to the calculator expression.

> Point, count, calculate — all without touching the screen.

## What it can do

- Recognize digits from hand and finger-count gestures.

- Support one- and two-hand input, including combined finger counts.

- Use gesture controls for addition, subtraction, multiplication, division, decimals, evaluation, clearing, and backspace.

- Display the detected hand landmarks and prediction feedback in the browser.

- Smooth predictions over consecutive frames so a gesture is confirmed intentionally instead of being triggered by a single noisy frame.

- Run locally with FastAPI or deploy through the included Vercel configuration.

- Retrain the classifier using the landmark dataset included in the repository.

## Gesture reference

The calculator uses the following gesture vocabulary. The exact appearance of a gesture can vary with camera angle and lighting, so the in-app gesture guide is the best place to check the expected pose while using the calculator.

| Gesture | Calculator action |
| --- | --- |
| Closed fist | `0` |
| One finger | `1` |
| Horizontal V-sign | `2` |
| Three fingers | `3` |
| Four fingers | `4` |
| Open hand | `5` |
| Six to ten fingers | `6` to `10` using one or two hands |
| Thumbs up | `+` |
| Thumbs down | `-` |
| Inverted rock gesture | `×` |
| Inverted peace gesture | `÷` |
| L-shape | `)` |
| Inverted four-finger gesture | `.` |
| Downward point | `=` |
| Inverted open hand | Clear the expression |
| Inverted fist | Backspace |

A gesture must remain stable for several frames before it is committed. After an input is accepted, a short cooldown helps prevent the same gesture from being entered repeatedly.

## Try the online version

You can try the deployed application without installing Python:

[**Open Gesture Calculator**](https://gesture-calculator-ten.vercel.app/)

Your browser will still need permission to use the camera. If the online version cannot access the camera, check the browser’s site permissions and make sure no other application is using the webcam.

## Installation and setup

### Requirements

Before getting started, make sure you have:

- **Python 3.10 or newer**

- **Git**

- A working webcam or camera

- An internet connection for installing dependencies and loading browser-side assets

### 1. Clone the repository

```bash
git clone https://github.com/samvitexibition/Gesture_Calculator.git
cd Gesture_Calculator
```

### 2. Create and activate a virtual environment

A virtual environment keeps this project’s Python packages separate from the rest of your system.

**Windows PowerShell or Command Prompt:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS or Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

When activation succeeds, your terminal prompt will usually begin with `(venv )`.

### 3. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

The dependency list includes FastAPI and Uvicorn for the web server, NumPy for numerical processing, scikit-learn for model inference and training, and WebSockets for real-time browser communication.

### 4. Check the model files

The application expects the following files to be available:

```
models/
├── hand_landmarker.task
└── sign_model.pkl
```

The repository includes both model files. If you retrain the classifier, `sign_model.pkl` will be replaced with the newly trained model.

### 5. Start the application

From the project root, run:

```bash
uvicorn app:app --reload
```

You should see Uvicorn start on a local address similar to:

```
http://127.0.0.1:8000
```

### 6. Open the calculator

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser and allow camera access when prompted. Once the hand tracker has loaded, use the gesture guide and start entering an expression.

To stop the development server, return to the terminal and press **Ctrl+C**.

## How the application works

The calculator is split into a browser experience and a Python backend:

1. The browser requests access to the webcam.

1. MediaPipe detects up to two hands and returns 3D landmark coordinates for each hand.

1. The frontend sends the landmarks and calculator state to the FastAPI application.

1. The backend normalizes and temporally smooths the landmarks.

1. Gesture logic combines geometric checks with the trained Random Forest classifier.

1. A stable prediction is mapped to a calculator symbol or control.

1. The updated expression, result, confidence, and visual feedback are returned to the browser.

This combination makes the interface feel responsive while reducing accidental inputs caused by small movements or momentary misclassification.

## Project structure

```
Gesture_Calculator/
├── app.py                    # FastAPI application and prediction endpoint
├── api/index.py              # Vercel serverless entry point
├── gesture_logic.py          # Gesture detection and classification rules
├── hand_detection.py         # Hand-landmark detection utilities
├── normalize.py              # Landmark normalization helpers
├── predict.py                # Local prediction/model utilities
├── train_model.py            # Train and save the Random Forest model
├── generate_synthetic_data.py# Optional dataset generation utility
├── test.py                   # General project checks
├── test_model.py             # Model-focused checks
├── requirements.txt          # Python dependencies
├── dataset/
│   ├── landmarks.csv         # Training dataset used by train_model.py
│   └── gesture_landmarks.csv # Additional landmark data
├── models/
│   ├── hand_landmarker.task  # MediaPipe hand-landmarker asset
│   └── sign_model.pkl        # Trained gesture classifier
└── static/
    ├── index.html            # Calculator page and UI markup
    ├── script.js             # Camera, tracking, and browser interaction logic
    └── style.css             # UI styling
```

## Retrain the gesture model

If you want to experiment with the classifier, make sure the training dataset is present at:

```
dataset/landmarks.csv
```

Then run:

```bash
python train_model.py
```

The training script loads the landmark features, separates features from labels, creates a stratified train/test split, trains a Random Forest classifier, prints a classification report, and saves the resulting model to:

```
models/sign_model.pkl
```

Retraining is optional for normal use because a trained model is already included in the repository.

## Troubleshooting

### `pip` is not recognized

Use Python to invoke pip directly:

```bash
python -m pip install -r requirements.txt
```

On macOS or Linux, you may need:

```bash
python3 -m pip install -r requirements.txt
```

### Python is not recognized

Check whether Python is available:

```bash
python --version
```

On Windows, this may also work:

```bash
py --version
```

If neither command works, install Python and enable **Add Python to PATH** during installation.

### The camera is not working

Confirm that the browser has camera permission, that another application is not using the webcam, and that the correct camera is selected. Running the app through `http://localhost` or `http://127.0.0.1` is recommended instead of opening the HTML file directly.

If the browser shows a camera permission prompt, choose **Allow**. You may need to refresh the page after changing the permission.

### `ModuleNotFoundError`

Make sure the virtual environment is active, then reinstall the dependencies:

```bash
python -m pip install -r requirements.txt
```

If the problem continues, update pip and try again:

```bash
python -m pip install --upgrade pip
```

### The model cannot be loaded

Confirm that both `models/sign_model.pkl` and `models/hand_landmarker.task` exist. If `sign_model.pkl` is missing, run `python train_model.py` after confirming that `dataset/landmarks.csv` is available.

### Predictions are inconsistent

Use good, even lighting; keep your hand inside the camera frame; avoid busy backgrounds; and hold each gesture steady. The calculator intentionally waits for a stable gesture before committing it, so short pauses are expected.

## Development notes

The project is designed as an approachable computer-vision experiment as well as a usable calculator. The classifier and gesture rules can be extended with new poses, while the frontend can be customized through the settings and styling already included in `static/`.

When adding a new gesture, update the gesture mapping in `app.py`, the corresponding detection logic in `gesture_logic.py`, and the in-app gesture guide so that the backend and user interface remain in sync.

## License

No license file is currently included in the repository. If you plan to reuse, modify, or distribute this project, add a license that reflects the permissions you want to grant.

## Acknowledgements

This project builds on the Python scientific-computing ecosystem, FastAPI, scikit-learn, and MediaPipe’s browser-based hand-landmark detection capabilities.

If you find a bug or have an idea for improving the gesture experience, open an issue or submit a pull request in the [repository](https://github.com/samvitexibition/Gesture_Calculator).

OR

You can also try the deployed version without installing anything:

https://gesture-calculator-ten.vercel.app/

For the online version, the browser may still require permission to access your camera.

---

Made with curiosity, computer vision, and a webcam.
