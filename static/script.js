import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

// ─── DOM Elements ───────────────────────────────────────────
const introScreen       = document.getElementById("intro-screen");
const dashboard         = document.getElementById("dashboard");
const startBtn          = document.getElementById("start-btn");

const video             = document.getElementById("webcam");
const canvasElement     = document.getElementById("output_canvas");
const canvasCtx         = canvasElement.getContext("2d");

const confidenceBadge   = document.getElementById("confidence-badge");
const fpsBadge          = document.getElementById("fps-badge");
const progressRingCircle= document.getElementById("progress-ring-circle");
const detectedSymbolText= document.getElementById("detected-symbol-text");
const gestureDesc       = document.getElementById("gesture-desc");

const calcExpression    = document.getElementById("calc-expression");
const calcResult        = document.getElementById("calc-result");
const signRenderer      = document.getElementById("sign-renderer");

// Settings
const settingsModal     = document.getElementById("settings-modal");
const settingsOverlay   = document.getElementById("settings-overlay");
const openSettingsBtn   = document.getElementById("open-settings-btn");
const closeSettingsBtn  = document.getElementById("close-settings-btn");
const saveSettingsBtn   = document.getElementById("save-settings-btn");
const resetSettingsBtn  = document.getElementById("reset-settings-btn");
const themeSelect       = document.getElementById("theme-select");
const accentSelect      = document.getElementById("accent-select");
const mirrorSelect      = document.getElementById("mirror-select");
const speedSelect       = document.getElementById("speed-select");
const skeletonSelect    = document.getElementById("skeleton-select");
const confidenceToggle  = document.getElementById("confidence-toggle");
const soundSelect       = document.getElementById("sound-select");
const sensitivitySelect = document.getElementById("sensitivity-select");

// Gesture Guide
const gestureModal      = document.getElementById("gesture-modal");
const gestureOverlay    = document.getElementById("gesture-overlay");
const openGestureBtn    = document.getElementById("open-gesture-btn");
const closeGestureBtn   = document.getElementById("close-gesture-btn");

// Calibration Wizard Elements
const calibModal          = document.getElementById("calibration-modal");
const calibOverlay        = document.getElementById("calibration-overlay");
const closeCalibBtn       = document.getElementById("close-calib-btn");
const calibIntroBtn       = document.getElementById("calib-intro-btn");
const openCalibBtn        = document.getElementById("open-calib-btn");
const recalibSettingsBtn  = document.getElementById("recalib-settings-btn");

const tabStep1            = document.getElementById("tab-step-1");
const tabStep2            = document.getElementById("tab-step-2");
const tabStep3            = document.getElementById("tab-step-3");
const stepContainer1      = document.getElementById("calib-step-1");
const stepContainer2      = document.getElementById("calib-step-2");
const stepContainer3      = document.getElementById("calib-step-3");

const calibBrightnessVal  = document.getElementById("calib-brightness-val");
const calibBrightnessFill = document.getElementById("calib-brightness-fill");
const calibLightingStatus = document.getElementById("calib-lighting-status");

const calibDistanceVal    = document.getElementById("calib-distance-val");
const calibDistanceFill   = document.getElementById("calib-distance-fill");
const calibDistanceStatus = document.getElementById("calib-distance-status");

const calibDetectedSymbol = document.getElementById("calib-detected-symbol");
const calibGestureFill    = document.getElementById("calib-gesture-fill");
const calibGestureStatus  = document.getElementById("calib-gesture-status");

const calibSkipBtn        = document.getElementById("calib-skip-btn");
const calibPrevBtn        = document.getElementById("calib-prev-btn");
const calibNextBtn        = document.getElementById("calib-next-btn");
const calibFinishBtn      = document.getElementById("calib-finish-btn");

// ─── State ──────────────────────────────────────────────────
let handLandmarker = undefined;
let webcamRunning  = false;
let lastVideoTime  = -1;
let ws             = null;
let currentSpeedMultiplier = 1.0;
let currentSkeletonStyle   = 'neon';
let showConfidence         = true;
let soundEnabled           = false;
let fpsFrames              = 0;
let fpsLastTime            = performance.now();

// ─── Hand skeleton connections ───────────────────────────────
const CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[0,17],[17,18],[18,19],[19,20]
];

// ─── Sign emoji map ──────────────────────────────────────────
const SIGN_MAP = {
    '0': '✊', '1': '☝️', '2': '✌️', '3': '🤘', '4': '🖖',
    '5': '🖐', '6': '☝️+🖐', '7': '✌️+🖐', '8': '🤘+🖐', '9': '🖐+🖖',
    '+': '👍', '-': '👎', '*': '🤘⬇', '/': '✌️⬇', '=': '👇'
};

// ─── FPS Counter ─────────────────────────────────────────────
function updateFPS() {
    fpsFrames++;
    const now = performance.now();
    if (now - fpsLastTime >= 1000) {
        fpsBadge.textContent = `${fpsFrames} FPS`;
        fpsFrames = 0;
        fpsLastTime = now;
    }
}

// ─── MediaPipe Setup ─────────────────────────────────────────
let pendingLaunch = false;

async function createHandLandmarker() {
    const modelAssetPaths = [
        "/models/hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ];

    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );

        for (const modelPath of modelAssetPaths) {
            for (const delegate of ["GPU", "CPU"]) {
                try {
                    handLandmarker = await HandLandmarker.createFromOptions(vision, {
                        baseOptions: {
                            modelAssetPath: modelPath,
                            delegate: delegate
                        },
                        runningMode: "VIDEO",
                        numHands: 2,
                        minHandDetectionConfidence: 0.5,
                        minHandPresenceConfidence: 0.5,
                        minTrackingConfidence: 0.5
                    });
                    console.log(`MediaPipe initialized (path: ${modelPath}, delegate: ${delegate})`);
                    if (startBtn && startBtn.textContent.includes("Loading")) {
                        startBtn.innerHTML = `<span>Launch App</span><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>`;
                    }
                    if (pendingLaunch) {
                        pendingLaunch = false;
                        handleLaunchClick();
                    }
                    return;
                } catch (e) {
                    console.warn(`MediaPipe attempt failed (path: ${modelPath}, delegate: ${delegate}):`, e);
                }
            }
        }
    } catch (err) {
        console.error("Failed to load MediaPipe vision resolver:", err);
    }
}
createHandLandmarker();

// ─── Persist & Load Settings ─────────────────────────────────
const DEFAULT_SETTINGS = {
    theme: 'dark',
    accent: 'blue',
    mirror: 'yes',
    speed: 'normal',
    skeleton: 'neon',
    confidence: 'yes',
    sound: 'off',
    sensitivity: 'medium'
};

function loadSettings() {
    const saved = JSON.parse(localStorage.getItem('gesturecalc_settings') || '{}');
    const s = { ...DEFAULT_SETTINGS, ...saved };

    applyTheme(s.theme);
    themeSelect.value = s.theme;

    applyAccent(s.accent);
    accentSelect.value = s.accent;

    applyMirror(s.mirror);
    mirrorSelect.value = s.mirror;

    applySpeed(s.speed);
    speedSelect.value = s.speed;

    applySkeleton(s.skeleton);
    skeletonSelect.value = s.skeleton;

    applyConfidenceToggle(s.confidence);
    confidenceToggle.value = s.confidence;

    applySound(s.sound);
    soundSelect.value = s.sound;

    applySensitivity(s.sensitivity);
    sensitivitySelect.value = s.sensitivity;
}

function saveSettings() {
    const settings = {
        theme:       themeSelect.value,
        accent:      accentSelect.value,
        mirror:      mirrorSelect.value,
        speed:       speedSelect.value,
        skeleton:    skeletonSelect.value,
        confidence:  confidenceToggle.value,
        sound:       soundSelect.value,
        sensitivity: sensitivitySelect.value,
    };
    localStorage.setItem('gesturecalc_settings', JSON.stringify(settings));

    applyTheme(settings.theme);
    applyAccent(settings.accent);
    applyMirror(settings.mirror);
    applySpeed(settings.speed);
    applySkeleton(settings.skeleton);
    applyConfidenceToggle(settings.confidence);
    applySound(settings.sound);
    applySensitivity(settings.sensitivity);

    // Flash button to confirm
    saveSettingsBtn.textContent = '✓ Saved!';
    saveSettingsBtn.style.background = 'var(--success)';
    setTimeout(() => {
        saveSettingsBtn.textContent = 'Save Settings';
        saveSettingsBtn.style.background = '';
    }, 1500);
    settingsModal.classList.add('hidden');
}

function resetSettings() {
    localStorage.removeItem('gesturecalc_settings');
    loadSettings();
    // Visual feedback
    resetSettingsBtn.textContent = '✓ Reset!';
    setTimeout(() => { resetSettingsBtn.textContent = 'Reset Defaults'; }, 1200);
}

function applyTheme(value) {
    if (value === 'light') {
        document.body.classList.add('light-theme');
        document.body.classList.remove('dark-theme');
    } else {
        document.body.classList.add('dark-theme');
        document.body.classList.remove('light-theme');
    }
}

function applyAccent(value) {
    document.body.setAttribute('data-accent', value);
}

function applyMirror(value) {
    if (value === 'no') {
        video.classList.add('video-no-mirror');
        canvasElement.classList.add('canvas-no-mirror');
    } else {
        video.classList.remove('video-no-mirror');
        canvasElement.classList.remove('canvas-no-mirror');
    }
}

function applySpeed(value) {
    const map = { fast: 2.0, normal: 1.0, slow: 0.5 };
    currentSpeedMultiplier = map[value] || 1.0;
}

function applySkeleton(value) {
    currentSkeletonStyle = value;
}

function applyConfidenceToggle(value) {
    showConfidence = value === 'yes';
    if (confidenceBadge) {
        confidenceBadge.style.display = showConfidence ? '' : 'none';
    }
}

function applySound(value) {
    soundEnabled = value === 'on';
}

function applySensitivity(value) {
    // This affects visual feedback mainly; actual model thresholds are server-side
    // We can use it to filter low-confidence predictions client-side
}

// Load on boot
loadSettings();

// ─── Settings Modal Events ───────────────────────────────────
openSettingsBtn.addEventListener('click', () => settingsModal.classList.remove('hidden'));
closeSettingsBtn.addEventListener('click', () => settingsModal.classList.add('hidden'));
settingsOverlay.addEventListener('click', () => settingsModal.classList.add('hidden'));
saveSettingsBtn.addEventListener('click', saveSettings);
resetSettingsBtn.addEventListener('click', resetSettings);

// ─── Gesture Guide Modal Events ──────────────────────────────
openGestureBtn.addEventListener('click', () => gestureModal.classList.remove('hidden'));
closeGestureBtn.addEventListener('click', () => gestureModal.classList.add('hidden'));
gestureOverlay.addEventListener('click', () => gestureModal.classList.add('hidden'));

// ─── Calibration Wizard Logic ──────────────────────────────
let calibCurrentStep  = 1;
let isCalibActive     = false;
let calibGestureCount = 0;
const CALIB_GESTURE_TARGET = 8;

const sampleCanvas = document.createElement('canvas');
const sampleCtx = sampleCanvas.getContext('2d', { willReadFrequently: true });
sampleCanvas.width = 160;
sampleCanvas.height = 90;

function evaluateVideoLighting() {
    if (!video || video.readyState < 2) return;
    try {
        sampleCtx.drawImage(video, 0, 0, sampleCanvas.width, sampleCanvas.height);
        const imgData = sampleCtx.getImageData(0, 0, sampleCanvas.width, sampleCanvas.height).data;
        let totalLuminance = 0;
        const totalSampledPixels = imgData.length / 16;
        for (let i = 0; i < imgData.length; i += 16) {
            const r = imgData[i];
            const g = imgData[i + 1];
            const b = imgData[i + 2];
            totalLuminance += (0.299 * r + 0.587 * g + 0.114 * b);
        }
        const avgLum = totalLuminance / totalSampledPixels;
        const lumPercent = Math.min(100, Math.max(0, Math.round((avgLum / 255) * 100)));

        if (calibBrightnessVal) calibBrightnessVal.textContent = `${lumPercent}%`;
        if (calibBrightnessFill) calibBrightnessFill.style.width = `${lumPercent}%`;

        if (calibLightingStatus) {
            if (lumPercent < 25) {
                calibLightingStatus.className = 'calib-status-badge danger';
                calibLightingStatus.textContent = '⚠️ Too Dark! Please turn on room lights or face a light source.';
            } else if (lumPercent < 35) {
                calibLightingStatus.className = 'calib-status-badge warning';
                calibLightingStatus.textContent = '💡 Lighting is slightly dim. Tracking may be sensitive.';
            } else if (lumPercent <= 85) {
                calibLightingStatus.className = 'calib-status-badge success';
                calibLightingStatus.textContent = '✓ Optimal lighting detected! Ready for gesture tracking.';
            } else {
                calibLightingStatus.className = 'calib-status-badge warning';
                calibLightingStatus.textContent = '☀️ High brightness detected. Ensure strong glare is reduced.';
            }
        }
    } catch (_) { /* ignore CORS/video read errors */ }
}

function evaluateHandDistance(landmarks) {
    if (!calibDistanceVal || !calibDistanceFill || !calibDistanceStatus) return;

    if (!landmarks || landmarks.length === 0) {
        calibDistanceVal.textContent = '0%';
        calibDistanceFill.style.width = '0%';
        calibDistanceStatus.className = 'calib-status-badge info';
        calibDistanceStatus.textContent = '🖐 Raise hand into camera view...';
        return;
    }

    let minX = 1.0, maxX = 0.0;
    landmarks.forEach(lm => {
        if (lm.x < minX) minX = lm.x;
        if (lm.x > maxX) maxX = lm.x;
    });

    const handWidthRatio = maxX - minX;
    const distancePercent = Math.min(100, Math.round(handWidthRatio * 100));

    calibDistanceVal.textContent = `${distancePercent}%`;
    calibDistanceFill.style.width = `${distancePercent}%`;

    if (distancePercent < 18) {
        calibDistanceStatus.className = 'calib-status-badge warning';
        calibDistanceStatus.textContent = '📏 Hand is too far. Move closer to the camera.';
    } else if (distancePercent > 48) {
        calibDistanceStatus.className = 'calib-status-badge warning';
        calibDistanceStatus.textContent = '📏 Hand is too close. Move back slightly.';
    } else {
        calibDistanceStatus.className = 'calib-status-badge success';
        calibDistanceStatus.textContent = '✓ Optimal hand distance! Clear tracking range.';
    }
}

function evaluateCalibGesture(data) {
    if (calibCurrentStep !== 3 || !isCalibActive) return;

    if (data.display_text) {
        const symbol = data.display_text.includes('→') ? data.display_text.split('→')[1].trim() : data.display_text;
        if (calibDetectedSymbol) calibDetectedSymbol.textContent = symbol || '—';

        // Check for fist ('0') or point ('1')
        if (symbol === '0' || symbol === '1' || data.prediction === 'close' || data.prediction === 'point') {
            calibGestureCount++;
            const fillPct = Math.min(100, Math.round((calibGestureCount / CALIB_GESTURE_TARGET) * 100));
            if (calibGestureFill) calibGestureFill.style.width = `${fillPct}%`;

            if (calibGestureCount >= CALIB_GESTURE_TARGET) {
                if (calibGestureStatus) {
                    calibGestureStatus.className = 'calib-status-badge success';
                    calibGestureStatus.textContent = `✓ Gesture "${symbol}" Verified! Setup Complete.`;
                }
                if (calibFinishBtn) calibFinishBtn.classList.remove('hidden');
                if (calibNextBtn) calibNextBtn.classList.add('hidden');
            } else {
                if (calibGestureStatus) {
                    calibGestureStatus.className = 'calib-status-badge info';
                    calibGestureStatus.textContent = `Hold steady... ${fillPct}%`;
                }
            }
        }
    }
}

function setCalibStep(step) {
    calibCurrentStep = step;
    [tabStep1, tabStep2, tabStep3].forEach((tab, idx) => {
        if (!tab) return;
        if (idx + 1 === step) tab.classList.add('active');
        else tab.classList.remove('active');
        if (idx + 1 < step) tab.classList.add('completed');
        else tab.classList.remove('completed');
    });

    if (stepContainer1) stepContainer1.classList.toggle('hidden', step !== 1);
    if (stepContainer2) stepContainer2.classList.toggle('hidden', step !== 2);
    if (stepContainer3) stepContainer3.classList.toggle('hidden', step !== 3);

    if (calibPrevBtn) calibPrevBtn.classList.toggle('hidden', step === 1);

    if (step === 3) {
        if (calibNextBtn) calibNextBtn.classList.add('hidden');
        if (calibFinishBtn) calibFinishBtn.classList.remove('hidden');
    } else {
        if (calibNextBtn) calibNextBtn.classList.remove('hidden');
        if (calibFinishBtn) calibFinishBtn.classList.add('hidden');
    }
}

function openCalibration(startStep = 1) {
    if (!webcamRunning) {
        enableCam();
    }
    if (calibModal) calibModal.classList.remove('hidden');
    isCalibActive = true;
    calibGestureCount = 0;
    if (calibGestureFill) calibGestureFill.style.width = '0%';
    setCalibStep(startStep);
}

function closeCalibration() {
    if (calibModal) calibModal.classList.add('hidden');
    isCalibActive = false;
    localStorage.setItem('gesturecalc_calibrated', 'true');
}

function handleLaunchClick() {
    if (localStorage.getItem('gesturecalc_calibrated') !== 'true') {
        openCalibration(1);
    } else {
        enableCam();
    }
}

if (closeCalibBtn) closeCalibBtn.addEventListener('click', closeCalibration);
if (calibOverlay) calibOverlay.addEventListener('click', closeCalibration);
if (calibSkipBtn) calibSkipBtn.addEventListener('click', closeCalibration);
if (calibFinishBtn) calibFinishBtn.addEventListener('click', closeCalibration);
if (calibPrevBtn) calibPrevBtn.addEventListener('click', () => setCalibStep(Math.max(1, calibCurrentStep - 1)));
if (calibNextBtn) calibNextBtn.addEventListener('click', () => setCalibStep(Math.min(3, calibCurrentStep + 1)));

if (tabStep1) tabStep1.addEventListener('click', () => setCalibStep(1));
if (tabStep2) tabStep2.addEventListener('click', () => setCalibStep(2));
if (tabStep3) tabStep3.addEventListener('click', () => setCalibStep(3));

if (calibIntroBtn) calibIntroBtn.addEventListener('click', () => openCalibration(1));
if (openCalibBtn) openCalibBtn.addEventListener('click', () => openCalibration(1));
if (recalibSettingsBtn) recalibSettingsBtn.addEventListener('click', () => {
    if (settingsModal) settingsModal.classList.add('hidden');
    openCalibration(1);
});

// ─── Enter key to launch ─────────────────────────────────────
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !introScreen.classList.contains('hidden') && !introScreen.classList.contains('slide-up')) {
        handleLaunchClick();
    }
    // Escape to close modals
    if (e.key === 'Escape') {
        if (!settingsModal.classList.contains('hidden')) settingsModal.classList.add('hidden');
        if (!gestureModal.classList.contains('hidden')) gestureModal.classList.add('hidden');
        if (calibModal && !calibModal.classList.contains('hidden')) closeCalibration();
    }
});

// ─── Sound Effect ────────────────────────────────────────────
function playCommitSound() {
    if (!soundEnabled) return;
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        oscillator.frequency.setValueAtTime(880, audioCtx.currentTime);
        oscillator.frequency.setValueAtTime(1100, audioCtx.currentTime + 0.05);
        gainNode.gain.setValueAtTime(0.15, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.15);
        oscillator.start(audioCtx.currentTime);
        oscillator.stop(audioCtx.currentTime + 0.15);
    } catch (_) { /* ignore audio errors */ }
}

// ─── WebSocket ───────────────────────────────────────────────
function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    const wsStatus = document.getElementById('ws-status');

    ws.onopen = () => {
        if (wsStatus) { wsStatus.style.color = 'var(--success)'; wsStatus.title = 'WebSocket connected'; }
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateHUD(data);
    };

    ws.onclose = () => {
        if (wsStatus) { wsStatus.style.color = 'var(--danger)'; wsStatus.title = 'WebSocket disconnected'; }
        setTimeout(connectWebSocket, 1000);
    };
}

// ─── Sign Renderer ───────────────────────────────────────────
function renderSigns(text) {
    signRenderer.innerHTML = '';
    const chars = text.toString().split('');
    if (chars.length === 0) {
        signRenderer.innerHTML = '<div class="sign-placeholder">No result yet</div>';
        return;
    }
    chars.forEach(char => {
        const emoji = SIGN_MAP[char];
        if (!emoji) return; // skip chars with no mapping

        const wrapper = document.createElement('div');
        wrapper.className = 'sign-wrapper';

        const circle = document.createElement('div');
        circle.className = 'sign-circle';
        // Use textContent so emoji renders natively (not as broken image)
        circle.textContent = emoji;

        const label = document.createElement('div');
        label.className = 'sign-label';
        label.textContent = `"${char}"`;

        wrapper.appendChild(circle);
        wrapper.appendChild(label);
        signRenderer.appendChild(wrapper);
    });
}

// ─── HUD Update ──────────────────────────────────────────────
let prevCooldown = false;

function updateHUD(data) {
    if (isCalibActive && calibCurrentStep === 3) {
        evaluateCalibGesture(data);
    }

    // Confidence badge
    const conf = data.confidence || 0;
    confidenceBadge.textContent = `Confidence: ${conf.toFixed(0)}%`;
    if (conf > 70)      confidenceBadge.style.color = 'var(--success)';
    else if (conf > 40) confidenceBadge.style.color = 'var(--warning)';
    else                confidenceBadge.style.color = 'var(--danger)';

    // Detected gesture
    if (data.prediction && data.display_text) {
        const symbol = data.display_text.includes('→') ? data.display_text.split('→')[1].trim() : data.display_text;
        detectedSymbolText.textContent = symbol;
        gestureDesc.textContent = `Detected "${data.display_text}" — hold to confirm`;
    } else {
        detectedSymbolText.textContent = '—';
        gestureDesc.textContent = 'Waiting for input…';
    }

    // Progress ring
    const circumference = 2 * Math.PI * 40; // r=40
    if (data.cooldown) {
        progressRingCircle.style.strokeDashoffset = 0;
        progressRingCircle.style.stroke = 'var(--success)';
        gestureDesc.textContent = '✓ Input Committed!';
        // Play sound on transition to cooldown
        if (!prevCooldown) playCommitSound();
    } else {
        let progress = Math.min((data.progress || 0) * currentSpeedMultiplier, 1.0);
        progressRingCircle.style.strokeDashoffset = circumference - (progress * circumference);
        progressRingCircle.style.stroke = 'var(--accent)';
    }
    prevCooldown = data.cooldown;

    // Equation
    calcExpression.textContent = data.calc_expression || '…';

    // Result & sign renderer
    if (data.calc_result) {
        calcResult.textContent = data.calc_result;
        if (data.calc_result === 'Error') {
            calcResult.style.color = 'var(--danger)';
            signRenderer.innerHTML = '<div class="sign-placeholder">Cannot render error result</div>';
        } else {
            calcResult.style.color = 'var(--accent)';
            renderSigns(data.calc_result);
        }
    } else {
        calcResult.textContent = '—';
        signRenderer.innerHTML = '<div class="sign-placeholder">Perform a calculation to see result signs</div>';
    }
}

// ─── Camera Start ────────────────────────────────────────────
function enableCam() {
    if (!handLandmarker) {
        pendingLaunch = true;
        if (startBtn) startBtn.innerHTML = "⏳ Loading AI Engine...";
        return;
    }

    introScreen.classList.add('slide-up');
    setTimeout(() => {
        introScreen.classList.add('hidden');
        dashboard.classList.remove('hidden');
    }, 700);

    webcamRunning = true;
    connectWebSocket();

    if (!video.srcObject) {
        navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: "user" } })
            .then(stream => {
                video.srcObject = stream;
                video.addEventListener("loadeddata", predictWebcam);
            })
            .catch(err => {
                alert("Webcam access denied or unavailable. Please check camera permissions.");
                console.error(err);
            });
    }
}

startBtn.addEventListener("click", handleLaunchClick);

// ─── Draw Landmarks ──────────────────────────────────────────
function drawLandmarks(landmarks) {
    if (currentSkeletonStyle === 'off') return;

    const isNeon    = currentSkeletonStyle === 'neon';
    const isMinimal = currentSkeletonStyle === 'minimal';

    const boneColor  = isNeon ? 'rgba(88, 166, 255, 0.85)' : 'rgba(200, 200, 200, 0.6)';
    const pointColor = isNeon ? 'rgba(63, 185, 80, 1)' : 'rgba(255, 255, 255, 0.8)';
    const dotBg      = 'rgba(0, 0, 0, 0.6)';

    // Draw bones (skip for minimal)
    if (!isMinimal) {
        canvasCtx.lineWidth = isNeon ? 3 : 2;
        canvasCtx.strokeStyle = boneColor;
        if (isNeon) {
            canvasCtx.shadowColor = 'rgba(88, 166, 255, 0.5)';
            canvasCtx.shadowBlur = 8;
        }

        CONNECTIONS.forEach(([a, b]) => {
            const p1 = landmarks[a], p2 = landmarks[b];
            canvasCtx.beginPath();
            canvasCtx.moveTo(p1.x * canvasElement.width, p1.y * canvasElement.height);
            canvasCtx.lineTo(p2.x * canvasElement.width, p2.y * canvasElement.height);
            canvasCtx.stroke();
        });

        canvasCtx.shadowBlur = 0;
    }

    // Draw points
    const pointRadius = isMinimal ? 3 : 5;
    landmarks.forEach(lm => {
        const x = lm.x * canvasElement.width;
        const y = lm.y * canvasElement.height;
        canvasCtx.beginPath();
        canvasCtx.arc(x, y, pointRadius, 0, 2 * Math.PI);
        canvasCtx.fillStyle = pointColor;
        canvasCtx.fill();
        if (!isMinimal) {
            canvasCtx.beginPath();
            canvasCtx.arc(x, y, 2.5, 0, 2 * Math.PI);
            canvasCtx.fillStyle = dotBg;
            canvasCtx.fill();
        }
    });
}

// ─── Webcam Detection Loop ───────────────────────────────────
async function predictWebcam() {
    if (canvasElement.width !== video.videoWidth) {
        canvasElement.width  = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }

    const nowMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        const results = handLandmarker.detectForVideo(video, nowMs);

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        if (results.landmarks && results.landmarks.length > 0) {
            const payloadHands = [];
            for (const landmarks of results.landmarks) {
                drawLandmarks(landmarks);
                const flatCoords = landmarks.flatMap(lm => [lm.x, lm.y, lm.z]);
                payloadHands.push(flatCoords);
            }
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ hands: payloadHands }));
            }
        }
        canvasCtx.restore();

        if (isCalibActive) {
            if (calibCurrentStep === 1) {
                evaluateVideoLighting();
            } else if (calibCurrentStep === 2) {
                evaluateHandDistance(results.landmarks && results.landmarks[0] ? results.landmarks[0] : null);
            }
        }
    }

    updateFPS();

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}
