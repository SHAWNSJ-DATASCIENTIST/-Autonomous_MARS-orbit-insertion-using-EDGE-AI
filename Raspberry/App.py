from flask import Flask, Response, jsonify, render_template_string, request
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import time
import bluetooth
import os
from collections import deque
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

sensor_data = {
    "time": "--:--:--",
    "temperature": 0.0,
    "pressure": 0.0,
    "altitude": 0.0,
    "distance": 0.0,
    "mag_x": 0.0,
    "mag_y": 0.0,
    "mag_z": 0.0,
    "mag_heading": 0.0,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "accel_x": 0.0,
    "accel_y": 0.0,
    "accel_z": 0.0,
    "gyro_x": 0,
    "gyro_y": 0,
    "gyro_z": 0
}

sensor_lock = threading.Lock()
frame_lock = threading.Lock()

latest_frame = None
camera_running = True
mars_detection = {"x": 0, "y": 0, "detected": False, "center_x": 0, "center_y": 0}

# ---------------- BiLSTM + QRF MODELS ----------------

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_LEN = 10  # BiLSTM look-back window; adjust to match how the model was trained

try:
    bilstm_model = load_model(os.path.join(MODEL_DIR, "bilstm_model.keras"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    qrf_model = joblib.load(os.path.join(MODEL_DIR, "qrf_model.pkl"))
    print("BiLSTM / scaler / QRF models loaded successfully")
except Exception as e:
    bilstm_model = None
    scaler = None
    qrf_model = None
    print("Model load error:", e)

prediction_lock = threading.Lock()
feature_buffer = deque(maxlen=SEQ_LEN)

model_predictions = {
    "bilstm_available": False,
    "bilstm_output": None,
    "qrf_lower": None,
    "qrf_median": None,
    "qrf_upper": None
}


def run_predictions(heading, distance):
    global model_predictions

    if scaler is None or qrf_model is None or bilstm_model is None:
        return

    try:
        features = np.array([[float(heading), float(distance)]], dtype=np.float32)
        scaled = scaler.transform(features)

        # ---- Quartile Random Forest Regressor ----
        qrf_pred = np.array(qrf_model.predict(scaled)).flatten()

        with prediction_lock:
            if qrf_pred.size >= 3:
                model_predictions["qrf_lower"] = float(qrf_pred[0])
                model_predictions["qrf_median"] = float(qrf_pred[1])
                model_predictions["qrf_upper"] = float(qrf_pred[2])
            elif qrf_pred.size == 1:
                model_predictions["qrf_median"] = float(qrf_pred[0])
                model_predictions["qrf_lower"] = None
                model_predictions["qrf_upper"] = None

        # ---- BiLSTM (needs a sequence window) ----
        feature_buffer.append(scaled[0])

        if len(feature_buffer) == SEQ_LEN:
            seq = np.array(feature_buffer, dtype=np.float32).reshape(
                1, SEQ_LEN, scaled.shape[1]
            )
            bilstm_out = bilstm_model.predict(seq, verbose=0)
            bilstm_out = np.array(bilstm_out).flatten().tolist()

            with prediction_lock:
                model_predictions["bilstm_available"] = True
                model_predictions["bilstm_output"] = bilstm_out
        else:
            with prediction_lock:
                model_predictions["bilstm_available"] = False

    except Exception as e:
        print("Model prediction error:", e)


picam2 = Picamera2()

camera_config = picam2.create_video_configuration(
    main={
        "size": (640, 480),
        "format": "RGB888"
    },
    controls={
        "FrameRate": 15
    }
)

picam2.configure(camera_config)
picam2.start()

time.sleep(2)


def bluetooth_receiver():
    global sensor_data

    while True:
        sock = None
        try:
            print("Scanning for ESP32 SATELLITE_DATA...")
            devices = bluetooth.discover_devices(duration=8, lookup_names=True)
            target = None

            for address, name in devices:
                if name and name.strip() == "SATELLITE_DATA":
                    target = address
                    break

            if not target:
                print("ESP32 not found. Retrying...")
                time.sleep(3)
                continue

            print("ESP32 found:", target)
            sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            sock.connect((target, 1))
            sock.settimeout(5.0)
            print("Bluetooth connected")

            buffer = ""

            while True:
                data = sock.recv(1024)
                if not data:
                    raise ConnectionError("Bluetooth connection closed")

                buffer += data.decode("utf-8", errors="ignore")
                lines = buffer.split("\n")
                buffer = lines.pop()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    values = [v.strip() for v in line.split(",")]

                    if len(values) != 15:
                        print("Invalid Bluetooth packet:", line)
                        continue

                    try:
                        new_data = {
                            "time": values[0],
                            "temperature": float(values[1]),
                            "pressure": float(values[2]),
                            "altitude": float(values[3]),
                            "distance": float(values[4]),
                            "mag_x": float(values[5]),
                            "mag_y": float(values[6]),
                            "mag_z": float(values[7]),
                            "mag_heading": float(values[8]),
                            "roll": 0.0,
                            "pitch": 0.0,
                            "yaw": float(values[8]),
                            "accel_x": float(values[9]),
                            "accel_y": float(values[10]),
                            "accel_z": float(values[11]),
                            "gyro_x": float(values[12]),
                            "gyro_y": float(values[13]),
                            "gyro_z": float(values[14])
                        }

                        with sensor_lock:
                            sensor_data.update(new_data)

                        run_predictions(new_data["mag_heading"], new_data["distance"])

                    except ValueError as e:
                        print("Bluetooth data error:", e)

        except Exception as e:
            print("Bluetooth error:", e)
            time.sleep(3)

        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass


bluetooth_thread = threading.Thread(
    target=bluetooth_receiver,
    daemon=True
)
bluetooth_thread.start()

def camera_worker():
    global latest_frame, mars_detection

    while camera_running:
        try:
            frame = picam2.capture_array()

            h, w = frame.shape[:2]

            center_x = w // 2
            center_y = h // 2

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            hsv_red_1 = cv2.inRange(
                hsv,
                np.array([0, 100, 50], dtype=np.uint8),
                np.array([8, 255, 255], dtype=np.uint8)
            )

            hsv_red_2 = cv2.inRange(
                hsv,
                np.array([172, 100, 50], dtype=np.uint8),
                np.array([179, 255, 255], dtype=np.uint8)
            )

            hsv_red = cv2.bitwise_or(hsv_red_1, hsv_red_2)

            r = frame[:, :, 2].astype(np.int16)
            g = frame[:, :, 1].astype(np.int16)
            b = frame[:, :, 0].astype(np.int16)

            rgb_red = (
                (r > 70) &
                (r > g * 1.15) &
                (r > b * 1.15) &
                (r - g > 20) &
                (r - b > 20)
            ).astype(np.uint8) * 255

            mask = cv2.bitwise_and(hsv_red, rgb_red)

            kernel = np.ones((11, 11), np.uint8)

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=2
            )

            mask = cv2.dilate(
                mask,
                kernel,
                iterations=2
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=2
            )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            detected = False
            target_x = center_x
            target_y = center_y
            best_score = 0
            best_contour = None

            for contour in contours:
                area = cv2.contourArea(contour)

                if area < 2000:
                    continue

                x, y, bw, bh = cv2.boundingRect(contour)

                if bw < 50 or bh < 50:
                    continue

                aspect = bw / float(bh)

                if aspect < 0.45 or aspect > 2.8:
                    continue

                rect_area = float(bw * bh)
                fill_ratio = area / rect_area if rect_area > 0 else 0

                if fill_ratio < 0.15:
                    continue

                score = area * (0.5 + fill_ratio)

                if score > best_score:
                    best_score = score
                    best_contour = contour

            if best_contour is not None:
                M = cv2.moments(best_contour)

                if M["m00"] != 0:
                    target_x = int(M["m10"] / M["m00"])
                    target_y = int(M["m01"] / M["m00"])
                    detected = True

            tolerance = int(w * 0.08)

            if not detected:
                x_state, y_state = 0, 0
            elif target_x < center_x - tolerance:
                x_state, y_state = 1, 0
            elif target_x > center_x + tolerance:
                x_state, y_state = 0, 1
            else:
                x_state, y_state = 0, 0

            with frame_lock:
                mars_detection = {
                    "x": x_state,
                    "y": y_state,
                    "detected": detected,
                    "center_x": target_x,
                    "center_y": target_y
                }

            cv2.drawMarker(
                frame,
                (center_x, center_y),
                (255, 255, 255),
                cv2.MARKER_CROSS,
                28,
                2
            )

            if detected:
                cv2.rectangle(
                    frame,
                    (max(0, target_x - 60), max(0, target_y - 60)),
                    (min(w - 1, target_x + 60), min(h - 1, target_y + 60)),
                    (255, 255, 255),
                    2
                )

                cv2.circle(
                    frame,
                    (target_x, target_y),
                    12,
                    (255, 255, 255),
                    2
                )

                cv2.line(
                    frame,
                    (center_x, center_y),
                    (target_x, target_y),
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "MARS",
                    (target_x + 15, target_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

            ret, jpeg = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 75]
            )

            if ret:
                with frame_lock:
                    latest_frame = jpeg.tobytes()

        except Exception as e:
            print("Camera error:", e)
            time.sleep(0.2)


camera_thread = threading.Thread(
    target=camera_worker,
    daemon=True
)

camera_thread.start()


@app.route("/")
def dashboard():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OCEANEmbed &middot; Mission Console</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>

<style>

:root {
    --bg: #04070d;
    --bg-grid: rgba(56, 189, 248, 0.045);
    --panel: #070d16;
    --panel-alt: #0a1220;
    --border: #163247;
    --border-soft: #0f2333;
    --cyan: #38f2ff;
    --cyan-dim: #1c8a9b;
    --amber: #ffb84d;
    --red: #ff5c5c;
    --green: #35f0a1;
    --text: #dff3fa;
    --text-dim: #6f8ca0;
    --mono: "Consolas", "SFMono-Regular", ui-monospace, Menlo, monospace;
}

* { box-sizing: border-box; }

body {
    margin: 0;
    min-height: 100vh;
    background:
        linear-gradient(var(--bg-grid) 1px, transparent 1px) 0 0/38px 38px,
        linear-gradient(90deg, var(--bg-grid) 1px, transparent 1px) 0 0/38px 38px,
        radial-gradient(ellipse at top left, #071522 0%, var(--bg) 60%);
    color: var(--text);
    font-family: "Segoe UI", Roboto, Arial, sans-serif;
}

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* ---------------- HEADER ---------------- */

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 30px;
    border-bottom: 1px solid var(--border-soft);
    background: linear-gradient(180deg, rgba(7,13,22,0.9), rgba(7,13,22,0.4));
    backdrop-filter: blur(6px);
    position: sticky;
    top: 0;
    z-index: 20;
}

.brand { display: flex; align-items: center; gap: 14px; }

.brand-mark {
    width: 46px; height: 46px;
    border-radius: 10px;
    background: conic-gradient(from 220deg, var(--cyan), #0a3b46, var(--cyan));
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono);
    font-weight: 700;
    color: #021014;
    font-size: 15px;
    box-shadow: 0 0 18px rgba(56,242,255,0.35);
}

.brand-text h1 {
    margin: 0;
    font-size: 21px;
    letter-spacing: 3px;
    font-weight: 700;
}

.brand-text h1 span { color: var(--cyan); }

.brand-text p {
    margin: 3px 0 0;
    font-size: 11.5px;
    letter-spacing: 1.5px;
    color: var(--text-dim);
    text-transform: uppercase;
}

.top-right { display: flex; align-items: center; gap: 18px; }

.mission-clock {
    font-family: var(--mono);
    font-size: 18px;
    letter-spacing: 2px;
    color: var(--cyan);
    text-shadow: 0 0 10px rgba(56,242,255,0.5);
}

.link-badge {
    display: flex; align-items: center; gap: 9px;
    padding: 7px 14px;
    border-radius: 999px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-family: var(--mono);
}

.link-badge .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--red);
    box-shadow: 0 0 8px var(--red);
}

.link-badge.online .dot {
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
}

/* ---------------- LAYOUT ---------------- */

.wrap {
    max-width: 1360px;
    margin: 0 auto;
    padding: 26px 30px 50px;
}

.grid {
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 20px;
}

@media (max-width: 980px) {
    .grid { grid-template-columns: 1fr; }
}

.panel {
    background: linear-gradient(180deg, var(--panel), var(--panel-alt));
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
}

.panel::before {
    content: "";
    position: absolute; inset: 0;
    border-radius: 14px;
    padding: 1px;
    background: linear-gradient(120deg, rgba(56,242,255,0.25), transparent 40%);
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    pointer-events: none;
}

.panel-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 14px;
}

.panel-head h2 {
    margin: 0;
    font-size: 12.5px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--cyan);
    font-weight: 700;
}

.panel-head .sub {
    font-size: 10.5px;
    color: var(--text-dim);
    letter-spacing: 1px;
    font-family: var(--mono);
}

/* ---------------- TELEMETRY READOUTS ---------------- */

.readout-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 18px;
}

@media (max-width: 700px) { .readout-grid { grid-template-columns: repeat(2,1fr); } }

.readout {
    background: rgba(56,242,255,0.03);
    border: 1px solid var(--border-soft);
    border-radius: 10px;
    padding: 12px 14px;
}

.readout .r-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    color: var(--text-dim);
    text-transform: uppercase;
    margin-bottom: 6px;
    font-family: var(--mono);
}

.readout .r-value {
    font-family: var(--mono);
    font-size: 24px;
    font-weight: 600;
    color: var(--text);
}

.readout .r-unit { font-size: 11px; color: var(--text-dim); margin-left: 3px; }

/* ---------------- CHARTS ---------------- */

.chart-box { margin-bottom: 18px; }
.chart-box:last-child { margin-bottom: 0; }
.chart-box canvas { max-height: 150px; }

.axis-legend {
    display: flex; gap: 16px; margin-top: 8px;
    font-family: var(--mono); font-size: 11.5px; color: var(--text-dim);
}
.axis-legend b { color: var(--text); }
.leg-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; }
.leg-x { background: #ff6b6b; } .leg-y { background: #4dd4ff; } .leg-z { background: #ffd166; }

/* ---------------- OPTICAL PAYLOAD ---------------- */

.optical-frame {
    position: relative;
    width: 100%;
    aspect-ratio: 1 / 1;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
    background: #01050a;
}

.optical-frame img { width: 100%; height: 100%; object-fit: cover; display: block; }

.optical-tag {
    position: absolute; top: 10px; left: 10px;
    background: rgba(2,10,16,0.7);
    border: 1px solid rgba(56,242,255,0.4);
    padding: 4px 10px; border-radius: 6px;
    font-family: var(--mono);
    font-size: 10.5px; letter-spacing: 1px;
    color: var(--cyan);
}

.optical-rec {
    position: absolute; top: 10px; right: 10px;
    display: flex; align-items: center; gap: 6px;
    font-family: var(--mono); font-size: 10.5px; color: var(--red);
    background: rgba(2,10,16,0.7); padding: 4px 9px; border-radius: 6px;
}
.optical-rec .pulse {
    width: 7px; height: 7px; border-radius: 50%; background: var(--red);
    box-shadow: 0 0 6px var(--red);
    animation: pulse 1.4s infinite;
}
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.25;} }

.mini-stats {
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px;
}
.mini-stats .m { font-family: var(--mono); font-size: 11px; color: var(--text-dim); }
.mini-stats .m b { display:block; font-size: 15px; color: var(--text); margin-top: 2px; }

.btn {
    display: inline-block; margin-top: 14px;
    background: linear-gradient(135deg, var(--cyan), var(--cyan-dim));
    color: #021014;
    padding: 8px 16px;
    border-radius: 8px;
    font-family: var(--mono);
    font-size: 11.5px;
    letter-spacing: 1px;
    text-decoration: none;
    font-weight: 700;
}

footer {
    text-align: center;
    color: var(--text-dim);
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 1px;
    padding: 30px 0 10px;
}


.mars-align{margin-top:10px;padding:8px 10px;border:1px solid rgba(56,242,255,.14);border-radius:8px;background:rgba(255,255,255,.02);display:flex;align-items:center;gap:10px}.mars-align-title{font:700 8px var(--mono);letter-spacing:1.2px;color:var(--text-dim);white-space:nowrap}.mars-align-fields{display:flex;gap:5px}.mars-field{display:flex;align-items:center;gap:4px;border:1px solid rgba(255,255,255,.1);border-radius:5px;padding:3px 7px;min-width:28px;justify-content:center}.mars-field span{font:700 8px var(--mono);color:var(--text-dim)}.mars-field b{font:800 12px var(--mono);color:var(--text-bright)}.mars-status{font:700 8px var(--mono);letter-spacing:.8px;color:var(--text-dim);margin-left:auto}@media(max-width:850px){.mars-align{justify-content:center}}
.top-nav-data{display:flex;align-items:center;gap:10px;margin-right:10px;}
.top-compass{display:flex;align-items:center;gap:7px;}
.compass-ring{width:52px;height:52px;border:1px solid rgba(56,242,255,.55);border-radius:50%;position:relative;box-sizing:border-box;}
.compass-ring:before{content:"";position:absolute;inset:6px;border:1px solid rgba(255,255,255,.12);border-radius:50%;}
.compass-ring span{position:absolute;font:700 7px var(--mono);color:var(--text-dim);z-index:2;}
.c-n{top:2px;left:23px}.c-e{right:3px;top:22px}.c-s{bottom:2px;left:23px}.c-w{left:3px;top:22px}
.compass-needle{position:absolute;left:25px;top:11px;width:2px;height:16px;background:var(--accent);transform-origin:1px 15px;transform:rotate(0deg);box-shadow:0 0 7px rgba(56,242,255,.7);}
.compass-needle:after{content:"";position:absolute;left:-3px;top:-3px;border-left:4px solid transparent;border-right:4px solid transparent;border-bottom:8px solid var(--accent);}
.heading-mini{display:flex;flex-direction:column;font:700 9px var(--mono);line-height:1.1;min-width:37px}.heading-mini b{font-size:12px;color:var(--text-bright)}.heading-mini span{font-size:7px;color:var(--text-dim);letter-spacing:1px}
.attitude-mini{display:flex;flex-direction:column;gap:2px;border-left:1px solid rgba(255,255,255,.1);padding-left:9px;}
.attitude-mini div{display:flex;align-items:center;gap:5px;font:700 9px var(--mono);line-height:1;}
.attitude-mini span{width:9px;color:var(--accent);font-size:8px}.attitude-mini b{min-width:48px;color:var(--text-bright);font-size:10px;}
@media(max-width:850px){.top-nav-data{display:none;}}
</style>

<style>
.mission-title {
    width: 100%;
    text-align: center;
    font-size: 34px;
    font-weight: 900;
    letter-spacing: 5px;
    margin: 10px 0 22px;
    text-transform: uppercase;
}
.attitude-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 14px 0;
}
.attitude-card {
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    background: rgba(255,255,255,.035);
}
.attitude-label {
    font-size: 12px;
    letter-spacing: 2px;
    opacity: .65;
    font-weight: 800;
}
.attitude-value {
    font-size: 30px;
    font-weight: 900;
    margin-top: 7px;
}
.mag-panel {
    margin-top: 16px;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 16px;
    padding: 14px;
    background: rgba(255,255,255,.035);
}

.mars-align{margin-top:10px;padding:8px 10px;border:1px solid rgba(56,242,255,.14);border-radius:8px;background:rgba(255,255,255,.02);display:flex;align-items:center;gap:10px}.mars-align-title{font:700 8px var(--mono);letter-spacing:1.2px;color:var(--text-dim);white-space:nowrap}.mars-align-fields{display:flex;gap:5px}.mars-field{display:flex;align-items:center;gap:4px;border:1px solid rgba(255,255,255,.1);border-radius:5px;padding:3px 7px;min-width:28px;justify-content:center}.mars-field span{font:700 8px var(--mono);color:var(--text-dim)}.mars-field b{font:800 12px var(--mono);color:var(--text-bright)}.mars-status{font:700 8px var(--mono);letter-spacing:.8px;color:var(--text-dim);margin-left:auto}@media(max-width:850px){.mars-align{justify-content:center}}
.top-nav-data{display:flex;align-items:center;gap:10px;margin-right:10px;}
.top-compass{display:flex;align-items:center;gap:7px;}
.compass-ring{width:52px;height:52px;border:1px solid rgba(56,242,255,.55);border-radius:50%;position:relative;box-sizing:border-box;}
.compass-ring:before{content:"";position:absolute;inset:6px;border:1px solid rgba(255,255,255,.12);border-radius:50%;}
.compass-ring span{position:absolute;font:700 7px var(--mono);color:var(--text-dim);z-index:2;}
.c-n{top:2px;left:23px}.c-e{right:3px;top:22px}.c-s{bottom:2px;left:23px}.c-w{left:3px;top:22px}
.compass-needle{position:absolute;left:25px;top:11px;width:2px;height:16px;background:var(--accent);transform-origin:1px 15px;transform:rotate(0deg);box-shadow:0 0 7px rgba(56,242,255,.7);}
.compass-needle:after{content:"";position:absolute;left:-3px;top:-3px;border-left:4px solid transparent;border-right:4px solid transparent;border-bottom:8px solid var(--accent);}
.heading-mini{display:flex;flex-direction:column;font:700 9px var(--mono);line-height:1.1;min-width:37px}.heading-mini b{font-size:12px;color:var(--text-bright)}.heading-mini span{font-size:7px;color:var(--text-dim);letter-spacing:1px}
.attitude-mini{display:flex;flex-direction:column;gap:2px;border-left:1px solid rgba(255,255,255,.1);padding-left:9px;}
.attitude-mini div{display:flex;align-items:center;gap:5px;font:700 9px var(--mono);line-height:1;}
.attitude-mini span{width:9px;color:var(--accent);font-size:8px}.attitude-mini b{min-width:48px;color:var(--text-bright);font-size:10px;}
@media(max-width:850px){.top-nav-data{display:none;}}
</style>

</head>

<body>

<div class="topbar">
    <div class="brand">
        <div class="brand-mark">OE</div>
        <div class="brand-text">
            <h1>OCEAN<span>Embed</span></h1>
            <p>Autonomous Satellite Telemetry &amp; Vision Console</p>
        </div>
    </div>
    <div class="top-right">
        <div class="top-nav-data">
            <div class="top-compass">
                <div class="compass-ring">
                    <span class="c-n">N</span><span class="c-e">E</span><span class="c-s">S</span><span class="c-w">W</span>
                    <div class="compass-needle" id="compass-needle"></div>
                </div>
                <div class="heading-mini"><b id="heading-value">0.0</b><span>DEG</span></div>
            </div>
            <div class="attitude-mini">
                <div><span>R</span><b id="roll-value">0.0°</b></div>
                <div><span>P</span><b id="pitch-value">0.0°</b></div>
                <div><span>Y</span><b id="yaw-value">0.0°</b></div>
            </div>
        </div>
        <div class="mission-clock" id="mission-clock">--:--:--</div>
        <div class="link-badge" id="link-badge">
            <span class="dot"></span>
            <span id="link-text">ACQUIRING SIGNAL</span>
        </div>
    </div>
</div>

<div class="wrap">
<div class="grid">

    <!-- LEFT: TELEMETRY -->
    <div class="left-col">

        <div class="panel" style="margin-bottom:20px;">
            <div class="panel-head">
                <h2>Environmental Telemetry</h2>
                <span class="sub">DOWNLINK · REAL-TIME</span>
            </div>

            <div class="readout-grid">
                <div class="readout">
                    <div class="r-label">Thermal</div>
                    <div class="r-value"><span id="temperature">--</span><span class="r-unit">&deg;C</span></div>
                </div>
                <div class="readout">
                    <div class="r-label">Barometric Pressure</div>
                    <div class="r-value"><span id="pressure">--</span><span class="r-unit">hPa</span></div>
                </div>
                <div class="readout">
                    <div class="r-label">Altitude AGL</div>
                    <div class="r-value"><span id="altitude">--</span><span class="r-unit">m</span></div>
                </div>
                <div class="readout">
                    <div class="r-label">Proximity Range</div>
                    <div class="r-value"><span id="distance">--</span><span class="r-unit">cm</span></div>
                </div>
            </div>

            <div class="chart-box">
                <canvas id="chart-env" height="70"></canvas>
            </div>
        </div>

        <div class="panel">
            <div class="panel-head">
                <h2>Inertial Measurement Unit</h2>
                <span class="sub">MPU6050 · 6-AXIS</span>
            </div>

            <div class="chart-box">
                <div class="sub" style="font-family:var(--mono); font-size:10.5px; color:var(--text-dim); margin-bottom:6px;">LINEAR ACCELERATION (m/s&sup2;)</div>
                <canvas id="chart-accel" height="80"></canvas>
                <div class="axis-legend">
                    <span><span class="leg-dot leg-x"></span>X <b id="accel_x">--</b></span>
                    <span><span class="leg-dot leg-y"></span>Y <b id="accel_y">--</b></span>
                    <span><span class="leg-dot leg-z"></span>Z <b id="accel_z">--</b></span>
                </div>
            </div>

            <div class="chart-box">
                <div class="sub" style="font-family:var(--mono); font-size:10.5px; color:var(--text-dim); margin-bottom:6px;">ANGULAR VELOCITY (raw &deg;/s)</div>
                <canvas id="chart-gyro" height="80"></canvas>
                <div class="axis-legend">
                    <span><span class="leg-dot leg-x"></span>X <b id="gyro_x">--</b></span>
                    <span><span class="leg-dot leg-y"></span>Y <b id="gyro_y">--</b></span>
                    <span><span class="leg-dot leg-z"></span>Z <b id="gyro_z">--</b></span>
                </div>
            </div>
        </div>

    </div>

    <!-- RIGHT: OPTICAL PAYLOAD -->
    <div class="right-col">
        <div class="panel">
            <div class="panel-head">
                <h2>Optical Payload</h2>
                <span class="sub">ONBOARD OPTICAL PAYLOAD</span>
            </div>

            <div class="optical-frame">
                <span class="optical-tag">LIVE FEED</span>
                <span class="optical-rec"><span class="pulse"></span>REC</span>
                <img src="/video_feed" alt="Onboard optical payload feed">
            </div>

            <div class="mars-align">
                <div class="mars-align-title">MARS ALIGNMENT</div>
                <div class="mars-align-fields">
                    <div class="mars-field"><span>X</span><b id="mars-x">0</b></div>
                    <div class="mars-field"><span>Y</span><b id="mars-y">0</b></div>
                </div>
                <div class="mars-status" id="mars-status">CENTERED</div>
            </div>

            <div class="mini-stats">
                <div class="m">Mission Clock<b id="rtc-time">--:--:--</b></div>
                <div class="m">Uplink Status<b id="esp-status">STANDBY</b></div>
            </div>

            <a class="btn" href="/health" target="_blank">SYSTEM HEALTH</a>
        </div>
    </div>

</div>
</div>

<footer>OCEANEMBED MISSION CONSOLE &middot; GROUND STATION LINK /sensor &middot; SIH DEMONSTRATION BUILD</footer>

<script>

const N = 40;
const hist = {
    labels: [],
    temp: [], press: [], alt: [], dist: [],
    ax: [], ay: [], az: [],
    gx: [], gy: [], gz: []
};

function pushHist(d) {
    hist.labels.push(d.time || "");
    hist.temp.push(d.temperature);
    hist.press.push(d.pressure);
    hist.alt.push(d.altitude);
    hist.dist.push(d.distance);
    hist.ax.push(d.accel_x); hist.ay.push(d.accel_y); hist.az.push(d.accel_z);
    hist.gx.push(d.gyro_x); hist.gy.push(d.gyro_y); hist.gz.push(d.gyro_z);
    Object.keys(hist).forEach(k => { if (hist[k].length > N) hist[k].shift(); });
}

function baseOptions(extra) {
    return Object.assign({
        responsive: true,
        animation: false,
        interaction: { mode: "index", intersect: false },
        scales: {
            x: { display: false },
            y: { ticks: { color: "#6f8ca0", font: { family: "Consolas", size: 10 } },
                 grid: { color: "rgba(56,242,255,0.06)" } }
        },
        plugins: { legend: { display: false }, tooltip: { enabled: false } }
    }, extra || {});
}

const envChart = new Chart(document.getElementById("chart-env"), {
    type: "line",
    data: {
        labels: [],
        datasets: [
            { label: "Temp", data: [], borderColor: "#38f2ff", borderWidth: 2, pointRadius: 0, tension: 0.35, yAxisID: "y" },
            { label: "Pressure", data: [], borderColor: "#ffb84d", borderWidth: 2, pointRadius: 0, tension: 0.35, yAxisID: "y1", hidden: false },
        ]
    },
    options: baseOptions({
        scales: {
            x: { display: false },
            y: { position: "left", ticks: { color: "#38f2ff", font: { family: "Consolas", size: 10 } }, grid: { color: "rgba(56,242,255,0.06)" } },
            y1: { position: "right", ticks: { color: "#ffb84d", font: { family: "Consolas", size: 10 } }, grid: { drawOnChartArea: false } }
        }
    })
});

const accelChart = new Chart(document.getElementById("chart-accel"), {
    type: "line",
    data: { labels: [], datasets: [
        { data: [], borderColor: "#ff6b6b", borderWidth: 2, pointRadius: 0, tension: 0.3 },
        { data: [], borderColor: "#4dd4ff", borderWidth: 2, pointRadius: 0, tension: 0.3 },
        { data: [], borderColor: "#ffd166", borderWidth: 2, pointRadius: 0, tension: 0.3 },
    ]},
    options: baseOptions()
});

const gyroChart = new Chart(document.getElementById("chart-gyro"), {
    type: "line",
    data: { labels: [], datasets: [
        { data: [], borderColor: "#ff6b6b", borderWidth: 2, pointRadius: 0, tension: 0.3 },
        { data: [], borderColor: "#4dd4ff", borderWidth: 2, pointRadius: 0, tension: 0.3 },
        { data: [], borderColor: "#ffd166", borderWidth: 2, pointRadius: 0, tension: 0.3 },
    ]},
    options: baseOptions()
});

function fmt(n, d = 2) {
    return (typeof n === "number" && !Number.isNaN(n)) ? n.toFixed(d) : "--";
}

function tickClock() {
    const el = document.getElementById("mission-clock");
    el.textContent = new Date().toUTCString().split(" ")[4] + " UTC";
}
setInterval(tickClock, 1000);
tickClock();

let lastGoodAt = 0;

async function updateData() {
    try {
        const response = await fetch("/api/data");
        const data = await response.json();

        const ax = Number(data.accel_x || 0);
        const ay = Number(data.accel_y || 0);
        const az = Number(data.accel_z || 0);
        const roll = Math.atan2(ay, az) * 180 / Math.PI;
        const pitch = Math.atan2(-ax, Math.sqrt(ay * ay + az * az)) * 180 / Math.PI;
        const yaw = Number(data.mag_heading || 0);
        document.getElementById("roll-value").textContent = roll.toFixed(1) + "°";
        document.getElementById("pitch-value").textContent = pitch.toFixed(1) + "°";
        document.getElementById("yaw-value").textContent = yaw.toFixed(1) + "°";
        document.getElementById("heading-value").textContent = yaw.toFixed(1);
        fetch("/api/mars").then(r => r.json()).then(m => {
            document.getElementById("mars-x").textContent = m.x;
            document.getElementById("mars-y").textContent = m.y;
            document.getElementById("mars-status").textContent = !m.detected ? "NO MARS" : (m.x === 0 && m.y === 0 ? "CENTERED" : (m.x === 1 ? "POINTING LEFT" : "POINTING RIGHT"));
        }).catch(() => {});
        document.getElementById("compass-needle").style.transform = `rotate(${yaw}deg)`;

        document.getElementById("temperature").textContent = fmt(data.temperature, 1);
        document.getElementById("pressure").textContent = fmt(data.pressure, 1);
        document.getElementById("altitude").textContent = fmt(data.altitude, 1);
        document.getElementById("distance").textContent = fmt(data.distance, 1);
        document.getElementById("rtc-time").textContent = data.time || "--:--:--";

        document.getElementById("accel_x").textContent = fmt(data.accel_x);
        document.getElementById("accel_y").textContent = fmt(data.accel_y);
        document.getElementById("accel_z").textContent = fmt(data.accel_z);
        document.getElementById("gyro_x").textContent = data.gyro_x ?? "--";
        document.getElementById("gyro_y").textContent = data.gyro_y ?? "--";
        document.getElementById("gyro_z").textContent = data.gyro_z ?? "--";

        pushHist(data);

        envChart.data.labels = hist.labels;
        envChart.data.datasets[0].data = hist.temp;
        envChart.data.datasets[1].data = hist.press;
        envChart.update();

        accelChart.data.labels = hist.labels;
        accelChart.data.datasets[0].data = hist.ax;
        accelChart.data.datasets[1].data = hist.ay;
        accelChart.data.datasets[2].data = hist.az;
        accelChart.update();

        gyroChart.data.labels = hist.labels;
        gyroChart.data.datasets[0].data = hist.gx;
        gyroChart.data.datasets[1].data = hist.gy;
        gyroChart.data.datasets[2].data = hist.gz;
        gyroChart.update();

        const badge = document.getElementById("link-badge");
        const text = document.getElementById("link-text");
        const isFresh = data.time && data.time !== "--:--:--";

        if (isFresh) {
            lastGoodAt = Date.now();
            badge.classList.add("online");
            text.textContent = "TELEMETRY LOCKED";
            document.getElementById("esp-status").textContent = "ACTIVE";
        } else if (Date.now() - lastGoodAt > 6000) {
            badge.classList.remove("online");
            text.textContent = "ACQUIRING SIGNAL";
            document.getElementById("esp-status").textContent = "STANDBY";
        }

    } catch (error) {
        const badge = document.getElementById("link-badge");
        badge.classList.remove("online");
        document.getElementById("link-text").textContent = "LINK LOST";
        document.getElementById("esp-status").textContent = "OFFLINE";
    }
}


setInterval(updateData, 1000);
updateData();

</script>

</body>
</html>
""")


@app.route("/sensor", methods=["POST"])
def receive_sensor():

    global sensor_data

    data = None

    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "status": "error",
                "message": "Invalid JSON"
            }), 400

        required_fields = [
            "time",
            "temperature",
            "pressure",
            "altitude",
            "distance",
            "accel_x",
            "accel_y",
            "accel_z",
            "gyro_x",
            "gyro_y",
            "gyro_z"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({
                    "status": "error",
                    "message": f"Missing field: {field}"
                }), 400

        with sensor_lock:
            sensor_data = {
                "time": data["time"],
                "temperature": float(data["temperature"]),
                "pressure": float(data["pressure"]),
                "altitude": float(data["altitude"]),
                "distance": float(data["distance"]),
                "accel_x": float(data["accel_x"]),
                "accel_y": float(data["accel_y"]),
                "accel_z": float(data["accel_z"]),
                "gyro_x": int(data["gyro_x"]),
                "gyro_y": int(data["gyro_y"]),
                "gyro_z": int(data["gyro_z"])
            }
            current_heading = sensor_data.get("mag_heading", 0.0)

        run_predictions(current_heading, data["distance"])

        print(
            f"DATA | "
            f"T={data['temperature']} C | "
            f"P={data['pressure']} hPa | "
            f"ALT={data['altitude']} m | "
            f"TOF={data['distance']} cm"
        )

        return jsonify({
            "status": "ok"
        })

    except Exception as e:

        print("Sensor error:", e)

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/mars")
def api_mars():
    with frame_lock:
        return jsonify(mars_detection)


@app.route("/api/data")
def api_data():

    with sensor_lock:
        return jsonify(sensor_data)


@app.route("/api/predictions")
def api_predictions():
    with prediction_lock:
        return jsonify(model_predictions)


def generate_frames():

    while True:

        with frame_lock:
            frame = latest_frame

        if frame is not None:

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/health")
def health():

    return jsonify({
        "flask": "ok",
        "camera": latest_frame is not None,
        "bilstm_loaded": bilstm_model is not None,
        "qrf_loaded": qrf_model is not None
    })


if __name__ == "__main__":

    print("======================================")
    print(" OCEANEmbed MISSION CONSOLE")
    print("======================================")
    print("Dashboard: http://0.0.0.0:5000")
    print("Sensor endpoint: /sensor")
    print("Camera endpoint: /video_feed")
    print("Predictions endpoint: /api/predictions")
    print("Camera: LIVE FEED ONLY")
    print("======================================")

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False
    )
