                "time_sec": time_sec,
                "TOF_Alt": tof_alt,
                "Magx": mag_x,
                "Magy": mag_y,
                "Magz": mag_z,
                "Mag_heading": mag_heading,
                "accelx": accel_x,
                "accely": accel_y,
                "accelz": accel_z,
                "angle roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "ldr": ldr
            }

        run_predictions(model_input_data)
        run_bilstm_prediction(model_input_data)

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
        current_mars = dict(mars_detection)
    return jsonify(current_mars)

@app.route("/api/data")
def api_data():

    with sensor_lock:
        return jsonify(sensor_data)

@app.route("/api/predictions")
def api_predictions():
    with prediction_lock:
        return jsonify(model_predictions)

# ************************************************************
#                    VIDEO STREAMING
# ************************************************************

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
        "qrf_loaded": qrf_model is not None,
        "bilstm_loaded": bilstm_model is not None,
        "bilstm_window": BILSTM_WINDOW,
        "model_features": FEATURE_COLUMNS,
        "qrf_input_shape": [1, N_FEATURES]
    })

import atexit
atexit.register(GPIO.cleanup)

# ************************************************************
#                    PROGRAM CONTROL
# ************************************************************

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
