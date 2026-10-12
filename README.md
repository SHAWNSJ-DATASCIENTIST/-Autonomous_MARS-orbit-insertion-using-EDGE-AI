<img width="1280" height="960" alt="SATELLITE" src="https://github.com/user-attachments/assets/8a7f9f51-4635-4cd5-a875-1e8be14812da" />
# Edge AI for Simulated Mars Orbit Insertion

What if a spacecraft could spot a developing trajectory problem and respond without waiting for instructions from Earth?

This project explores that idea through a low-cost hardware-in-the-loop (HIL) prototype for a simulated Mars Orbit Insertion (MOI) scenario. It brings together embedded sensing, computer vision, telemetry prediction, state estimation, and simulated actuation in one autonomous pipeline.

A Raspberry Pi 4B acts as the main edge-computing platform. A camera and onboard sensors provide visual and telemetry inputs. YOLO11n detects Mars and estimates its position in the camera view; a BiLSTM predicts future altitude values; and a Quantile Random Forest estimates correction demand with uncertainty bounds. An Extended Kalman Filter (EKF) combines the model forecast with live Time-of-Flight sensor readings to produce a more stable state estimate.

The decision layer compares predicted correction demand with a configured safety threshold. It then selects HOLD or CORRECT and, when needed, triggers simulated thruster indicators and steering motors. The goal is to demonstrate the full **sense → predict → estimate → decide → act** loop on an embedded prototype—not to model a flight-ready spacecraft.

## Project status

The Raspberry Pi 4B is the active edge-computing platform. FPGA acceleration using the Vicharak Shrike-Fi / Renesas Forge is a planned direction for future work; it should not be considered an implemented part of the current system.
