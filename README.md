# AI-Powered Robotic Hand

A personal engineering and computer vision project that mirrors human hand movements in real time using a 3D-printed mechanical hand powered by 15 independent servo motors.

## 📖 Overview

Combining artificial intelligence, computer vision, and mechanical engineering, this project tracks hand movements through a standard webcam and translates them into physical motion. Using **OpenCV** and **MediaPipe Tasks (`HandLandmarker`)**, the software captures real-time spatial coordinates for all 21 hand landmarks frame-by-frame. By applying vector geometry and trigonometry to these landmarks, the system calculates the exact joint angles needed to independently drive 15 servos across all five fingers on the 3D-printed hand.

### Current Status
* **Software & Tracking:** Complete and fully functional. Real-time async tracking, 15-joint trigonometric angle calculation, virtual servo simulation, HUD visualization, and movement data logging (`finger_pos.txt`) have been tested and verified.
* **Hardware & Mechanics:** Currently in development.

## Tech Stack & Hardware

* **Computer Vision & AI:** Python, OpenCV (`cv2`), MediaPipe Tasks Vision (`hand_landmarker.task`)
* **Math & Processing:** NumPy (Vector dot products, norms, and linear interpolation)
* **Microcontroller:** Arduino
* **Actuators:** 15 Micro/Standard Servos (3 per finger: MCP, PIP, and DIP joints, 180° range of motion)
* **CAD & Prototyping:** Blender / Autodesk Fusion 360, 3D Printing

## How It Works

1. **Async Video Capture & Landmark Detection:** OpenCV captures live webcam frames and feeds them asynchronously into MediaPipe's `HandLandmarker` (`LIVE_STREAM` mode) to detect 21 3D hand coordinates.
2. **Trigonometric Angle Calculation:** For each of the 5 fingers, vector math (`calculate_angle`) computes the bend angles at the **MCP** (knuckle), **PIP** (middle), and **DIP** (top) joints using dot products and arccosine formulas.
3. **Servo Mapping & Jitter Filtering:** 
   * Human joint angles ($70^\circ\text{--}175^\circ$) are linearly interpolated into standard servo output angles ($180^\circ\text{--}0^\circ$).
   * A `virtual_servo_registry` tracks historical positions and applies a $2^\circ$ deadzone threshold alongside a 50ms hardware-loop throttle to prevent servo jitter.
4. **Live HUD & Movement Logging:** 
   * An OpenCV dashboard renders the hand skeleton and real-time target angles (`S[ID]:[Angle]deg`) directly over each joint.
   * When exited (`q`), the program processes the session's `movement_history` and exports fingertip displacement deltas ($\Delta X, \Delta Y$) over time to `finger_pos.txt`.

## Setup & Usage

### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install opencv-python mediapipe numpy
```
*Note: Ensure the `hand_landmarker.task` model file is located in your project directory.*

### Running the Tracker
```bash
python Move_Hand.py
```
* Press **`q`** while the camera window is active to quit and automatically generate the `finger_pos.txt` movement log.

## Future Roadmap & Vision

* **3D-Printed Pivoting Arm:** Design a modular, 3D-printed arm in Blender or Fusion 360 featuring flexible, axle-connected joints and dedicated servo housing capable of 180-degree single-axis movement.
* **Integrated Electronics:** Build a dedicated chamber near the base of the arm to securely house the Arduino and wiring, transitioning from the virtual servo simulator to live Serial communication.
* **Movement Playback & Prosthetic Applications:** Use the recorded coordinate and displacement data (`finger_pos.txt`) to play back specific movement sequences. The ultimate goal is to adapt this technology for prosthetics—allowing users to easily replicate everyday actions like opening doors, twisting doorknobs, and grasping objects.
