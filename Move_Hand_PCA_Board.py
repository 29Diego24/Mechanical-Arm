import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import serial

# ==========================================
# CONFIGURE YOUR ARDUINO SERIAL PORT HERE
# ==========================================
try:
    # Run 'ls /dev/cu.usbmodem*' in terminal to find your exact port name
    arduino = serial.Serial(port='/dev/cu.usbmodem101', baudrate=115200, timeout=0.1)
    time.sleep(2) # Give Arduino time to reset safely
    print("[SUCCESS] Connected to Arduino Uno.")
except Exception as e:
    print(f"[WARNING] Serial connection failed: {e}. Running in simulation mode.")
    arduino = None

# Tracking groups for all 5 fingers
# Each list contains: [Knuckle/MCP, Middle/PIP, Top/DIP, Tip]
FINGER_JOINTS = {
    "Thumb":  [1, 2, 3, 4],
    "Index":  [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring":   [13, 14, 15, 16],
    "Pinky":  [17, 18, 19, 20]
}

# MediaPipe structural connections mapping to draw skeleton lines
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17)             # Palm base
]

# Track movement history for saving to file
movement_history = {}
latest_result = None
last_send_time = 0

# Internal State Array: Holds the last registered servo angle for every joint index
# Index 0 through 20 (21 total slots) to handle individual joint deadzones
virtual_servo_registry = [-1] * 21

def save_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# --- Map MediaPipe ID to a clean PCA9685 Channel (0 to 14) ---
def get_pca9685_channel(joint_id):
    mapping = {
        1: 0,  2: 1,  3: 2,   # Thumb
        5: 3,  6: 4,  7: 5,   # Index
        9: 6,  10:7,  11:8,   # Middle
        13:9,  14:10, 15:11,  # Ring
        17:12, 18:13, 19:14   # Pinky
    }
    return mapping.get(joint_id, None)

# Initialize Landmarker options with Fixed Absolute Path
base_options = python.BaseOptions(
    model_asset_path='/Users/david/Documents/Python Projects/AI Project/AI-Project/hand_landmarker.task'
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=save_result
)

cap = cv2.VideoCapture(0)

def calculate_angle(p1, p2, p3):
    """Calculates the angle (in degrees) at joint p2 between lines p2-p1 and p2-p3"""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]]) 
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]]) 
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
        
    cosine_angle = dot_product / (norm_v1 * norm_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return int(np.degrees(np.arccos(cosine_angle)))

def update_virtual_servos(updated_joints):
    """Sends the data over the hardware serial line in standard format: <channel,angle>"""
    global last_send_time, arduino
    current_time = time.time()
    
    # Throttle processing to 40ms intervals (~25 FPS) to prevent buffering lag
    if current_time - last_send_time < 0.04:
        return

    if updated_joints:
        if arduino:
            for joint_id, ang in updated_joints:
                channel = get_pca9685_channel(joint_id)
                if channel is not None:
                    packet = f"<{channel},{ang}>\n"
                    arduino.write(packet.encode('utf-8'))
        else:
            # Fallback local display print if serial device isn't plugged in
            log_output = [f"Servo[{jid}]: {ang}°" for jid, ang in updated_joints]
            print(f"[SIMULATED HARDWARE] -> " + " | ".join(log_output))
            
        last_send_time = current_time

def track_hand():
    timestamp = 0 
    
    with vision.HandLandmarker.create_from_options(options) as detector:
        print("Camera running. Hand tracking active. Press 'q' to quit.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detector.detect_async(mp_image, timestamp)

            if latest_result and latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    
                    # Map out absolute screen coordinates
                    coords = {idx: (int(landmark.x * w), int(landmark.y * h)) for idx, landmark in enumerate(hand_landmarks)}
                    movement_history[timestamp] = coords

                    # Draw structural skeleton lines
                    for connection in HAND_CONNECTIONS:
                        start_pt = coords[connection[0]]
                        end_pt = coords[connection[1]]
                        cv2.line(frame, start_pt, end_pt, (200, 200, 200), 2)

                    # Gather updates across all fingers during this specific frame
                    frame_servo_updates = []

                    for name, IDs in FINGER_JOINTS.items():
                        # Determine base/anchor reference points
                        p_base = coords[IDs[0] - 1] if name == "Thumb" else coords[0] 
                        p_mcp, p_pip, p_dip, p_tip = coords[IDs[0]], coords[IDs[1]], coords[IDs[2]], coords[IDs[3]]

                        # Calculate absolute angles for all physical flexing joints
                        angle_mcp = calculate_angle(p_base, p_mcp, p_pip)
                        angle_pip = calculate_angle(p_mcp, p_pip, p_dip)
                        angle_dip = calculate_angle(p_pip, p_dip, p_tip)

                        # Match the physical MediaPipe IDs directly to their target angles
                        target_joints = [
                            (IDs[0], angle_mcp), # MCP Joint
                            (IDs[1], angle_pip), # PIP Joint
                            (IDs[2], angle_dip)  # DIP Joint
                        ]

                        for joint_id, calculated_angle in target_joints:
                            # Map full biological joint movement (70°-175°) to standard Servo output (180° down to 0°)
                            mapped_angle = int(np.interp(calculated_angle, [70, 175], [180, 0]))
                            servo_angle = max(0, min(180, mapped_angle))

                            # Evaluate each joint against its own historical deadzone threshold (3 degrees to stop jitter)
                            if abs(servo_angle - virtual_servo_registry[joint_id]) > 3:
                                frame_servo_updates.append((joint_id, servo_angle))
                                virtual_servo_registry[joint_id] = servo_angle

                        # Render every key tracking point as a solid color dot
                        cv2.circle(frame, p_mcp, 6, (0, 255, 255), cv2.FILLED)
                        cv2.circle(frame, p_pip, 6, (0, 255, 0), cv2.FILLED)
                        cv2.circle(frame, p_dip, 6, (255, 100, 0), cv2.FILLED)
                        cv2.circle(frame, p_tip, 6, (255, 255, 255), cv2.FILLED)

                        # On-Screen HUD Labels
                        cv2.putText(frame, f"S{IDs[0]}:{virtual_servo_registry[IDs[0]]}deg", (p_mcp[0] + 10, p_mcp[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"S{IDs[1]}:{virtual_servo_registry[IDs[1]]}deg", (p_pip[0] + 10, p_pip[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"S{IDs[2]}:{virtual_servo_registry[IDs[2]]}deg", (p_dip[0] + 10, p_dip[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 0), 1, cv2.LINE_AA)
                        cv2.putText(frame, name, (p_tip[0] - 15, p_tip[1] - 12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                    # Trigger processing update
                    if frame_servo_updates:
                        update_virtual_servos(frame_servo_updates)

            cv2.imshow('15-Servo Independent Joint Mapping Dashboard', frame)
            timestamp += 1

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    analyze_recorded_movements()

def analyze_recorded_movements():
    """Loops back through data and logs it into a file after execution ceases."""
    print("\n==============================================")
    print("      SAVING MOVEMENT DATA TO finger_pos.txt    ")
    print("==============================================\n")
    timestamps = sorted(movement_history.keys())
    if len(timestamps) < 2:
        return

    with open("finger_pos.txt", "w") as f:
        f.write("Time_Start(ms),Time_End(ms),Finger,Delta_X(px),Delta_Y(px)\n")
        for i in range(1, len(timestamps)):
            t_prev, t_curr = timestamps[i - 1], timestamps[i]
            prev_frame, curr_frame = movement_history[t_prev], movement_history[t_curr]
            
            for finger_name, joint_ids in FINGER_JOINTS.items():
                tip_id = joint_ids[3]
                if tip_id in prev_frame and tip_id in curr_frame:
                    dx = curr_frame[tip_id][0] - prev_frame[tip_id][0]
                    dy = curr_frame[tip_id][1] - prev_frame[tip_id][1]
                    f.write(f"{t_prev},{t_curr},{finger_name},{dx},{dy}\n")

    print("[SUCCESS] Tracking logs written safely to 'finger_pos.txt'!")

if __name__ == "__main__":
    track_hand()