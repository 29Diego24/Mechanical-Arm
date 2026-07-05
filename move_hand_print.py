import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import serial
import time

# Serial setup
try:
    print("Connecting to Arduino...")
    arduino_port = '/dev/cu.usbmodem14101'  # Change to your exact port
    arduino = serial.Serial(port=arduino_port, baudrate=9600, timeout=0.1)
    time.sleep(2)
    print("Arduino connected successfully!")
except Exception as e:
    print(f"Could not connect to Arduino: {e}\nRunning in simulation mode.")
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

# create a dictionary to hold the movement history for each finger for each timestamp
movement_history = {}

latest_result = None

def save_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
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
    
    angle = np.arccos(cosine_angle)
    return int(np.degrees(angle))

def track_hand():
    timestamp = 0 
    
    with vision.HandLandmarker.create_from_options(options) as detector:
        print("Camera active. Displaying all joint and tip angles. Press 'q' to quit.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detector.detect_async(mp_image, timestamp)
            timestamp += 1


            if latest_result and latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    
                    # Map out absolute screen coordinates
                    coords = {}
                    for idx, landmark in enumerate(hand_landmarks):
                        coords[idx] = (int(landmark.x * w), int(landmark.y * h))
                        # Draw every key tracking point as a solid red circle
                        cv2.circle(frame, coords[idx], 8, (0, 0, 255), cv2.FILLED)

                    movement_history[timestamp] = coords

                    # Iterate through each finger
                    for name, IDs in FINGER_JOINTS.items():
                        p_base = coords[IDs[0] - 1] if name == "Thumb" else coords[0] # Wrist/Palm anchor
                        p_mcp  = coords[IDs[0]]
                        p_pip  = coords[IDs[1]]
                        p_dip  = coords[IDs[2]]
                        p_tip  = coords[IDs[3]]

                        # Calculate Angles for internal joints
                        angle_mcp = calculate_angle(p_base, p_mcp, p_pip)
                        angle_pip = calculate_angle(p_mcp, p_pip, p_dip)
                        angle_dip = calculate_angle(p_pip, p_dip, p_tip)
                        
                        # Calculate Angle for the terminal Tip relative to the PIP joint and DIP joint
                        angle_tip = calculate_angle(p_pip, p_dip, p_tip)

                        # Render joint values on screen next to their respective points
                        cv2.putText(frame, f"{angle_mcp}deg", (p_mcp[0] + 8, p_mcp[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"{angle_pip}deg", (p_pip[0] + 8, p_pip[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"{angle_dip}deg", (p_dip[0] + 8, p_dip[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2, cv2.LINE_AA)

                        # Display both the finger name label AND the calculated tip orientation angle
                        cv2.putText(frame, f"{name}: {angle_tip}deg", (p_tip[0] - 25, p_tip[1] - 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        # Stream Data packets over USB connection
                        if name == "Index" and arduino:
                            clamped_angle = max(0, min(180, angle_pip))
                            arduino.write(f"{clamped_angle}\n".encode('utf-8'))

            cv2.imshow('Complete 20-Point Angular Tracking HUD', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()
    # playback data calculations immediately after the feed stops
    analyze_recorded_movements()


# PLAYBACK & MOVEMENT CALCULATION
def analyze_recorded_movements():
    """
    Loops back through the recorded tracking history dictionary 
    and calculates the coordinate changes between frames.
    """
    print("\n==============================================")
    print("      PROCESSING RECORDED MOVEMENTS PLAYBACK    ")
    print("==============================================\n")
    
    # Sort timestamps to guarantee sequential playback order
    timestamps = sorted(movement_history.keys())
    
    if len(timestamps) < 2:
        print("Not enough tracking data recorded to calculate changes.")
        return


    # Loop starting from the second frame so we can compare it to the one before it
    for i in range(1, len(timestamps)):
        t_prev = timestamps[i - 1]
        t_curr = timestamps[i]
        
        prev_frame = movement_history[t_prev]
        curr_frame = movement_history[t_curr]
        
        print(f"--- Movement from Timestamp {t_prev}ms -> {t_curr}ms ---")
        
        # Calculate coordinate shifts for specific key points (e.g., fingertips)
        # ID 8 = Index Tip, ID 12 = Middle Tip, ID 4 = Thumb Tip
        for finger_name, joint_ids in FINGER_JOINTS.items():
            tip_id = joint_ids[3] # Grab the Tip ID
            
            # Extract historical coordinate states from RAM dictionary
            x_prev, y_prev = prev_frame[tip_id]
            x_curr, y_curr = curr_frame[tip_id]
            
            # Calculate the physical pixel change (Delta)
            delta_x = x_curr - x_prev
            delta_y = y_curr - y_prev
            
            # Print shifts. Example: (+5, -12) means moved right 5px, up 12px
            print(f"  {finger_name:<7} Tip Shift -> Delta X: {delta_x:+4d}px | Delta Y: {delta_y:+4d}px")
        
        print("-" * 50)
        time.sleep(0.05) # Adds a slight artificial delay so the terminal doesn't flash instantly

if __name__ == "__main__":
    track_hand()