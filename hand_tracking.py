# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # Dictionary to map landmark IDs to text labels
# FINGER_LABELS = {
#     4: "Thumb",
#     8: "Index",
#     12: "Middle",
#     16: "Ring",
#     20: "Pinky"
# }

# # Global variable to hold the latest detection results safely across threads
# latest_result = None

# def save_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     global latest_result
#     latest_result = result

# # Configure the Landmarker Options
# base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
# options = vision.HandLandmarkerOptions(
#     base_options=base_options,
#     running_mode=vision.RunningMode.LIVE_STREAM,
#     num_hands=2,
#     result_callback=save_result
# )

# # Setup Video Capture
# cap = cv2.VideoCapture(0)

# def track_hand():
#     # Move the timestamp initialization inside the function to avoid global scoping errors
#     timestamp = 0 
    
#     # Open the text file in write mode ('w') to clear old data at start
#     with vision.HandLandmarker.create_from_options(options) as detector, open("finger_pos.txt", "w") as f:
#         print("Camera active. Writing data to 'finger_pos.txt'. Press 'q' to quit.")
#         f.write("Timestamp(ms),Finger,X,Y\n")
        
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 continue

#             # Mirror view
#             frame = cv2.flip(frame, 1)
#             h, w, _ = frame.shape

#             # Convert frame format for MediaPipe
#             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
#             detector.detect_async(mp_image, timestamp)
#             timestamp += 1  # Increments perfectly now!

#             # Check if we have hand detections to draw and log
#             if latest_result and latest_result.hand_landmarks:
#                 for hand_landmarks in latest_result.hand_landmarks:
#                     for idx, landmark in enumerate(hand_landmarks):
#                         cx, cy = int(landmark.x * w), int(landmark.y * h)
                        
#                         # Draw small green joints on every landmark point
#                         cv2.circle(frame, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

#                         # If this landmark index belongs to a finger tip, label and log it
#                         if idx in FINGER_LABELS:
#                             label = FINGER_LABELS[idx]
                            
#                             # Draw text near the fingertip
#                             cv2.putText(
#                                 frame,
#                                 label,
#                                 (cx - 20, cy - 15),
#                                 cv2.FONT_HERSHEY_SIMPLEX,
#                                 2,
#                                 (255, 0, 0),
#                                 4,
#                                 cv2.LINE_AA
#                             )
                            
#                             # Write coordinates into the text file
#                             f.write(f"{timestamp},{label},{cx},{cy}\n")

#             # Display the live window
#             cv2.imshow('Fingertip Labeling Stream', frame)

#             if cv2.waitKey(5) & 0xFF == ord('q'):
#                 break

#     # Make sure resources clean up safely when exiting the loop
#     cap.release()
#     cv2.destroyAllWindows()




# BASE CODE FOR HAND TRACKING USING MEDIAPIPE
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Define the callback function to receive live detection results
def print_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # This function is triggered every time a hand is detected in a frame.
    # We can access the coordinates using result.hand_landmarks
    if result.hand_landmarks:
        print(f"Hand landmarks detected at timestamp: {timestamp_ms}ms")

# 2. Configure the Landmarker for Live Stream Mode
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM, # <-- Crucial for webcam
    num_hands=2,
    result_callback=print_result # <-- Connects the callback function
)

# 3. Initialize the webcam feed and detector
cap = cv2.VideoCapture(0) # 0 is usually the default built-in camera
timestamp = 0

with vision.HandLandmarker.create_from_options(options) as detector:
    print("Camera active. Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a mirrored selfie-view experience
        frame = cv2.flip(frame, 1)

        # Convert the BGR OpenCV image to RGB MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Send the frame to the detector with an incrementing timestamp
        detector.detect_async(mp_image, timestamp)
        timestamp += 1

        # Display the live window
        cv2.imshow('MediaPipe Hand Tracking', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Clean up resources
cap.release()
cv2.destroyAllWindows()