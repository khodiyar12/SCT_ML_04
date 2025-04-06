import cv2
import mediapipe as mp
import glob

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Function to process a single image and detect gestures
def process_single_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image {image_path}")
        return

    # Resize the image to standard dimensions
    image = cv2.resize(image, (640, 480))

    # Convert the image to RGB for Mediapipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe
    result = hands.process(rgb_image)

    # Check for hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks and connections on the image
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if the hand gesture is "OK"
            gesture_status = detect_ok_gesture(hand_landmarks)
            print(f"Hand detected in {image_path}: {gesture_status}")

            # Overlay gesture status text on the image
            cv2.putText(image, f"Gesture: {gesture_status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        print(f"No hand detected in the image: {image_path}")
        cv2.putText(image, "No hand detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the image with visualization
    cv2.imshow("Hand Gesture Recognition", image)
    cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()

# Function to determine if the hand gesture is "OK"
def detect_ok_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    # Check if the thumb and index finger tips are close to each other
    thumb_index_close = abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05

    # Check if other fingers are extended
    other_fingers_extended = (
        middle_tip.y < middle_mcp.y and
        ring_tip.y < ring_mcp.y and
        pinky_tip.y < pinky_mcp.y
    )

    return "OK" if thumb_index_close and other_fingers_extended else "Not OK"

# Function to process multiple images in a folder
def process_multiple_images(folder_path):
    # Collect all PNG images from the specified folder
    image_paths = glob.glob(folder_path + r'\*.png')
    if not image_paths:
        print("No PNG images found in the folder.")
        return

    for image_path in image_paths:
        print("Processing image:", image_path)
        process_single_image(image_path)

# Example: Process all images in the folder
folder_path = r'C:\Users\USER\OneDrive\Desktop\SCT_ML_04\leapGestRecog'
process_multiple_images(folder_path)
