import cv2
import mediapipe as mp
import requests
import numpy as np
import time
import lateral_raises
import overhead_press
import overhead_tricep_extension
import bicep_curls
import round_rectangle

url = "http://192.0.0.4:8080/shot.jpg"
#cap = cv2.VideoCapture(2)

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity = 0
)

hand_model = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

left_bicep_curl = 0
right_bicep_curl = 0
count = 0
left_count_on = False
right_count_on = False
count_on = False
selected_exercise = None
exercise_start_time = None
menu_selected_time = None
stop_selected_time = None

# Define exercise functions
def check_pose_landmarks(landmarks, indices):
    return all(landmarks[i].visibility > 0.5 for i in indices)


def draw_menu(frame):
    h, w, _ = frame.shape

    # Exercise buttons to display
    round_rectangle.draw_rounded_rectangle(frame, (5, 10), (int(5+w/5), 110), (246,236,176), -1, 30)
    round_rectangle.draw_rounded_rectangle(frame, (5, 10), (int(5+w/5), 110), (0,0,0), 1, 30)
    cv2.putText(frame, 'Bicep', (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Curls', (16, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 1, cv2.LINE_AA)

    round_rectangle.draw_rounded_rectangle(frame, (int(10+w/5), 10), (int(10+2*w/5), 110), (255, 178, 90), -1, 30)
    round_rectangle.draw_rounded_rectangle(frame, (int(10+w/5), 10), (int(10+2*w/5), 110), (0,0,0), 1, 30)
    cv2.putText(frame, 'Lateral', (int(10+w/5)+5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.83, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Raises', (int(10+w/5)+3, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 1, cv2.LINE_AA)

    round_rectangle.draw_rounded_rectangle(frame, (int(10+2*w/5)+5, 10), (int(10+3*w/5)+35, 110), (246,236,176), -1, 30)
    round_rectangle.draw_rounded_rectangle(frame, (int(10+2*w/5)+5, 10), (int(10+3*w/5)+35, 110), (0,0,0), 1, 30)
    cv2.putText(frame, 'Overhead', (int(10+2*w/5)+10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Press', (int(10+2*w/5)+30, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 1, cv2.LINE_AA)

    round_rectangle.draw_rounded_rectangle(frame, (int(10+3*w/5)+40, 10), (int(80+4*w/5)+12, 110), (255, 178, 90), -1, 30)
    round_rectangle.draw_rounded_rectangle(frame, (int(10+3*w/5)+40, 10), (int(80+4*w/5)+12, 110), (0,0,0), 1, 30)
    cv2.putText(frame, 'Overhead', (int(10+3*w/5)+50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Tricep', (int(10+3*w/5)+75, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Extension', (int(10+3*w/5)+50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)

def detect_hand_choice(hand_landmarks, frame):
    if hand_landmarks:
        for hand in hand_landmarks:
            x = int(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            y = int(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
            h, w, _ = frame.shape

            if 5 < x < int(5+w/5) and 10 < y < 110:
                return 'Bicep Curls'
            elif int(10+w/5) < x < int(10+2*w/5) and 10 < y < 110:
                return 'Lateral Raises'
            elif int(10+2*w/5)+5 < x < int(10+3*w/5)+35 and 10 < y < 110:
                return 'Overhead Press'
            elif int(10+3*w/5)+40 < x < int(80+4*w/5)+12 and 10 < y < 110:
                return 'Overhead Tricep Extension'
    return None

# Main loop
while True:
    #"""
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)
    """
    _, frame = cap.read()
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if selected_exercise is None:
        # Menu mode
        draw_menu(frame)
        hand_results = hand_model.process(rgb_frame)
        
        choice = detect_hand_choice(hand_results.multi_hand_landmarks, frame)
        if choice:
            if menu_selected_time is None:
                menu_selected_time = time.time()
            elif time.time() - menu_selected_time > 1:
                selected_exercise = choice
                exercise_start_time = None
                left_bicep_curl = 0
                right_bicep_curl = 0
                count = 0
                left_count_on = False
                right_count_on = False
                count_on = False
                menu_selected_time = None
        else:
            menu_selected_time = None
    else:
        # Exercise mode
        results = model.process(rgb_frame)

        if selected_exercise == 'Bicep Curls':
            left_bicep_curl, right_bicep_curl, left_count_on,right_count_on = bicep_curls.count_bicep_curls(frame,results,left_bicep_curl,right_bicep_curl,left_count_on,right_count_on)
        elif selected_exercise == 'Overhead Press':
            count, count_on = overhead_press.count_overhead_presses(frame, results, count, count_on)
        elif selected_exercise == 'Overhead Tricep Extension':
            count, count_on = overhead_tricep_extension.count_overhead_tricep_extensions(frame, results, count, count_on)
        elif selected_exercise == 'Lateral Raises':
            count, count_on = lateral_raises.count_lateral_raises(frame, results, count, count_on)
        
        # Draw stop button
        round_rectangle.draw_rounded_rectangle(frame, (13, 100), (135, 150), (0, 0, 255), -1, 20)
        round_rectangle.draw_rounded_rectangle(frame, (13, 100), (135, 150), (255, 255, 255), 1, 20)
        cv2.putText(frame, 'STOP', (28, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        hand_results = hand_model.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            x = int(hand_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            y = int(hand_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
            
            if 13 < x < 135 and 100 < y < 150:
                if stop_selected_time is None:
                    stop_selected_time = time.time()
                elif time.time() - stop_selected_time > 1:
                    selected_exercise = None
                    stop_selected_time = None
            else:
                stop_selected_time = None

    cv2.imshow('Exercise Counting Application', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
