import cv2
import numpy as np
import imageio
import round_rectangle

overhead_presses = 0
count_on = False

# Load GIF frames
gif_path = './GIF/overhead_press.gif'
gif = imageio.mimread(gif_path)
gif_frames = [cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR) for frame in gif]
gif_frames = [cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)) for frame in gif_frames]
gif_index = 0

def check_pose_landmarks(landmarks, indices):
    return all(landmarks[i].visibility>0.5 for i in indices)

def count_overhead_presses(frame,results,overhead_presses,count_on):
    required_landmarks = [11,12,13,14,15,16]
    if results.pose_landmarks:
        if check_pose_landmarks(results.pose_landmarks.landmark,required_landmarks):
            lm11x = results.pose_landmarks.landmark[11].x
            lm12x = results.pose_landmarks.landmark[12].x
            lm13x = results.pose_landmarks.landmark[13].x
            lm14x = results.pose_landmarks.landmark[14].x
            lm15x = results.pose_landmarks.landmark[15].x
            lm16x = results.pose_landmarks.landmark[16].x

            lm11y = results.pose_landmarks.landmark[11].y
            lm12y = results.pose_landmarks.landmark[12].y
            lm13y = results.pose_landmarks.landmark[13].y
            lm14y = results.pose_landmarks.landmark[14].y
            lm15y = results.pose_landmarks.landmark[15].y
            lm16y = results.pose_landmarks.landmark[16].y

            distance = abs(lm12x-lm11x)

            if abs(lm14y-lm12y)<distance/2 and abs(lm16y-lm12y)<distance*4 and \
               abs(lm14x-lm12x)>distance/2 and abs(lm14x-lm16x)<distance and \
               abs(lm13y-lm11y)<distance/2 and abs(lm15y-lm11y)<distance*4 and \
               abs(lm13x-lm11x)>distance/2 and abs(lm13x<lm15x)<distance :
                count_on= True


            if lm16y<lm14y<lm12y and abs(lm16x-lm12x)<distance/3 and abs(lm14x-lm12x)<distance/4 and \
               lm15y<lm13y<lm11y and abs(lm15x-lm11x)<distance/3 and abs(lm13x-lm11x)<distance/4 and count_on==True :
                overhead_presses += 1
                count_on = False

    round_rectangle.draw_rounded_rectangle(frame,(10,10),(470,80),(255, 178, 90),-1,30)
    round_rectangle.draw_rounded_rectangle(frame,(10,10),(470,80),(0,0,0),1,30)
    cv2.putText(frame,'Overhead Dumbbell Presses: '+str(overhead_presses),(25,50),
                cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,0,0),2,cv2.LINE_AA)

    # Display GIF
    global gif_index
    gif_frame = gif_frames[gif_index]
    gif_height, gif_width, _ = gif_frame.shape
    start_x = 380 
    start_y = 92  
    end_x = start_x + gif_width
    end_y = start_y + gif_height
    frame[start_y:end_y, start_x:end_x] = gif_frame

    gif_index = (gif_index + 4) % len(gif_frames)
        
    return overhead_presses, count_on


    
    
