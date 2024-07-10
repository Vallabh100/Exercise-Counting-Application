import cv2
import numpy as np
import imageio
import round_rectangle


left_bicep_curls = 0
right_bicep_curls = 0
left_count_on = False
right_count_on = False

# Load GIF frames
gif_path = './GIF/bicep_curls.gif'
gif = imageio.mimread(gif_path)
gif_frames = [cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR) for frame in gif]
gif_frames = [cv2.resize(frame, (frame.shape[1] // 5, frame.shape[0] // 5)) for frame in gif_frames]
gif_index = 0


#function for checking all landmarks are present or not
def check_pose_landmarks(landmarks, indices):
    return all(landmarks[i].visibility > 0.5 for i in indices)


def count_bicep_curls(frame,results,left_bicep_curls,right_bicep_curls,left_count_on,right_count_on):
    if results.pose_landmarks:
        required_landmarks = [12, 16, 24, 26, 11, 15, 23, 25]
        if check_pose_landmarks(results.pose_landmarks.landmark, required_landmarks):
            lm12x = results.pose_landmarks.landmark[12].x
            lm16x = results.pose_landmarks.landmark[16].x
            lm24x = results.pose_landmarks.landmark[24].x
            lm26x = results.pose_landmarks.landmark[26].x
                    
            lm12y = results.pose_landmarks.landmark[12].y
            lm14y = results.pose_landmarks.landmark[14].y
            lm16y = results.pose_landmarks.landmark[16].y
            lm24y = results.pose_landmarks.landmark[24].y
            lm26y = results.pose_landmarks.landmark[26].y

            xdistance = lm24y - lm12y
            ydistance = lm26y - lm12y

            if abs(lm24x-lm16x)<= xdistance/1.5 and abs(lm24y-lm16y)<= ydistance/4.5:
                left_count_on = True

            if abs(lm16x-lm12x)<= xdistance/3.5 and abs(lm16y-lm12y)<= ydistance/9.5 and abs(lm24y-lm14y)<=xdistance/1.5 and left_count_on==True :
                left_bicep_curls = left_bicep_curls + 1
                left_count_on = False


            
            lm11x = results.pose_landmarks.landmark[11].x
            lm15x = results.pose_landmarks.landmark[15].x
            lm23x = results.pose_landmarks.landmark[23].x
            lm25x = results.pose_landmarks.landmark[25].x
                    
            lm11y = results.pose_landmarks.landmark[11].y
            lm13y = results.pose_landmarks.landmark[13].y
            lm15y = results.pose_landmarks.landmark[15].y
            lm23y = results.pose_landmarks.landmark[23].y
            lm25y = results.pose_landmarks.landmark[25].y

            xdistance = lm23y - lm11y
            ydistance = lm25y - lm11y

            if abs(lm23x-lm15x)<= xdistance/1.5 and abs(lm23y-lm15y)<= ydistance/4.5:
                right_count_on = True

            if abs(lm15x-lm11x)<= xdistance/3.5 and abs(lm15y-lm11y)<= ydistance/9.5 and abs(lm23y-lm13y)<=xdistance/1.5 and right_count_on==True :
                right_bicep_curls = right_bicep_curls + 1
                right_count_on = False

    round_rectangle.draw_rounded_rectangle(frame,(10,10),(470,90),(255, 178, 90),-1,30)
    round_rectangle.draw_rounded_rectangle(frame,(10,10),(470,90),(0,0,0),1,30)
    cv2.putText(frame,'Left  Bicep Curls: '+str(left_bicep_curls),(100,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,'Right Bicep Curls: '+str(right_bicep_curls),(100,75),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2,cv2.LINE_AA)

    # Display GIF
    global gif_index
    gif_frame = gif_frames[gif_index]
    gif_height, gif_width, _ = gif_frame.shape
    start_x = 380 
    start_y = 107  
    end_x = start_x + gif_width
    end_y = start_y + gif_height
    frame[start_y:end_y, start_x:end_x] = gif_frame

    gif_index = (gif_index + 3) % len(gif_frames)

    
    return left_bicep_curls, right_bicep_curls, left_count_on,right_count_on



