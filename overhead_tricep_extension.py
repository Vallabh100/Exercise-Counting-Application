import cv2
import numpy as np
import imageio
import round_rectangle

overhead_tricep_extension = 0
count_on = False

# Load GIF frames
gif_path = './GIF/overhead_tricep_extension.gif'
gif = imageio.mimread(gif_path)
gif_frames = [cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR) for frame in gif]
gif_frames = [cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3)) for frame in gif_frames]
gif_index = 0

#function for checking all landmarks are present or not
def check_pose_landmarks(landmarks, indices):
    return all(landmarks[i].visibility > 0.5 for i in indices)

#function for counting overhead tricep extensions
def count_overhead_tricep_extensions(frame,results,overhead_tricep_extension,count_on):
        required_landmarks = [11,12,13,14]
        not_required_landmarks = [15,16]
        if results.pose_landmarks:
            if check_pose_landmarks(results.pose_landmarks.landmark, required_landmarks):
                lm11y = results.pose_landmarks.landmark[11].y
                lm12y = results.pose_landmarks.landmark[12].y
                lm13y = results.pose_landmarks.landmark[13].y
                lm14y = results.pose_landmarks.landmark[14].y
                if check_pose_landmarks(results.pose_landmarks.landmark, not_required_landmarks):
                    lm15y = results.pose_landmarks.landmark[15].y
                    lm16y = results.pose_landmarks.landmark[16].y


                if lm14y<lm12y and lm13y<lm11y \
                   and check_pose_landmarks(results.pose_landmarks.landmark, not_required_landmarks):
                    count_on = True
                     

                if not(check_pose_landmarks(results.pose_landmarks.landmark, not_required_landmarks)) \
                   and (lm14y<lm12y) and (lm13y<lm11y) and count_on==True:
                        overhead_tricep_extension += 1
                        count_on = False

        round_rectangle.draw_rounded_rectangle(frame,(10,10),(470,80),(255, 178, 90),-1,30)
        round_rectangle.draw_rounded_rectangle(frame,(10,10),(470,80),(0,0,0),1,30)
        cv2.putText(frame,'Overhead Tricep Extensions: '+str(overhead_tricep_extension),(15,50),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2,cv2.LINE_AA)

        # Display GIF
        global gif_index
        gif_frame = gif_frames[gif_index]
        gif_height, gif_width, _ = gif_frame.shape
        start_x = 410  
        start_y = 100  
        end_x = start_x + gif_width
        end_y = start_y + gif_height
        frame[start_y:end_y, start_x:end_x] = gif_frame

        gif_index = (gif_index + 3) % len(gif_frames)
        
        return overhead_tricep_extension, count_on

