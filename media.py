# required libraries
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_to_coco_map = {
    0: 'nose',              # NOSE -> 'nose'
    2: 'left_eye',          # LEFT_EYE -> 'left_eye'
    5: 'right_eye',         # RIGHT_EYE -> 'right_eye'
    7: 'left_ear',          # LEFT_EAR -> 'left_ear'
    8: 'right_ear',         # RIGHT_EAR -> 'right_ear'
    11: 'left_shoulder',    # LEFT_SHOULDER -> 'left_shoulder'
    12: 'right_shoulder',   # RIGHT_SHOULDER -> 'right_shoulder'
    13: 'left_elbow',       # LEFT_ELBOW -> 'left_elbow'
    14: 'right_elbow',      # RIGHT_ELBOW -> 'right_elbow'
    15: 'left_wrist',       # LEFT_WRIST -> 'left_wrist'
    16: 'right_wrist',      # RIGHT_WRIST -> 'right_wrist'
    23: 'left_hip',         # LEFT_HIP -> 'left_hip'
    24: 'right_hip',        # RIGHT_HIP -> 'right_hip'
    25: 'left_knee',        # LEFT_KNEE -> 'left_knee'
    26: 'right_knee',       # RIGHT_KNEE -> 'right_knee'
    27: 'left_ankle',       # LEFT_ANKLE -> 'left_ankle'
    28: 'right_ankle'       # RIGHT_ANKLE -> 'right_ankle'
}
cap = cv2.VideoCapture("sunt_video2.mp4")
while cap.isOpened():
    # read frame
    _, frame = cap.read()
    try:
        # resize the frame for portrait video
        # frame = cv2.resize(frame, (350, 600))
        # convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # process the frame for pose detection
        pose_results = pose.process(frame_rgb)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            import IPython; IPython.embed()
            # Extract only 17 keypoints based on the COCO format
            coco_keypoints = {}
            for mp_index, coco_index in mp_to_coco_map.items():
                coco_keypoints[coco_index] = landmarks[mp_index]
            
            print("COCO format keypoints:")
            for idx, kp in coco_keypoints.items():
                print(f"Keypoint {idx}: ({kp.x}, {kp.y}, {kp.z})")
                
            
            # Draw pose landmarks using Mediapipe for visualization
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # print(pose_results.pose_landmarks)
        # print(pose_results)
        # draw skeleton on the frame
        # mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # display the frame
        cv2.imshow('Output', frame)
    except:
        break
        
    if cv2.waitKey(1) == ord('q'):
        break
          
cap.release()
cv2.destroyAllWindows()