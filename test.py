from src.lstm import ActionClassificationLSTM
from src.video_analyzer_web import analyse_video

# import some common Detectron2 utilities
#from detectron2 import model_zoo
#from detectron2.engine import DefaultPredictor
#from detectron2.config import get_cfg
import time
import cv2
import numpy as np
import ntpath
from src.utils import filter_persons, draw_keypoints
from src.lstm import WINDOW_SIZE
import torch
import torch.nn.functional as F


import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

SKIP_FRAME_COUNT = 1
LABELS = {
    0: "JUMPING",
    1: "JUMPING_JACKS",
    2: "BOXING",
    3: "WAVING_2HANDS",
    4: "WAVING_1HAND",
    5: "CLAPPING_HANDS"
}

import math


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

def analyse_video(pose_detector, lstm_classifier, video_path):
    # open the video
    cap = cv2.VideoCapture(video_path)
    # width of image frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height of image frame
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frames per second of the input video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # total number of frames in the video
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # video output codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # extract the file name from video path
    file_name = ntpath.basename(video_path)
    # video writer
    vid_writer = cv2.VideoWriter('res_{}'.format(
        file_name), fourcc, 30, (width, height))
    # counter
    counter = 0
    # buffer to keep the output of detectron2 pose estimation
    buffer_window = []
    # start time
    start = time.time()
    label = None
    print("Processing video")
    # iterate through the video
    while True:
        # read the frame
        ret, frame = cap.read()
        # return if end of the video
        if ret == False:
            break
        # make a copy of the frame
        img = frame.copy()
        if(counter % (SKIP_FRAME_COUNT+1) == 0):
            # predict pose estimation on the frame
            outputs = pose_detector(frame)
            # filter the outputs with a good confidence score
            persons, pIndicies = filter_persons(outputs)
            if len(persons) >= 1:
                # pick only pose estimation results of the first person.
                # actually, we expect only one person to be present in the video.
                p = persons[0]
                # draw the body joints on the person body
                draw_keypoints(p, img)
                # input feature array for lstm
                features = []
                # add pose estimate results to the feature array
                for i, row in enumerate(p):
                    features.append(row[0])
                    features.append(row[1])

                # append the feature array into the buffer
                # not that max buffer size is 32 and buffer_window operates in a sliding window fashion
                if len(buffer_window) < WINDOW_SIZE:
                    buffer_window.append(features)
                else:
                    # convert input to tensor
                    model_input = torch.Tensor(np.array(buffer_window, dtype=np.float32))
                    # add extra dimension
                    model_input = torch.unsqueeze(model_input, dim=0)
                    # predict the action class using lstm
                    y_pred = lstm_classifier(model_input)
                    # import IPython; IPython.embed()
                    prob = F.softmax(y_pred, dim=1)
                    # get the index of the max probability
                    pred_index = prob.data.max(dim=1)[1]
                    # pop the first value from buffer_window and add the new entry in FIFO fashion, to have a sliding window of size 32.
                    buffer_window.pop(0)
                    buffer_window.append(features)
                    label = LABELS[pred_index.numpy()[0]]
                    #print("Label detected ", label)

        # add predicted label into the frame
        if label is not None:
            cv2.putText(img, 'Action: {}'.format(label),
                        (int(width-400), height-50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (102, 255, 255), 2)
        # increment counter
        counter += 1
        # write the frame into the result video
        vid_writer.write(img)
        # compute the completion percentage
        percentage = int(counter*100/tot_frames)
        # return the completion percentage
        yield "data:" + str(percentage) + "\n\n"

        # show video results
        cv2.imshow("image", img)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    analyze_done = time.time()
    print("Video processing finished in ", analyze_done - start)

#start = time.time()
# obtain detectron2's default config
#cfg = get_cfg()
# load the pre trained model from Detectron2 model zoo
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
# set confidence threshold for this model
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# load model weights
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

#cfg.MODEL.DEVICE = "cuda:0"
# create the predictor for pose estimation using the config
#pose_detector = DefaultPredictor(cfg)





#model_load_done = time.time()


lstm_classifier = ActionClassificationLSTM.load_from_checkpoint("models/epoch=394-step=17774.ckpt")
device = torch.device("cuda")
    
# Move the LSTM model to the correct device
lstm_classifier.to(device)
lstm_classifier.eval()

# analyse_video(pose_detector, lstm_classifier, )
video_path = "sunt_video2.mp4"
# open the video
cap = cv2.VideoCapture(video_path)
# width of image frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height of image frame
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frames per second of the input video
fps = int(cap.get(cv2.CAP_PROP_FPS))
# total number of frames in the video
tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# video output codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# extract the file name from video path
file_name = ntpath.basename(video_path)
# video writer
vid_writer = cv2.VideoWriter('res_{}'.format(
    file_name), fourcc, 30, (width, height))
# counter
counter = 0
# buffer to keep the output of detectron2 pose estimation
buffer_window = []
# start time
start = time.time()
label = None
print("Processing video")
# iterate through the video
while True:
    # read the frame
    ret, frame = cap.read()
    # return if end of the video
    if ret == False:
        break
    # make a copy of the frame
    img = frame.copy()
    if(counter % (SKIP_FRAME_COUNT+1) == 0):
        # predict pose estimation on the frame
        # outputs = pose_detector(frame)
        image_rows, image_cols, _ = img.shape
        pose_outputs = pose.process(frame)
        outputs = []
        persons = {}
        # persons = {}
        if pose_outputs.pose_landmarks:
            landmarks = pose_outputs.pose_landmarks.landmark
            # Extract only 17 keypoints based on the COCO format
            coco_keypoints = {}
            for mp_index, coco_index in mp_to_coco_map.items():
                coco_keypoints[coco_index] = landmarks[mp_index]

            person = torch.tensor([[min(math.floor(kp.x * image_cols), image_cols - 1), min(math.floor(kp.y * image_rows), image_rows - 1), kp.z] for kp in coco_keypoints.values()])
            persons[0] = person
            # for idx, kp in coco_keypoints.items():
            #     print(f"Keypoint {idx}: ({kp.x}, {kp.y}, {kp.z})")

            #     outputs.append([kp.x, kp.y])

        # persons, pIndicies = filter_persons(outputs)
        # import IPython; IPython.embed()

        # filter the outputs with a good confidence score
        if len(persons) >= 1:
            # pick only pose estimation results of the first person.
            # actually, we expect only one person to be present in the video.
            p = persons[0]

            # draw the body joints on the person body
            draw_keypoints(p, img)
            # input feature array for lstm
            features = []
            # add pose estimate results to the feature array
            for i, row in enumerate(p):
                # x = min(math.floor(row[0] * image_cols), image_cols - 1)
                # y = min(math.floor(row[1] * image_rows), image_rows - 1)
                
                features.append(row[0])
                features.append(row[1])

            # import IPython; IPython.embed()
            # append the feature array into the buffer
            # not that max buffer size is 32 and buffer_window operates in a sliding window fashion
            if len(buffer_window) < WINDOW_SIZE:
                buffer_window.append(features)
            else:
                # convert input to tensor
                model_input = torch.Tensor(np.array(buffer_window, dtype=np.float32)).to(device)
                # add extra dimension
                model_input = torch.unsqueeze(model_input, dim=0)
                # predict the action class using lstm
                y_pred = lstm_classifier(model_input)
                # import IPython; IPython.embed()
                prob = F.softmax(y_pred, dim=1)
                # get the index of the max probability
                pred_index = prob.data.max(dim=1)[1]
                # pop the first value from buffer_window and add the new entry in FIFO fashion, to have a sliding window of size 32.
                buffer_window.pop(0)
                buffer_window.append(features)
                label = LABELS[pred_index.cpu().numpy()[0]]
                # print("Label detected ", label)

    # add predicted label into the frame
    if label is not None:
        cv2.putText(img, 'Action: {}'.format(label),
                    (int(width-400), height-50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (102, 255, 255), 2)
    # increment counter
    counter += 1
    # write the frame into the result video
    vid_writer.write(img)
    # compute the completion percentage
    percentage = int(counter*100/tot_frames)
    # return the completion percentage

    # show video results
    cv2.imshow("image", img)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

analyze_done = time.time()
print("Video processing finished in ", analyze_done - start)