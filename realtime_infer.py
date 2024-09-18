import cv2
import torch
import numpy as np
import time
import torch.nn.functional as F
from src.lstm import ActionClassificationLSTM, WINDOW_SIZE
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from src.utils import filter_persons, draw_keypoints

LABELS = {
    0: "JUMPING",
    1: "JUMPING_JACKS",
    2: "BOXING",
    3: "WAVING_2HANDS",
    4: "WAVING_1HAND",
    5: "CLAPPING_HANDS"
}

# Setup Detectron2 model for keypoint detection
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
pose_detector = DefaultPredictor(cfg)

# Load LSTM action classifier
lstm_classifier = ActionClassificationLSTM.load_from_checkpoint("models/epoch=394-step=17774.ckpt")
lstm_classifier.eval()

# Buffer for LSTM input
buffer_window = []

SKIP_FRAME_COUNT = 1

def analyse_live_feed(pose_detector, lstm_classifier):
    cap = cv2.VideoCapture(1)  # Capture from default webcam (change 0 to another index for other cameras)
    counter = 0
    label = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = frame.copy()
        height, width = img.shape[:2]

        if counter % (SKIP_FRAME_COUNT + 1) == 0:
            outputs = pose_detector(frame)
            persons, pIndices = filter_persons(outputs)

            if len(persons) >= 1:
                p = persons[0]
                draw_keypoints(p, img)

                features = []
                for i, row in enumerate(p):
                    features.append(row[0])
                    features.append(row[1])

                if len(buffer_window) < WINDOW_SIZE:
                    buffer_window.append(features)
                else:
                    model_input = torch.Tensor(np.array(buffer_window, dtype=np.float32))
                    model_input = torch.unsqueeze(model_input, dim=0)
                    y_pred = lstm_classifier(model_input)
                    prob = F.softmax(y_pred, dim=1)
                    pred_index = prob.data.max(dim=1)[1]
                    buffer_window.pop(0)
                    buffer_window.append(features)
                    label = LABELS[pred_index.numpy()[0]]

        if label is not None:
            cv2.putText(img, f'Action: {label}', (int(width - 400), height - 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.9, (102, 255, 255), 2)

        cv2.imshow("Real-time Action Recognition", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    analyse_live_feed(pose_detector, lstm_classifier)