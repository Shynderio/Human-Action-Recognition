import cv2
import math
from typing import List, Mapping, Optional, Tuple, Union
WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 255, 0)


def draw_line(image, p1, p2, color, thickness=2):
    # Ensure the points are tuples of integers
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    
    # Draw the line using OpenCV
    cv2.line(image, p1, p2, color, thickness=thickness)


def find_person_indicies(scores):
    return [i for i, s in enumerate(scores) if s > 0.9]


def filter_persons(outputs):
    persons = {}
    p_indicies = find_person_indicies(outputs["instances"].scores)
    for x in p_indicies:
        desired_kp = outputs["instances"].pred_keypoints[x][:].to("cpu")
        persons[x] = desired_kp
    return (persons, p_indicies)


def draw_keypoints(person, img):
    # Convert all points to (x, y) integer tuples, ignoring the confidence score
    l_eye = tuple(map(int, person[1][:2]))
    r_eye = tuple(map(int, person[2][:2]))
    l_ear = tuple(map(int, person[3][:2]))
    r_ear = tuple(map(int, person[4][:2]))
    nose = tuple(map(int, person[0][:2]))
    l_shoulder = tuple(map(int, person[5][:2]))
    r_shoulder = tuple(map(int, person[6][:2]))
    l_elbow = tuple(map(int, person[7][:2]))
    r_elbow = tuple(map(int, person[8][:2]))
    l_wrist = tuple(map(int, person[9][:2]))
    r_wrist = tuple(map(int, person[10][:2]))
    l_hip = tuple(map(int, person[11][:2]))
    r_hip = tuple(map(int, person[12][:2]))
    l_knee = tuple(map(int, person[13][:2]))
    r_knee = tuple(map(int, person[14][:2]))
    l_ankle = tuple(map(int, person[15][:2]))
    r_ankle = tuple(map(int, person[16][:2]))

    # Draw lines
    draw_line(img, l_shoulder, l_elbow, GREEN_COLOR)
    draw_line(img, l_elbow, l_wrist, GREEN_COLOR)
    draw_line(img, l_shoulder, r_shoulder, GREEN_COLOR)
    draw_line(img, l_shoulder, l_hip, GREEN_COLOR)
    draw_line(img, r_shoulder, r_hip, GREEN_COLOR)
    draw_line(img, r_shoulder, r_elbow, GREEN_COLOR)
    draw_line(img, r_elbow, r_wrist, GREEN_COLOR)
    draw_line(img, l_hip, r_hip, GREEN_COLOR)
    draw_line(img, l_hip, l_knee, GREEN_COLOR)
    draw_line(img, l_knee, l_ankle, GREEN_COLOR)
    draw_line(img, r_hip, r_knee, GREEN_COLOR)
    draw_line(img, r_knee, r_ankle, GREEN_COLOR)

    # Draw circles
    cv2.circle(img, l_eye, 4, WHITE_COLOR, -1)
    cv2.circle(img, r_eye, 4, WHITE_COLOR, -1)
    cv2.circle(img, l_wrist, 4, WHITE_COLOR, -1)
    cv2.circle(img, r_wrist, 4, WHITE_COLOR, -1)
    cv2.circle(img, l_shoulder, 4, WHITE_COLOR, -1)
    cv2.circle(img, r_shoulder, 4, WHITE_COLOR, -1)
    cv2.circle(img, l_elbow, 4, WHITE_COLOR, -1)
    cv2.circle(img, r_elbow, 4, WHITE_COLOR, -1)
    cv2.circle(img, l_hip, 4, WHITE_COLOR, -1)
    cv2.circle(img, r_hip, 4, WHITE_COLOR, -1)
    cv2.circle(img, l_knee, 4, WHITE_COLOR, -1)
    cv2.circle(img, r_knee, 4, WHITE_COLOR, -1)
    cv2.circle(img, l_ankle, 4, WHITE_COLOR, -1)
    cv2.circle(img, r_ankle, 4, WHITE_COLOR, -1)


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px