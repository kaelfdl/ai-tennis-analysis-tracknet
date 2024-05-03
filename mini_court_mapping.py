import numpy as np
import cv2
from trace_header import video_file, check_path

cap = cv2.VideoCapture(video_file)
check_path(video_file)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

widthP = int(967)
heightP = int(1585)

width = int(967)
height = int(1585)

court_width_meters = 10.97
court_height_meters = 23.77
court_baseline_serviceline_height_meters = 5.48
doubles_alley_width_meters = 1.37
no_mans_height_meters = 5.48

ratio = (court_width_meters/court_height_meters)

mini_court_height = int(height * 0.6)
mini_court_width = int(mini_court_height * ratio)
y_offset = int((height - mini_court_height) / 2)
x_offset = int((width - mini_court_width) / 2)

mini_court_tl = [x_offset, y_offset]
mini_court_tr = [mini_court_width + x_offset, y_offset]
mini_court_bl = [x_offset, mini_court_height + y_offset]
mini_court_br = [mini_court_width + x_offset, mini_court_height + y_offset]

mini_half_court_width = int(mini_court_width * 0.5)
mini_half_court_height = int(mini_court_height * 0.5)
mini_doubles_alley_width = int(mini_court_width * (doubles_alley_width_meters / court_width_meters))
mini_no_mans_height = int(mini_court_height * (no_mans_height_meters / court_height_meters))

def mini_court_map(frame, tl , tr, bl, br):
    pts1 = np.float32([[tl, tr, bl, br]])
    pts2 = np.float32([mini_court_tl, mini_court_tr, mini_court_bl, mini_court_br])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(frame, M, (width, height))
    return dst, M

def show_lines(frame):
    cv2.rectangle(frame, (0, 0), (width, height), (255, 255, 255), 6)
    # Whole court
    cv2.rectangle(frame, (x_offset, y_offset), (mini_court_width + x_offset, mini_court_height + y_offset), (255, 255, 255), 2)
    # Net
    cv2.rectangle(frame, (x_offset, y_offset + mini_half_court_height), (mini_court_width + x_offset, mini_court_height + y_offset + mini_half_court_height), (255, 255, 255), 2)
    # Singles court
    cv2.rectangle(frame, (x_offset + mini_doubles_alley_width, y_offset), (mini_court_width + x_offset - mini_doubles_alley_width, mini_court_height + y_offset), (255, 255, 255), 2)
    # No mans land
    cv2.rectangle(frame, (x_offset + mini_doubles_alley_width, y_offset + mini_no_mans_height), (mini_court_width + x_offset - mini_doubles_alley_width, mini_court_height + y_offset - mini_no_mans_height), (255, 255, 255), 2)
    # Service boxes
    cv2.rectangle(frame, (x_offset + mini_half_court_width, y_offset + mini_no_mans_height), (mini_court_width + x_offset - mini_half_court_width, mini_court_height + y_offset - mini_no_mans_height), (255, 255, 255), 2)
    return frame

def show_point(frame, M, point):
    points = np.float32([[point]])
    transformed = cv2.perspectiveTransform(points, M)[0][0]
    cv2.circle(frame, (int(transformed[0]), int(transformed[0])), radius=0, color=(0, 0, 255), thickness=25)
    return frame

def give_point(M, point):
    points = np.float32([[point]])
    transformed = cv2.perspectiveTransform(points, M)[0][0]
    return (int(transformed[0]), int(transformed[1]))

