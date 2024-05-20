from mediapipe import solutions
import cv2
from math_utils import euclidean_distance



# Player pose declaration
mp_pose = solutions.pose
mp_drawing = solutions.drawing_utils

class Body1:
    pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.25, min_tracking_confidence=0.25)
    x: int
    x_avg: float = 0
    y: int
    y_avg: float = 0

class Body2:
    pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.25, min_tracking_confidence=0.25)
    x: int
    x_avg: float = 0
    y: int
    y_avg: float = 0

def body_map(frame, pose1, pose2, crop1, crop2):

    # Mapping of player 1
    frame1 = frame[crop1.y_offset:crop1.y + crop1.y_offset, crop1.x_offset:crop1.x + crop1.x_offset]
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    results1 = pose1.process(frame1)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, solutions.pose.POSE_CONNECTIONS)
    if results1.pose_landmarks is not None:
        l1_foot_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * crop1.x) + crop1.x_offset
        l1_foot_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * crop1.y) + crop1.y_offset

        r1_foot_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * crop1.x) + crop1.x_offset
        r1_foot_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * crop1.y) + crop1.y_offset

        l1_hand_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x * crop1.x) + crop1.x_offset
        l1_hand_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y * crop1.y) + crop1.y_offset

        r1_hand_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x * crop1.x) + crop1.x_offset
        r1_hand_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y * crop1.y) + crop1.y_offset

        nose1_x = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * crop1.x) + crop1.x_offset
        nose1_y = int(results1.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * crop1.y) + crop1.y_offset
    else:
        l1_foot_x = None
        l1_foot_y = None

        
        r1_foot_x = None
        r1_foot_y = None

        l1_hand_x = None
        l1_hand_y = None

        r1_hand_x = None
        r1_hand_y = None

        nose1_x = None
        nose1_y = None

    # Mapping of player 2
    frame2 = frame[crop2.y_offset : crop2.y + crop2.y_offset, crop2.x_offset : crop2.x + crop2.x_offset]
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    results2 = pose2.process(frame2)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(frame2, results2.pose_landmarks, solutions.pose.POSE_CONNECTIONS)

    if results2.pose_landmarks is not None:
        l2_foot_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * crop2.x) + crop2.x_offset
        l2_foot_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * crop2.y) + crop2.y_offset

        r2_foot_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * crop2.x) + crop2.x_offset
        r2_foot_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * crop2.y) + crop2.y_offset

        l2_hand_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x * crop2.x) + crop2.x_offset
        l2_hand_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y * crop2.y) + crop2.y_offset

        r2_hand_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x * crop2.x) + crop2.x_offset
        r2_hand_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y * crop2.y) + crop2.y_offset

        nose2_x = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * crop2.x) + crop2.x_offset
        nose2_y = int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * crop2.y) + crop2.y_offset
    else:
        l2_foot_x = None
        l2_foot_y = None

        r2_foot_x = None
        r2_foot_y = None

        l2_hand_x = None
        l2_hand_y = None

        r2_hand_x = None
        r2_hand_y = None

        nose2_x = None
        nose2_y = None

    return ([[[l1_foot_x,l1_foot_y],[r1_foot_x,r1_foot_y],[l2_foot_x,l2_foot_y],[r2_foot_x,r2_foot_y]], [[l1_hand_x,l1_hand_y],[r1_hand_x,r1_hand_y],[l2_hand_x,l2_hand_y],[r2_hand_x,r2_hand_y]], [[nose1_x, nose1_y], [nose2_x, nose2_y]]])

def body_tracking(frame, feet_points, hand_points, nose_points, body1, body2, n, counter):

    # cv2.circle(frame, hand_points[0], radius=0, color=(0, 0, 255), thickness=10)
    # cv2.circle(frame, hand_points[1], radius=0, color=(0, 0, 255), thickness=10)
    # cv2.circle(frame, hand_points[2], radius=0, color=(0, 0, 255), thickness=30)
    # cv2.circle(frame, hand_points[3], radius=0, color=(0, 0, 255), thickness=30)

    # cv2.circle(frame, feet_points[0], radius=0, color=(0, 0, 255), thickness=10)
    # cv2.circle(frame, feet_points[1], radius=0, color=(0, 0, 255), thickness=10)
    # cv2.circle(frame, feet_points[2], radius=0, color=(0, 0, 255), thickness=30)
    # cv2.circle(frame, feet_points[3], radius=0, color=(0, 0, 255), thickness=30)

    # Prioritizing lower foot y in body average y position
    if feet_points[0][1] > feet_points[1][1]:
        lower_foot1 = feet_points[0][1]
        higher_foot1 = feet_points[1][1]
    else:
        lower_foot1 = feet_points[1][1]
        higher_foot1 = feet_points[0][1]

    if feet_points[2][1] > feet_points[3][1]:
        lower_foot2 = feet_points[2][1]
        higher_foot2 = feet_points[3][1]
    else:
        lower_foot2 = feet_points[3][1]
        higher_foot2 = feet_points[2][1]

    # Allocated 75 % preference to lower foot y positions
    body1.x = (feet_points[0][0] + feet_points[1][0]) / 2
    body1.y = lower_foot1 * 0.8 + higher_foot1 * 0.2

    body2.x = (feet_points[2][0] + feet_points[3][0]) / 2
    body2.y = lower_foot2 * 0.8 + higher_foot2 * 0.2

    # Body coordinate smoothing
    counter += 1
    coeff = 1. / min(counter, n)
    body1.x_avg = coeff * body1.x + (1. - coeff) * body1.x_avg
    body1.y_avg = coeff * body1.y + (1. - coeff) * body1.y_avg
    body2.x_avg = coeff * body2.x + (1. - coeff) * body2.x_avg
    body2.y_avg = coeff * body2.y + (1. - coeff) * body2.y_avg

    # Calculate euclidian distance between average of feet and hand indexes for both players
    circle_radius_body1 = int(0.65 * euclidean_distance(nose_points[0], [body1.x, body1.y]))
    circle_radius_body2 = int(0.6 * euclidean_distance(nose_points[1], [body2.x, body2.y]))

    # Draw a circle around both hands for both players
    # cv2.circle(frame, (hand_points[0]), circle_radius_body1, (255, 0, 0), 2) # left
    cv2.circle(frame, (hand_points[1]), circle_radius_body1, (255, 0, 0), 2) # right

    # cv2.circle(frame, (hand_points[2]), circle_radius_body2, (255, 0, 0), 2) # left
    cv2.circle(frame, (hand_points[3]), circle_radius_body2, (255, 0, 0), 2) # right

    return frame, circle_radius_body1, circle_radius_body2, counter