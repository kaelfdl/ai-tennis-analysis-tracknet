from mediapipe import solutions
import cv2

mp_pose = solutions.pose
mp_drawing = solutions.drawing_utils
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
    frame2 = frame[crop2.y_offset:crop2.y+crop2.y_offset,crop2.x_offset:crop2.x+crop2.x_offset]
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