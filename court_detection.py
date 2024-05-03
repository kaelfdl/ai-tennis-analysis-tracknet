import cv2
import numpy as np
import torch
from mediapipe import solutions
from mini_court_mapping import widthP, heightP, mini_court_map, show_lines, show_point, give_point
from math_utils import calculate_pixels, find_intersection, within_circle, euclidean_distance
from ball_detection import BallDetector
from body_tracking import body_map


video_file = 'input_videos/input_video_a.mp4'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Retrieve video from video file
cap = cv2.VideoCapture(video_file)
width = int(cap.get(3))
height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
clip = cv2.VideoWriter('output/output.mp4', fourcc, 25.0, (widthP, heightP))
processed_frame = None

# Ratios of the crop width, height, and offsets
# If centered is 1, program ignores offset and centers frame

class crop1:
    x: float = 50/100
    x_offset: float = 0/100
    x_center: int = 1

    y: float = 33/100
    y_offset: float = 0/100
    y_center: int = 0

class crop2:
    x: float = 83/100
    x_offset: float = 0/100
    x_center: int = 1

    y: float = 60/100
    y_offset: float = 40/100
    y_center: int = 0


# Calculations for pixels used in both crops
crop1 = calculate_pixels(crop1, width, height)
crop2 = calculate_pixels(crop2, width, height)

# Body smoothing, n is number of frames averaged
n = 3
counter = 0

# Player pose declaration
mp_pose = solutions.pose

class body1:
    pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.25, min_tracking_confidence=0.25)
    x: int
    x_avg: float = 0
    y: int
    y_avg: float = 0

class body2:
    pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.25, min_tracking_confidence=0.25)
    x: int
    x_avg: float = 0
    y: int
    y_avg: float = 0

# Setting reference frame lines
extra_len = width / 3

class axis:
    top = [[-extra_len, 0], [width + extra_len, 0]]
    right = [[width + extra_len, 0], [width + extra_len, height]]
    bottom = [[-extra_len, height], [width + extra_len, height]]
    left = [[-extra_len, 0], [-extra_len, height]]

# Setting comparison points
n_top_left_p = None
n_top_right_p = None
n_bottom_left_p = None
n_bottom_right_p = None

ball_detector = BallDetector(device, 'training/exps/1/model_last.pt', out_channels=256)
ball_proximity = []
ball = None
last_seen = None
hand_points = None
flag = [0, 0, 0, 0]
coords = []
min_dst1 = height * width
min_dst2 = height * width
velocities = []

while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break

    # Apply filters that removes noise and simplifies image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 156, 255, cv2.THRESH_BINARY)[1]
    canny = cv2.Canny(bw, 100, 200)

    # Using hough lines probablistic to find lines with most intersections
    hp_lines = cv2.HoughLinesP(canny, 1, np.pi / 180, threshold=150, minLineLength=100, maxLineGap=10)
    intersect_num = np.zeros((len(hp_lines), 2))
    i = 0

    for hp_line1 in hp_lines:
        line1_x1, line1_y1, line1_x2, line1_y2 = hp_line1[0]
        line1 = [[line1_x1, line1_y1], [line1_x2, line1_y2]]

        for hp_line2 in hp_lines:
            line2_x1, line2_y1, line2_x2, line2_y2 = hp_line2[0]
            line2 = [[line2_x1, line2_y1], [line2_x2, line2_y2]]

            if line1 is line2:
                continue
            if line1_x1 > line1_x2:
                temp = line1_x1
                line1_x1 = line1_x2
                line1_x2 = temp

            if line1_y1 > line1_y2:
                temp = line1_y1
                line1_y1 = line1_y2
                line1_y2 = temp

            intersect = find_intersection(line1, line2, line1_x1 - 200, line1_y1 - 200, line1_x2 + 200, line1_y2 + 200)
            if intersect is not None:
                intersect_num[i][0] += 1
        intersect_num[i][1] = i
        i += 1
    
    # Lines with most intersections get a fill mask command on them
    i = p = 0
    dilation = cv2.dilate(bw, np.ones((5, 5), np.uint8), iterations=1)
    non_rect_area = dilation.copy()
    intersect_num = intersect_num[(-intersect_num)[:, 0].argsort()]

    for hp_line in hp_lines:
        x1, y1, x2, y2 = hp_line[0]
        
        for p in range(8):
            if (i == intersect_num[p][1]) and (intersect_num[i][0] > 0):
                cv2.floodFill(non_rect_area, np.zeros((height + 2, width + 2), np.uint8), (x1, y1), 1)
                cv2.floodFill(non_rect_area, np.zeros((height + 2, width + 2), np.uint8), (x2, y2), 1)
        i += 1
    
    dilation[np.where(non_rect_area == 255)] = 0
    dilation[np.where(non_rect_area == 1)] = 255
    eroded = cv2.erode(dilation, np.ones((5, 5), np.uint8))
    canny_main = cv2.Canny(eroded, 90, 100)

    # Extreme lines found every frame
    x_o_left = width + extra_len
    x_o_right = 0 - extra_len
    x_f_left = width + extra_len
    x_f_right = 0 - extra_len

    y_o_top = height
    y_o_bottom = 0
    y_f_top = height
    y_f_bottom = 0

    # Finding all lines then allocate them to specified extreme variables
    h_lines = cv2.HoughLines(canny_main, 2, np.pi / 180, 300)

    for h_line in h_lines:
        for rho, theta in h_line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + width * (-b))
            y1 = int(y0 + width * (a))
            x2 = int(x0 - width * (-b))
            y2 = int(y0 - width * (a))

            # Furthest intersecting point at every axis calculations done here
            intersect_x_f = find_intersection(axis.bottom, [[x1, y1], [x2, y2]], -extra_len, 0, width + extra_len, height)
            intersect_y_o = find_intersection(axis.left, [[x1, y1], [x2, y2]], -extra_len, 0, width + extra_len, height)
            intersect_x_o = find_intersection(axis.top, [[x1, y1], [x2, y2]], -extra_len, 0, width + extra_len, height)
            intersect_y_f = find_intersection(axis.right, [[x1, y1], [x2, y2]], -extra_len, 0, width + extra_len, height)

            if (intersect_x_o is None) and (intersect_x_f is None) and (intersect_y_o is None) and (intersect_y_f is None):
                continue

            if intersect_x_o is not None:
                if intersect_x_o[0] < x_o_left:
                    x_o_left = intersect_x_o[0]
                    x_o_left_line = [[x1, y1], [x2, y2]]
                if intersect_x_o[0] > x_o_right:
                    x_o_right = intersect_x_o[0]
                    x_o_right_line = [[x1, y1], [x2, y2]]
            if intersect_y_o is not None:
                if intersect_y_o[1] < y_o_top:
                    y_o_top = intersect_y_o[1]
                    y_o_top_line = [[x1, y1], [x2, y2]]
                if intersect_y_o[1] > y_o_bottom:
                    y_o_bottom = intersect_y_o[1]
                    y_o_bottom_line = [[x1, y1], [x2, y2]]


            if intersect_x_f is not None:
                if intersect_x_f[0] < x_f_left:
                    x_f_left = intersect_x_f[0]
                    x_f_left_line = [[x1, y1], [x2, y2]]
                if intersect_x_f[0] > x_f_right:
                    x_f_right = intersect_y_f[0]
                    x_f_right_line = [[x1, y1], [x2, y2]]
            if intersect_y_f is not None:
                if intersect_y_f[1] < y_f_top:
                    y_f_top = intersect_y_f[1]
                    y_f_top_line = [[x1, y1], [x2, y2]]
                if intersect_y_f[1] > y_f_bottom:
                    y_f_bottom = intersect_y_f[1]
                    y_f_bottom_line = [[x1, y1], [x2, y2]]

    # Top line has margin of error that affects all court mapped outputs
    y_o_top[0][1] = y_o_top[0][1] + 4
    y_o_top[1][1] = y_o_top[1][1] + 4

    y_f_top[0][1] = y_f_top[0][1] + 4
    y_f_top[1][1] = y_f_top[1][1] + 4

    # Find four corners of the court and display it
    top_left_p = find_intersection(x_o_left_line, y_o_top_line, -extra_len, 0, width + extra_len, height)
    top_right_p = find_intersection(x_o_right_line, y_f_top_line, -extra_len, 0, width + extra_len, height)
    bottom_left_p = find_intersection(x_f_left_line, y_o_bottom_line, -extra_len, 0, width + extra_len, height)
    bottom_right_p = find_intersection(x_f_right_line, y_f_bottom_line, -extra_len, 0, width + -extra_len, height)

    # If all corner points are different or at least one of them are not found, rerun print
    if (not(top_left_p == n_top_left_p)) and (not(top_right_p == n_top_right_p)) and (not(bottom_left_p == n_bottom_left_p)) and (not(bottom_right_p == n_bottom_right_p)):
        n_top_left_p = top_left_p
        n_top_right_p = top_right_p
        n_bottom_left_p = bottom_left_p
        n_bottom_right_p = bottom_right_p

    # Displaying feet and hand points from body_map function
    hand_points_prev = hand_points
    feet_points, hand_points, nose_points = body_map(frame, body1.pose, body2.pose, crop1, crop2)

    if (not(any(item is None for sublist in feet_points for item in sublist)) or (not(any(item is None for sublist in hand_points for item in sublist)) or (not(any(item is None for sublist in nose_points for item in sublist))))):

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
        circle_radius_body2 = int(0.65 * euclidean_distance(nose_points[1], [body2.x, body2.y]))

        # Distorting frame and outputting results
        processed_frame, M = mini_court_map(frame, n_top_left_p, n_top_right_p, n_bottom_left_p, n_bottom_right_p)

        # Create black background
        cv2.rectangle(processed_frame, (0, 0), (967, 1585), (188, 145, 103), 2000)
        processed_frame = show_lines(processed_frame)

        processed_frame = show_point(processed_frame, M, [body1.x_avg, body1.y_avg])
        processed_frame = show_point(processed_frame, M, [body2.x_avg, body2.y_avg])

        ball_prev = ball
        ball_detector.detect_ball(frame)
        if ball_detector.xy_coordinates[-1][0] is not None:
            ball = ball_detector.xy_coordinates[-1]
            last_seen = counter
        
        # Draw a circle around both hands for both players
        # cv2.circle(frame, (hand_points[0]), circle_radius_body1, (255, 0, 0), 2) # left
        cv2.circle(frame, (hand_points[1]), circle_radius_body1, (255, 0, 0), 2) # right

        # cv2.circle(frame, (hand_points[2]), circle_radius_body2, (255, 0, 0), 2) # left
        cv2.circle(frame, (hand_points[3]), circle_radius_body2, (255, 0, 0), 2) # right

        if ball is not None:
            cv2.circle(frame, ball, 4, (0, 255, 0), 3)
            cv2.circle(frame, ball_prev, 3, (0, 255, 0), 2)

            # If ball location is unique
            if ball is not ball_prev:
                # Find locations where the ball gets closer to the body
                if within_circle(hand_points[1], circle_radius_body1, ball):
                    if min_dst1 > euclidean_distance(hand_points[1], ball):
                        min_dst1 = euclidean_distance(hand_points[1], ball)
                        coords.append((ball, give_point(M, ball), give_point(M, (body1.x, body1.y)), counter))
                else:
                    min_dst1 = circle_radius_body1

                if within_circle(hand_points[3], circle_radius_body2, ball):
                    if min_dst1 > euclidean_distance(hand_points[3], ball):
                        min_dst1 = euclidean_distance(hand_points[3], ball)
                        coords.append((ball, give_point(M, ball), give_point(M, (body2.x, body2.y)), counter))
                else:
                    min_dst1 = circle_radius_body2

                # Find locations of ball bounce

                if ball_detector.xy_coordinates[-2][0] is not None:
                    x_velocity = ball[0] - ball_prev[0]
                    y_velocity = (ball[1] - ball_prev[1]) * (1 + (height - ball[1]) * 0.4 / height)
                    if within_circle(hand_points[3], circle_radius_body2, ball) or within_circle(hand_points[1], circle_radius_body1, ball):
                        within = True
                    else:
                        within = False
                    velocities.append(([x_velocity, y_velocity], counter, give_point(M, ball), within))
                
        # If the previous ball coordinate is close to the current one, remove the previous one
        if len(coords) >= 2:
            if euclidean_distance(coords[-1][0], coords[-2][0]) < 200:
                del coords[-2]
        
        # Display hit points
        for i in range(len(coords)):
            cv2.circle(frame, coords[i][0], 4, (0, 0, 255), 4)
    
    # Write processed frame to clip
    if processed_frame is not None:
        clip.write(processed_frame)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Last found location of ball should be appended to ball array
coords.append((ball, give_point(M, ball), give_point(M, ball), last_seen))

clip.release()
cap.release()
cv2.destroyAllWindows()

accelerations = []
for i in range(2, len(velocities)):
    if velocities[i][1] - 2 == velocities[i - 2][1] and velocities[i - 1][3] is False:
        x_acceleration = (velocities[i][0][0] - velocities[i - 2][0][0]) / 2
        y_acceleration = (velocities[i][0][1] - velocities[i - 2][0][1]) / 2
        accelerations.append((int(y_acceleration), velocities[i][1]))

        if abs(y_acceleration) > (height / 77):
            for k in range(len(coords)):
                if coords[k][3] > velocities[i - 1][1]:
                    coords.insert(k, (velocities[i - 1][2], velocities[i - 1][2], velocities[i - 1][2], velocities[i - 1][1]))
                    break

# Create inbetween points for the ball points found
ball_array = []
while len(coords) > 1:
    time = coords[0][3]
    location = [coords[0][1][0], coords[0][2][1]]                
    del coords[0]
    time_diff = coords[0][3] - time
    for i in range(time, coords[0][3]):
        x = int(location[0] + ((i - time) / time_diff) * (coords[0][1][0] - location[0]))
        y = int(location[1] + ((i - time) / time_diff) * (coords[0][2][1] - location[1]))
        ball_array.append(((x, y), i))

print(coords)
ball_array.append(((coords[0][1][0], coords[0][2][1]), coords[0][3]))

# Overlay ball information on the previous video
clip = cv2.VideoWriter('output/output1.mp4', fourcc, 25.0, (widthP, heightP))
counter = 0
cap = cv2.VideoCapture('output/output.mp4')
write_flag = False

while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break

    counter += 1

    for i in range(len(ball_array)):
        if counter == ball_array[i][1]:
            cv2.circle(frame, (ball_array[i][0]), 4, (0, 255, 255), 3)
            break

    if counter == ball_array[0][1]:
        write_flag = True

    if ball_array[-1][1] == counter:
        write_flag = False
    
    if (write_flag):
        index = counter - ball_array[0][1]
        cv2.circle(frame, (ball_array[index][0]), 2, (0, 255, 255), 3)

    clip.write()

cap.release()
clip.release()
cv2.destroyAllWindows()
            

