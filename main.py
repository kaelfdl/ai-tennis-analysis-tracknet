import cv2
import numpy as np
import torch
from mediapipe import solutions
from mini_court_mapping import video_file, widthP, heightP, mini_court_map, show_lines, show_point, give_point
from math_utils import calculate_pixels, find_intersection, within_circle, euclidean_distance
from ball_detection import BallDetector
from body_tracking import body_map, body_tracking, Body1, Body2
from court_mapping import court_map
from parameters import Crop1, Crop2
from video_utils import read_video, save_video

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read video from video file
# cap = cv2.VideoCapture(video_file)
# width = int(cap.get(3))
# height = int(cap.get(4))

frames = read_video(video_file)
height, width = frames[0].shape[0], frames[0].shape[1]
print(height, width)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# clip = cv2.VideoWriter('output/output.mp4', fourcc, 25.0, (width, height))
# clip_a = cv2.VideoWriter('output/output_a.mp4', fourcc, 25.0, (widthP, heightP))
processed_frame = None


# Calculations for pixels used in both crops
crop1 = calculate_pixels(Crop1, width, height)
crop2 = calculate_pixels(Crop2, width, height)

# Body smoothing, n is number of frames averaged
n = 3
counter = 0


# Setting reference frame lines
extra_len = width / 3

class Axis:
    top = [[-extra_len, 0], [width + extra_len, 0]]
    right = [[width + extra_len, 0], [width + extra_len, height]]
    bottom = [[-extra_len, height], [width + extra_len, height]]
    left = [[-extra_len, 0], [-extra_len, height]]

# Setting comparison points
n_top_left_p = None
n_top_right_p = None
n_bottom_left_p = None
n_bottom_right_p = None

ball_detector = BallDetector(device, 'training/exps/1/Weights.pth', out_channels=2)
ball_proximity = []
ball = None
last_seen = None
hand_points = None
flag = [0, 0, 0, 0]
coords = []
min_dst1 = height * width
min_dst2 = height * width
velocities = []

# while cap.isOpened():
#     ret, frame = cap.read()
#     if frame is None:
#         break

output_frames = []
output_processed_frames = []

for frame in frames:
    # Apply filters that removes noise and simplifies image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 156, 255, cv2.THRESH_BINARY)[1]
    canny = cv2.Canny(bw, 100, 200)

    top_left_p, top_right_p, bottom_left_p, bottom_right_p = court_map(frame, canny, bw, width, height, extra_len, Axis, n_top_left_p, n_top_right_p, n_bottom_left_p, n_bottom_right_p)

    n_top_left_p = top_left_p
    n_top_right_p = top_right_p
    n_bottom_left_p = bottom_left_p
    n_bottom_right_p = bottom_right_p


    # Displaying feet and hand points from body_map function
    hand_points_prev = hand_points
    feet_points, hand_points, nose_points = body_map(frame, Body1.pose, Body2.pose, crop1, crop2)
    if (not any(item is None for sublist in feet_points for item in sublist)) or (not any(item is None for sublist in hand_points for item in sublist)) or (not any(item is None for sublist in nose_points for item in sublist)):
        ball_prev = ball
        ball_detector.detect_ball(frame)
        if ball_detector.xy_coordinates[-1][0] is not None:
            ball = ball_detector.xy_coordinates[-1]
            last_seen = counter
        
        frame, circle_radius_body1, circle_radius_body2, counter = body_tracking(frame, feet_points, hand_points, nose_points, Body1, Body2, n, counter)

        print(circle_radius_body1)
        # Distorting frame and outputting results
        processed_frame, M = mini_court_map(frame, n_top_left_p, n_top_right_p, n_bottom_left_p, n_bottom_right_p)

        # Create black background
        cv2.rectangle(processed_frame, (0, 0), (967, 1585), (188, 145, 103), 2000)
        processed_frame = show_lines(processed_frame)

        processed_frame = show_point(processed_frame, M, [Body1.x_avg, Body1.y_avg])
        processed_frame = show_point(processed_frame, M, [Body2.x_avg, Body2.y_avg])


        if ball is not None:
            cv2.circle(frame, ball, 4, (0, 255, 0), 3)
            cv2.circle(frame, ball_prev, 3, (0, 255, 0), 2)

            # If ball location is unique
            if ball is not ball_prev:
                # Find locations where the ball gets closer to the body
                if within_circle(hand_points[1], circle_radius_body1, ball):
                    if min_dst1 > euclidean_distance(hand_points[1], ball):
                        print(circle_radius_body1)
                        min_dst1 = euclidean_distance(hand_points[1], ball)
                        coords.append((ball, give_point(M, ball), give_point(M, (Body1.x, Body1.y)), counter))
                else:
                    min_dst1 = circle_radius_body1

                if within_circle(hand_points[3], circle_radius_body2, ball):
                    if min_dst2 > euclidean_distance(hand_points[3], ball):
                        min_dst2 = euclidean_distance(hand_points[3], ball)
                        coords.append((ball, give_point(M, ball), give_point(M, (Body2.x, Body2.y)), counter))
                else:
                    min_dst2 = circle_radius_body2

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
    
    # clip.write(frame)
    # Write processed frame to clip
    if processed_frame is not None:
        # clip_a.write(processed_frame)
        output_processed_frames.append(processed_frame)
    # cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    output_frames.append(frame)

# Last found location of ball should be appended to ball array
coords.append((ball, give_point(M, ball), give_point(M, ball), last_seen))

save_video(output_frames, 'output/output.mp4')
save_video(output_processed_frames, 'output/output_a.mp4')

# clip.release()
# clip_a.release()
# cap.release()
# cv2.destroyAllWindows()

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

ball_array.append(((coords[0][1][0], coords[0][2][1]), coords[0][3]))

# Overlay ball information on the previous video
# clip_b = cv2.VideoWriter('output/output_b.mp4', fourcc, 25.0, (widthP, heightP))
counter = 0
# cap = cv2.VideoCapture('output/output_a.mp4')
write_flag = False

frames = read_video('output/output_a.mp4')

print(len(coords))
print(coords)
print(len(ball_array))
# while cap.isOpened():
#     ret, frame = cap.read()
#     if frame is None:
#         break

output_frames = []
for frame in frames:
    counter += 1
    for i in range(len(ball_array)):
        if counter == ball_array[i][1]:
            cv2.circle(frame, (ball_array[i][0]), 4, (0, 255, 255), 3)
            break
    
    # for i in range(len(accelerations)):
    #     if counter == accelerations[i][1]:
    #         print(accelerations[i])
    #         break

    if counter == ball_array[0][1]:
        write_flag = True

    if ball_array[-1][1] == counter:
        write_flag = False
    
    if (write_flag):
        index = counter - ball_array[0][1]
        cv2.circle(frame, (ball_array[index][0]), 2, (0, 255, 255), 3)

    # clip.write(frame)
    output_frames.append(frame)

save_video(output_frames, 'output/output_b.mp4')
# cap.release()
# clip.release()
# cv2.destroyAllWindows()
            

