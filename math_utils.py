import os
import sys
import math

def check_path(filepath):
    flag = 0
    if not os.path.exists(filepath):
        print(f'video_file: path {filepath} does not exist')
        flag = 1
    if (flag):
        exit()


def calculate_pixels(frame, width, height):
    frame.x = int(width * frame.x)
    frame.y = int(height * frame.y)

    if frame.x_center:
        frame.x_offset = int((width - frame.x) / 2)
    else:
        frame.x_offset = int((width * frame.x_offset))

    if frame.y_center:
        frame.y_offset = int((height - frame.y) / 2)
    else:
        frame.y_offset = int((height * frame.y_offset))
    return frame

def determinant(a, b):
    return a[0] * b[1] - a[1] * b[0]

def find_intersection(line1, line2, x_start, y_start, x_end, y_end):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    div = determinant(x_diff, y_diff)

    if div == 0:
        return None
    
    d = (determinant(*line1), determinant(*line2))
    x = int(determinant(d, x_diff) / div)
    y = int(determinant(d, y_diff) / div)

    if (x < x_start) or (x > x_end):
        return None
    if (y < y_start) or (y > y_end):
        return None
    
    return x, y

def euclidean_distance(point1, point2):
    return math.dist(point1, point2)

def within_circle(center, radius, point):
    return radius > euclidean_distance(center, point)

def closest_point(prev_center, curr_center, prev_point, curr_point):
    if euclidean_distance(prev_center, prev_point) <= euclidean_distance(curr_center, curr_point):
        return True