import math

def euclidean_distance(point1, point2):
    return math.dist(point1, point2)

def within_circle(center, radius, point):
    return radius > euclidean_distance(center, point)

def closest_point(prev_center, curr_center, prev_point, curr_point):
    if euclidean_distance(prev_center, prev_point) <= euclidean_distance(curr_center, curr_point):
        return True