import cv2
import numpy as np

from math_utils import find_intersection

def court_map(frame, canny, bw, width, height, extra_len, axis, n_top_left_p, n_top_right_p, n_bottom_left_p, n_bottom_right_p):
    # Using hough lines probablistic to find lines with most intersections
    hp_lines = cv2.HoughLinesP(canny, 1, np.pi / 180, threshold=150, minLineLength=100, maxLineGap=10)
    
    if hp_lines is not None:

        # if hp_lines is not None:
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
            # cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            for p in range(2):
                if (i == intersect_num[p][1]) and (intersect_num[i][0] > 0):
                    # cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
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
        h_lines = cv2.HoughLines(canny_main, 2, np.pi / 180, 210)

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
                        x_f_right = intersect_x_f[0]
                        x_f_right_line = [[x1, y1], [x2, y2]]
                if intersect_y_f is not None:
                    if intersect_y_f[1] < y_f_top:
                        y_f_top = intersect_y_f[1]
                        y_f_top_line = [[x1, y1], [x2, y2]]
                    if intersect_y_f[1] > y_f_bottom:
                        y_f_bottom = intersect_y_f[1]
                        y_f_bottom_line = [[x1, y1], [x2, y2]]
                # cv2.line(frame, (x1, y1), (x2, y2), (0, 0 , 255), 2)
        
        # line_endpoints = []
        # line_endpoints.append(x_o_left_line)
        # line_endpoints.append(x_o_right_line)
        # line_endpoints.append(y_o_top_line)
        # line_endpoints.append(y_o_bottom_line)
        # line_endpoints.append(x_f_left_line)
        # line_endpoints.append(x_f_right_line)
        # line_endpoints.append(y_f_top_line)
        # line_endpoints.append(y_f_bottom_line)

        # for i in range(len(line_endpoints)):
        #     cv2.line(frame, (line_endpoints[i][0][0], line_endpoints[i][0][1]), (line_endpoints[i][1][0], line_endpoints[i][1][1]), (0, 0, 255), 2)

        # Top line has margin of error that affects all court mapped outputs
        y_o_top_line[0][1] = y_o_top_line[0][1] + 4
        y_o_top_line[1][1] = y_o_top_line[1][1] + 4

        y_f_top_line[0][1] = y_f_top_line[0][1] + 4
        y_f_top_line[1][1] = y_f_top_line[1][1] + 4

        # Find four corners of the court and display it
        top_left_p = find_intersection(x_o_left_line, y_o_top_line, -extra_len, 0, width + extra_len, height)
        top_right_p = find_intersection(x_o_right_line, y_f_top_line, -extra_len, 0, width + extra_len, height)
        bottom_left_p = find_intersection(x_f_left_line, y_o_bottom_line, -extra_len, 0, width + extra_len, height)
        bottom_right_p = find_intersection(x_f_right_line, y_f_bottom_line, -extra_len, 0, width + extra_len, height)

        # If all corner points are different or at least one of them are not found, rerun print
        if (not(top_left_p == n_top_left_p)) and (not(top_right_p == n_top_right_p)) and (not(bottom_left_p == n_bottom_left_p)) and (not(bottom_right_p == n_bottom_right_p)):
            # cv2.line(frame, top_left_p, top_right_p, (0, 0, 255), 2)
            # cv2.line(frame, bottom_left_p, bottom_right_p, (0, 0, 255), 2)
            # cv2.line(frame, top_left_p, bottom_left_p, (0, 0, 255), 2)
            # cv2.line(frame, top_right_p, bottom_right_p, (0, 0, 255), 2)

            # cv2.circle(frame, top_left_p, radius=0, color=(255, 0, 255), thickness=10)
            # cv2.circle(frame, top_right_p, radius=0, color=(255, 0, 255), thickness=10)
            # cv2.circle(frame, bottom_left_p, radius=0, color=(255, 0, 255), thickness=10)
            # cv2.circle(frame, bottom_right_p, radius=0, color=(255, 0, 255), thickness=10)

            n_top_left_p = top_left_p
            n_top_right_p = top_right_p
            n_bottom_left_p = bottom_left_p
            n_bottom_right_p = bottom_right_p
        
        # else:
            # cv2.line(frame, n_top_left_p, n_top_right_p, (0, 0, 255), 2)
            # cv2.line(frame, n_bottom_left_p, n_bottom_right_p, (0, 0, 255), 2)
            # cv2.line(frame, n_top_left_p, n_bottom_left_p, (0, 0, 255), 2)
            # cv2.line(frame, n_top_right_p, n_bottom_right_p, (0, 0, 255), 2)


            # cv2.circle(frame, n_top_left_p, radius=0, color=(255, 0, 255), thickness=10)
            # cv2.circle(frame, n_top_right_p, radius=0, color=(255, 0, 255), thickness=10)
            # cv2.circle(frame, n_bottom_left_p, radius=0, color=(255, 0, 255), thickness=10)
            # cv2.circle(frame, n_bottom_right_p, radius=0, color=(255, 0, 255), thickness=10)

    return n_top_left_p, n_top_right_p, n_bottom_left_p, n_bottom_right_p