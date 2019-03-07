import os
from IPython.display import HTML
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
import matplotlib.pyplot as plt
import numpy as np


# coding: utf-8
# Import everything needed to edit/save/watch video clips


def process_image(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
# This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 300), (490, 300), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on

    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 30    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30  #minimum number of pixels making up a line
    max_line_gap = 50    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
# Iterate over the output "lines" and draw lines on a blank image

# Create placeholders for slope(m) and intercept(b) for the left line and right line

    left_m = np.array([])
    left_b = np.array([])
    right_m = np.array([])
    right_b = np.array([])
    left_y_min = image.shape[0]
    right_y_min = image.shape[0]
    for line in lines:
          for x1, y1, x2, y2 in line:
            # Protect against "divide by zero"
            if ( ):
                continue
            # Get the slope and intercept of the current line
            m = (y2 - y1) / (x2 - x1)
            b = y2 - m * x2

            # Left lane
            if (m <= -0.5 and m >= -0.9 and b >= image.shape[0]):
                # Save the slope and the intercept
                left_m = np.append(left_m, m)
                left_b = np.append(left_b, b)
                # Update the top point
                left_y_min = min(left_y_min, min(y1, y2))
            # Right lane
            elif (m >= 0.5 and m <= 0.9 and m * image.shape[1] + b >= image.shape[0]):
                # Save the slope and the intercept
                right_m = np.append(right_m, m)
                right_b = np.append(right_b, b)
                # Update the top point
                right_y_min = min(right_y_min, min(y1, y2))

            # Uncomment the next line to draw the "pre-processd" lines
            # If we found any lines
            if (len(left_m) > 0):
                # Get the average slope and intercept
                left_avg_m = np.mean(left_m)
                left_avg_b = np.mean(left_b)
                # Get the x value for the top point
                left_x_min = (left_y_min - left_avg_b) / left_avg_m
                # Get the x value for the bottom point (draw all the way to bottom)
                left_x_max = (image.shape[0] - left_avg_b) / left_avg_m
                # Draw the left line
                cv2.line(line_image, (int(left_x_min), int(left_y_min)), (int(left_x_max), image.shape[0]), (255, 0, 0),
                         10)
            if (len(right_m) > 0):
                # Get the average slope and intercept
                right_avg_m = np.mean(right_m)
                right_avg_b = np.mean(right_b)
                # Get the x value for the top point
                right_x_min = (right_y_min - right_avg_b) / right_avg_m
                # Get the x value for the bottom point (draw all the way to bottom)
                right_x_max = (image.shape[0] - right_avg_b) / right_avg_m
                cv2.line(line_image, (int(right_x_min), int(right_y_min)), (int(right_x_max), image.shape[0]),
                         (255, 0, 0), 10)




     # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return lines_edges

white_output = 'white-out.mp4'
clip1=VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))





