import cv2 as cv
import sys

import matplotlib.pyplot as plt
import pandas as pd
import os
import xlsxwriter

(major_ver, minor_ver, subminor_ver) = cv.__version__.split('.')

PATH = "C:/Users/mehme/OneDrive/Masaüstü/KCF-CSRT/videos/low/light/14 cm window/" \
       "trial07_DL_sos_black_fish04_IL_WN_LM_CL_1600_Tue_Aug_30_15 53 47_2022.avi"

templates = []

frame_numbers = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1499]
# Read video
video = cv.VideoCapture(PATH)
# video = cv2.VideoCapture(0) # for using CAM

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

i = 0
while True:
    if len(frame_numbers) > i:
        print(i)
        video.set(cv.CAP_PROP_POS_FRAMES, frame_numbers[i])
        i += 1
    else:
        break

    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    bbox = cv.selectROI(frame, True, True)
    # Getting Fish Template Operations
    yi = bbox[1]  # initial y position
    yf = yi + bbox[3]  # final y position
    xi = bbox[0]  # initial x position
    xf = xi + bbox[2]  # final x position
    fish_template = frame[yi:yf, xi:xf]
    templates.append(fish_template)



