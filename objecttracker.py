import cv2 as cv
import sys
import os
import pandas as pd
import xlsxwriter
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

(major_ver, minor_ver, subminor_ver) = cv.__version__.split('.')


class ObjectTracker:
    def __init__(self, video_path, excel_path, video_number, debug_path, debug, rec, threshold):
        self.DEBUG = debug
        self.tracker_types = ['KCF', 'CSRT']
        self.tracker_type = self.tracker_types[0]
        self.fish_tracker = None
        self.cage_tracker = None
        self.fish_position = []
        self.cage_position = []
        self.bbox = None
        self.video_path = video_path
        self.excel_path = excel_path
        self.debug_path = debug_path
        self.video = None
        self.current_frame = None
        self.create_excel_file = False
        self.video_number = video_number
        self.fish_template = None
        self.max_correlation = None
        self.correlation_threshold = threshold
        self.record = rec
        self.interrupt_key = None  # it waits every 1ms an intetrupt from keyboard
        self.pressed_key = None   # it waits every 1ms an comand from keyboard (next,back,roi,quit)

    def create_tracker(self):
        """
        Creates opencv tracker object

        :return: empty
        """
        if int(minor_ver) < 3:
            self.fish_tracker = cv.Tracker_create(self.tracker_type)
            self.fish_tracker = cv.Tracker_create(self.tracker_type)
        else:
            if self.tracker_type == 'MIL':
                self.fish_tracker = cv.TrackerMIL_create()
                self.cage_tracker = cv.TrackerMIL_create()
            elif self.tracker_type == 'KCF':
                self.fish_tracker = cv.TrackerKCF_create()
                self.cage_tracker = cv.TrackerKCF_create()
            elif self.tracker_type == "CSRT":
                self.fish_tracker = cv.TrackerCSRT_create()
                self.cage_tracker = cv.TrackerCSRT_create()

    def initialize_tracker(self):
        """
        Reads one frame from Video Object,
        Requests bounding box for template from user,
        Initialize tracker object with selected template.

        :return: empty
        """
        self.video = cv.VideoCapture(self.video_path)

        print(f"Total Frame {int(self.video.get(cv.CAP_PROP_FRAME_COUNT))}")
        # it allocates list as total number of video frame
        self.fish_position = [-1000 for _ in range(int(self.video.get(cv.CAP_PROP_FRAME_COUNT)))]
        self.cage_position = [-1000 for _ in range(int(self.video.get(cv.CAP_PROP_FRAME_COUNT)))]

        # Exit if video not opened.
        if not self.video.isOpened():
            print("Could not open video")
            sys.exit()

        # Read first frame.
        ok, self.current_frame = self.video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()

        cv.putText(self.current_frame, 'To select box press "SPACE" (First Fish !)', (30, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2)
        cv.putText(self.current_frame, 'To continue press "ESC"', (30, 55), cv.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2)

        cv.namedWindow(f"{self.video_number}. Video's Templates")
        cv.moveWindow(f"{self.video_number}. Video's Templates", 100, 100)

        # Define an initial bounding box
        # [Top_Left_X, Top_Left_Y, Width, Height]
        self.bbox = cv.selectROIs(f"{self.video_number}. Video's Templates",
                                  img=self.current_frame, showCrosshair=True, fromCenter=True)

        cv.destroyWindow(f"{self.video_number}. Video's Templates")

        if len(self.bbox) == 2:
            self.fish_tracker.init(self.current_frame, self.bbox[0])
            self.cage_tracker.init(self.current_frame, self.bbox[1])
        else:
            print("Error : Be sure that you select two box, fish and cage object separately !")
            sys.exit()

        self.get_template(self.current_frame)

    def update_tracker(self):
        """
        Main Function
        Updates Cage and Fish Tracker each cycle with new frame.
        Calls other functions each cycle. (get_template, calculate_correlation_coefficient, find_position)

        :return: empty
        """

        # Start from first frame
        _ = self.video.set(cv.CAP_PROP_POS_FRAMES, 0)
        one_shot = True
        while True:
            draw_circle_on_template = True  # when new template created flag prevent any drawing on it
            # Start timer
            timer = cv.getTickCount()

            # Read a new frame
            ok, self.current_frame = self.video.read()
            if not ok:
                break

            # Update Cage and Fish Tracker
            ok1, self.bbox[0] = self.fish_tracker.update(self.current_frame)
            ok2, self.bbox[1] = self.cage_tracker.update(self.current_frame)

            self.calculate_correlation_coefficient()

            # Re-creating fish tracker each time correlation coefficient is smaller than threshold.
            if self.max_correlation < self.correlation_threshold:
                ok1 = False

            # Re-creating fish tracker each time unsuccessful when detection or low correlation coefficient occur.
            if not ok1:
                cv.putText(self.current_frame, 'Tracking failure detected, re-create        template !', (20, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.75, (68, 105, 255), 2)
                cv.putText(self.current_frame, '"FISH"', (460, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
                cv.putText(self.current_frame, 'To select box press "SPACE"', (20, 45),
                           cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                cv.putText(self.current_frame, f'Frame: {str(int(self.video.get(cv.CAP_PROP_POS_FRAMES)))}', (20, 80),
                           cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                # Display correlation coefficient number
                cv.putText(self.current_frame, "Fish Correlation coefficient:  " + str(round(self.max_correlation, 4)),
                           (300, 45), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                # Create new fish tracker
                if self.tracker_type == "KCF":
                    self.fish_tracker = cv.TrackerKCF_create()
                elif self.tracker_type == "CSRT":
                    self.fish_tracker = cv.TrackerCSRT_create()

                # Define an initial bounding box
                self.bbox[0] = cv.selectROI(windowName=f"{self.video_number}. Video's Templates",
                                            img=self.current_frame, showCrosshair=True, fromCenter=True)
                while True:
                    if 0 in self.bbox[0]:
                        self.bbox[0] = cv.selectROI(windowName=f"{self.video_number}. Video's Templates",
                                                    img=self.current_frame, showCrosshair=True, fromCenter=True)
                    else:
                        break

                cv.destroyWindow(winname=f"{self.video_number}. Video's Templates")

                # Get new fish template
                self.get_template(self.current_frame)
                draw_circle_on_template = False
                # Initialize and Update Fish Tracker
                self.fish_tracker.init(self.current_frame, self.bbox[0])
                ok1, self.bbox[0] = self.fish_tracker.update(self.current_frame)

            # Re-creating refuge tracker each time unsuccessful detection occur.
            if not ok2:
                cv.putText(self.current_frame, 'Tracking failure detected, re-create           template !', (20, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.75, (68, 105, 255), 2)
                cv.putText(self.current_frame, '"REFUGE"', (460, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv.putText(self.current_frame, 'To select box press "SPACE"', (20, 45),
                           cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                cv.putText(self.current_frame, f'Frame: {str(int(self.video.get(cv.CAP_PROP_POS_FRAMES)))}', (20, 80),
                           cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                # Create new cage tracker
                if self.tracker_type == "KCF":
                    self.cage_tracker = cv.TrackerKCF_create()
                elif self.tracker_type == "CSRT":
                    self.cage_tracker = cv.TrackerCSRT_create()

                # Define an initial bounding box
                self.bbox[1] = cv.selectROI(windowName=f"{self.video_number}. Video's Templates",
                                            img=self.current_frame, showCrosshair=True, fromCenter=True)
                while True:
                    if 0 in self.bbox[1]:
                        self.bbox[1] = cv.selectROI(windowName=f"{self.video_number}. Video's Templates",
                                                    img=self.current_frame, showCrosshair=True, fromCenter=True)
                    else:
                        break

                cv.destroyWindow(winname=f"{self.video_number}. Video's Templates")

                # Initialize and Update Cage Tracker
                self.cage_tracker.init(self.current_frame, self.bbox[1])
                ok2, self.bbox[1] = self.cage_tracker.update(self.current_frame)

            # Draw Fish bounding box
            if ok1 and draw_circle_on_template:
                # Fish Tracking success
                p1 = (int(self.bbox[0][0]), int(self.bbox[0][1]))
                p2 = (int(self.bbox[0][0] + self.bbox[0][2]), int(self.bbox[0][1] + self.bbox[0][3]))
                cv.rectangle(self.current_frame, p1, p2, (255, 0, 0), 2, 1)
                fish_head_point = (int(self.bbox[0][0] + self.bbox[0][2] / 2),
                                   int(self.bbox[0][1] + self.bbox[0][3] / 2))
                cv.circle(self.current_frame, fish_head_point, 7, (255, 0, 0), -1)

            # Draw Cage bounding box
            if ok2:
                # Cage Tracking success
                p3 = (int(self.bbox[1][0]), int(self.bbox[1][1]))
                p4 = (int(self.bbox[1][0]) + self.bbox[1][2]), int(self.bbox[1][1] + self.bbox[1][3])
                cage_point = (int(self.bbox[1][0] + self.bbox[1][2] / 2), int(self.bbox[1][1] + self.bbox[1][3] / 2))
                cv.rectangle(self.current_frame, p3, p4, (0, 0, 255), 2, 1)
                cv.circle(self.current_frame, cage_point, 7, (0, 0, 255), -1)

            self.find_position()

            # Calculate Frames per second (FPS)
            fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

            # Display tracker type on frame
            cv.putText(self.current_frame, self.tracker_type + " Tracker", (20, 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display FPS on frame
            cv.putText(self.current_frame, "FPS : " + str(int(fps)), (20, 80),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display correlation coefficient number
            cv.putText(self.current_frame, "Fish Correlation coefficient:  " + str(round(self.max_correlation, 4)),
                       (300, 45), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            # Display Frame number
            cv.putText(self.current_frame, f'Frame: {str(int(self.video.get(cv.CAP_PROP_POS_FRAMES)))}', (20, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display result
            cv.imshow("Tracking", self.current_frame)

            self.interrupt_flow(draw_circle_on_template)

            # for Recording Purposes
            if self.record:
                if one_shot:  # Creates just one times VideoWriter Object
                    result = cv.VideoWriter(self.video_path[:-4] + "_TRACK.avi", cv.VideoWriter_fourcc(*'XVID'), 25,
                                            (self.current_frame.shape[1], self.current_frame.shape[0]), isColor=True)
                    one_shot = False
                result.write(self.current_frame)

        result.release()
        self.video.release()
        cv.destroyAllWindows()

    def find_position(self):
        """
        Finds x positions of cage and fish objects.

        :return: empty
        """

        fish_x_position = self.bbox[0][0] + self.bbox[0][2] / 2
        cage_x_position = self.bbox[1][0] + self.bbox[1][2] / 2
        if fish_x_position == 0.0:
            fish_x_position = -1000
        if cage_x_position == 0.0:
            fish_x_position = -1000

        print(f"{int(self.video.get(cv.CAP_PROP_POS_FRAMES))} : {fish_x_position}")  # mid point of tracking area
        self.fish_position[int(self.video.get(cv.CAP_PROP_POS_FRAMES)) - 1] = fish_x_position
        self.cage_position[int(self.video.get(cv.CAP_PROP_POS_FRAMES)) - 1] = cage_x_position

    def store_positions(self):
        """
        Writes cage and fish positions to excel file.

        :return:
        """
        file_name = os.path.join(self.excel_path)
        workbook = xlsxwriter.Workbook(file_name)
        worksheet = workbook.add_worksheet()
        content = ["Fish", "Cage"]
        columns_width = [6, 6]
        header_format = workbook.add_format({
            'font_color': 'black',
            'bold': 1,
            'font_size': 13,
            'align': 'left'
        })
        column = 0
        for item in content:
            worksheet.write(0, column, item, header_format)
            worksheet.set_column(column, column, columns_width[column])
            column += 1

        data = {
            "Fish": self.fish_position,
            "Cage": self.cage_position
        }
        df = pd.DataFrame(data)
        df.to_excel(file_name, index=False)

    def start_offline_analysis(self, start=0, stop=1500):
        """
        Creates cage and fish position and theirs fft analyses

        :param start: start frame
        :param stop: stop frame
        :return: empty
        """
        # Time Domain Signals (Cage and Fish)
        cage_arr_np = np.array(self.cage_position, dtype="float32")
        fish_arr_np = np.array(self.fish_position, dtype="float32")

        crop_time = range(start, stop)
        N = len(cage_arr_np[crop_time,])
        Ts = 40e-3
        Fs = 1 / Ts
        t = np.arange(N) * Ts
        f = np.arange(0, N / 10) / N * Fs

        final_cropped_cage_signal = cage_arr_np[crop_time,] - np.mean(cage_arr_np[crop_time,])
        final_cropped_fish_signal = fish_arr_np[crop_time,] - np.mean(fish_arr_np[crop_time,])

        fig, (ax1) = plt.subplots(1, 1, figsize=(20, 10))
        ax1.set(xlabel="Zaman", ylabel="Balık ve Tüp Koordinatları")
        fig.suptitle("Takip Analizi", fontsize="xx-large")
        ax1.plot(t, final_cropped_cage_signal, color="r", label="tüp", linewidth=1.5)
        ax1.plot(t, final_cropped_fish_signal, color="b", label="balık", linewidth=1.5)
        ax1.legend()
        if self.DEBUG:
            fig.savefig(self.debug_path + '1_Time_Domain_cage_vs_fish.jpg')
            plt.close(fig)
        else:
            plt.show()

        # Frequency Domain Signals (Cage and Fish)
        fft_complex_cage = fft(final_cropped_cage_signal, N)
        fft_abs_cage = np.abs(fft_complex_cage)
        fft_complex_fish = fft(final_cropped_fish_signal, N)
        fft_abs_fish = np.abs(fft_complex_fish)

        fig = plt.figure(figsize=(20, 10))
        _, _, _ = plt.stem(f, fft_abs_cage[:len(f), ], linefmt='red', markerfmt='Dr', bottom=1.1)
        _, _, _ = plt.stem(f, fft_abs_fish[:len(f), ], linefmt='black', markerfmt='ok', bottom=1.1)
        plt.legend(("refuge", "fish"))
        if self.DEBUG:
            fig.savefig(self.debug_path + '2_FFT_cage_vs_fish.jpg')
            plt.close(fig)
        else:
            plt.show()

    def get_template(self, frame):
        """
        Returns Estimated,Cropped fish object

        :param frame:
        :return: empty
        """

        # Getting Fish Template Operations
        yi = self.bbox[0][1]   # initial y position
        yf = yi + self.bbox[0][3]  # final y position
        xi = self.bbox[0][0]  # initial x position
        xf = xi + self.bbox[0][2]  # final x position
        self.fish_template = frame[yi:yf, xi:xf]

    def calculate_correlation_coefficient(self):
        """
        Calculate correlation coefficient user selected template and tracker estimated template

        :return: empty
        """

        yi = self.bbox[0][1]  # initial y position
        yf = yi + self.bbox[0][3]  # final y position
        xi = self.bbox[0][0]  # initial x position
        xf = xi + self.bbox[0][2]  # final x position
        fish_match = cv.matchTemplate(self.current_frame[yi:yf + 1, xi:xf + 1], self.fish_template, cv.TM_CCOEFF_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(fish_match)
        self.max_correlation = maxVal
        # print(self.max_correlation)
        # cv.imshow("Template", self.fish_template)
        # cv.imshow("Fish", self.current_frame[yi:yf + 1, xi:xf + 1])

    def interrupt_flow(self, draw_circle_on_template):

        self.interrupt_key = cv.waitKey(1) & 0xFF

        # Terminate program if t pressed
        if self.interrupt_key == ord('t'):
            self.store_positions()
            sys.exit()

        # Stops program if c pressed
        if self.interrupt_key == ord("p"):
            print("c pressed")
            print('"r":select Roi, "n":next frame, "b":previous frame, "q":continue')
            flag = True
            while True:
                next_frame = self.video.get(cv.CAP_PROP_POS_FRAMES)
                curr_frame = next_frame - 1
                previous_frame = curr_frame - 1

                # Update Cage and Fish Tracker
                ok1, self.bbox[0] = self.fish_tracker.update(self.current_frame)
                ok2, self.bbox[1] = self.cage_tracker.update(self.current_frame)

                self.calculate_correlation_coefficient()
                # Draw Fish bounding box
                if ok1 and draw_circle_on_template:
                    # Fish Tracking success
                    p1 = (int(self.bbox[0][0]), int(self.bbox[0][1]))
                    p2 = (int(self.bbox[0][0] + self.bbox[0][2]), int(self.bbox[0][1] + self.bbox[0][3]))
                    cv.rectangle(self.current_frame, p1, p2, (255, 0, 0), 2, 1)
                    fish_head_point = (int(self.bbox[0][0] + self.bbox[0][2] / 2),
                                       int(self.bbox[0][1] + self.bbox[0][3] / 2))
                    cv.circle(self.current_frame, fish_head_point, 7, (255, 0, 0), -1)

                draw_circle_on_template = True

                # Draw Cage bounding box
                if ok2:
                    # Cage Tracking success
                    p3 = (int(self.bbox[1][0]), int(self.bbox[1][1]))
                    p4 = (int(self.bbox[1][0]) + self.bbox[1][2]), int(self.bbox[1][1] + self.bbox[1][3])
                    cage_point = (int(self.bbox[1][0] + self.bbox[1][2] / 2),
                                  int(self.bbox[1][1] + self.bbox[1][3] / 2))
                    cv.rectangle(self.current_frame, p3, p4, (0, 0, 255), 2, 1)
                    cv.circle(self.current_frame, cage_point, 7, (0, 0, 255), -1)

                if flag:
                    flag = False
                else:
                    self.find_position()

                    # Display tracker type on frame
                    cv.putText(self.current_frame, self.tracker_type + " Tracker", (20, 20),
                               cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

                    # Display correlation coefficient number
                    cv.putText(self.current_frame,
                               "Fish Correlation coefficient:  " + str(round(self.max_correlation, 4)),
                               (300, 45), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                    # Display Frame number
                    cv.putText(self.current_frame, f'Frame: {str(int(self.video.get(cv.CAP_PROP_POS_FRAMES)))}',
                               (20, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

                # Display result
                cv.imshow("Tracking", self.current_frame)

                self.pressed_key = cv.waitKey(0) & 0xFF

                # Exit if q pressed
                if self.pressed_key == ord('q'):
                    break

                # Previous frame
                elif self.pressed_key == ord('b'):
                    print("b pressed")
                    self.video.set(cv.CAP_PROP_POS_FRAMES, previous_frame)
                    # Read a new frame
                    ok, self.current_frame = self.video.read()
                    if not ok:
                        break
                    print(int(self.video.get(cv.CAP_PROP_POS_FRAMES)))

                # Next Frame
                elif self.pressed_key == ord("n"):
                    print("n pressed")
                    self.video.set(cv.CAP_PROP_POS_FRAMES, next_frame)
                    # Read a new frame
                    ok, self.current_frame = self.video.read()
                    if not ok:
                        break
                    print(int(self.video.get(cv.CAP_PROP_POS_FRAMES)))

                # Select Roi
                elif self.pressed_key == ord("r"):
                    print("Select new bound box!")
                    self.video.set(cv.CAP_PROP_POS_FRAMES, curr_frame)

                    # Read a new frame
                    ok, self.current_frame = self.video.read()
                    if not ok:
                        break

                    # Create new fish tracker
                    if self.tracker_type == "KCF":
                        self.fish_tracker = cv.TrackerKCF_create()
                    elif self.tracker_type == "CSRT":
                        self.fish_tracker = cv.TrackerCSRT_create()

                    # Define an initial bounding box
                    self.bbox[0] = cv.selectROI(windowName=f"{self.video_number}. Video's Templates",
                                                img=self.current_frame, showCrosshair=True, fromCenter=True)
                    while True:
                        if 0 in self.bbox[0]:
                            self.bbox[0] = cv.selectROI(windowName=f"{self.video_number}. Video's Templates",
                                                        img=self.current_frame, showCrosshair=True,
                                                        fromCenter=True)
                        else:
                            break
                    cv.destroyWindow(winname=f"{self.video_number}. Video's Templates")

                    # Get new fish template
                    self.get_template(self.current_frame)
                    draw_circle_on_template = False

                    # Initialize and Update Fish Tracker
                    self.fish_tracker.init(self.current_frame, self.bbox[0])
                    ok1, self.bbox[0] = self.fish_tracker.update(self.current_frame)
                    break
                else:
                    print("Please enter valid key !!")
                    print('"r":select Roi, "n":next frame, "b":previous frame, "q":continue')
                    flag = True
