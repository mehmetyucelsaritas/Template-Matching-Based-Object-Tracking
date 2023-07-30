# Template-Matching-Based-Object-Tracking
### Template Matching Based Fully Automated Fast Ghost Fish Tracking Project

This project uses the Template matching method to track objects.
[See OpenCV Template Matching Methode](https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html)



****You can  [watch](https://youtu.be/ZMCFsfx7JNw) the demo of project.****

See also [Object-Tracking-GUI](https://github.com/mehmetyucelsaritas/Object-Tracking-GUI) Project that mainly focus on deleting outlier data.

**Basic Usage of Project :**

1. Select Range Of Interests (ROIs) --> Press SPACE after each.
2. After selecting 2 ROIs press ESC (You can change the code to add a new roi or to delete one of them)
3. You can press "t" to terminate the program.
4. You can press "p" to pause the video flow.. 
5. You can use "n", "b", "q", and "r"  keys to pass the next frame, to pass the previous frame, to quit from pause, and select a new ROI respectively..

**After you successfully run the project, You get the following :**
1. Object Tracked video of the original video.
2. Tracking data in the format of .xlsx
3. The plot of data in the format of .png

Notes : You can increase the [correlation coefficient ](https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html) threshold (Constant in main.py named CORRELATION_THRESHOLD) to get more accurate tracking result. But the selection number of ROI increases.



