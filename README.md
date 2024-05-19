# RA

For this task the main script is ra.py. Most of the code is based on the previus work of my teacher, but there are some changes and some new features. The main changes are:
- The function get_M: To get the transformation matrix instead of getting and approximation of the K matrix I used a file extracted from the page https://calibdb.net. This page contains the calibration data of previous calibrated cameras, speeding up the process of calibration and allowing me to work not only with the K matrix but also with the distortion coefficients.
- bestPose was addapted to work with the new get_M function.

The added features could be summarized in the following list:
- The user can now move objects with the mouse, this is done by pressing w, a, s and d.
- The user can now move the object with the mouse by clicking on the place where he wants the object to be. The object would be move in the same plane as the reference object.