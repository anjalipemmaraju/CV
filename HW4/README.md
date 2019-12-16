# Structure From Motion

Given two camera views of an object, we want to be able to calculate a transformation of some points in those camera frames to get a 3D representation of the object. We assume that we only have images of a simple temple from 2 cameras. More detail about the assignment is in hw4.pdf

### Code
* submission.py : main file that has functions to calculate the fundamental matrix using the 8 point algorithm, 7 point algorithm, and RANSAC. It also contains functions to determine the reprojection error of points from 2 images, and a function to find the points in image 2 that correspond to points in image 1. The final function uses the Rodrigues formula to calculate Bundle Adjustment on a set of noisy correspondences between 2 camera frames.
* findM2.py: Defines a function to find the camera extrinsics of camera 2 given the intrinsics of both cameras and the point correspondences.
* main.py : script to run some of the functions in submission.py
* test5_3.py : script to test the bundle adjustment function
* helper.py & visualize.py : files given to us

### Results
Pictures q2_1 and q2_2 show how each of the different ways that I calculated the fundamental matrix or essential matrix resulted in correct epipolar lines. Pictures q4_1 and q5_1 are for finding the epipolar lines in image 2 given points in image 1. temple1, temple2, and temple3 are examples of using all these together to reproject the 2D points from camera 1 and 2 into 3D to make a 3D representation of the temple shown in the previous images. new_points, old_points, and points_shift relate to the bundle adjustment portion of the assignment when we were given noisy correspondences. They show how the reprojected noisy points and the old points are pretty similar (hard to tell unless you were a grader and knew what to look for)
