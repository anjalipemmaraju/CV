# Lucas Kanade (and variants)
### Background
<<<<<<< HEAD
Lucas Kanade is an algorithm relating to optical flow, or the way of describing how pixels are moving between frames. In the general case, it uses the gradient of the image with respect to x and y and the change in pixel intensities to determine where a template image is in the current image. This implementation attempts to use Lucas Kanade to track different objects throughout a sequence of image frames. A description of the tasks is provided in hw3.pdf.
=======
Lucas Kanade is an algorithm relating to optical flow, or the way of describing how pixels are moving between frames. In the general case, it uses the gradient of the image with respect to x and y and the change in pixel intensities to determine where a template image is in the current image. This implementation attempts to use Lucas Kanade to track different objects throughout a sequence of image frames. A description of the tasks is provided in hw3.pdf. Data is 100 MB and cannot be uploaded to github right now.
>>>>>>> 433a14c9bf929ddd16114ca561c67f3316b7c49e

### Code
The code folder contains all the code for this project.
* testCarSequence.py reads in a sequence of frames of a car driving. We were given the bounding box of the car in the first image frame to use as the template. The script runs the LucasKanade function to compute the position of the bounding box around the car in each of the subsequent frames. These results are shown in the results folder.
* testCarSequenceWithTemplateCorrection.py reads in the same sequence of frames as testCarSequence but applies template correction to correct for template drift between the frames. The results are shown in the results folder.
* testSylvSequence.py reads in a sequence of frames of a sylvester stuffed animal oving through different lighting and closer and farther from the camera. Both the original Lucas Kanade and LucasKanade with Bases implementation are run. The comparison of their results is shown in the results folder.
* testAerialSequence.py reads in some frames of an aerial view of cars moving and attempts to determine which pixels are different between the frames. It calls subtract dominant motion to do this.
* SubtractDominantMotion.py uses the Lucas Kanade with affine transformation to determine which pixels are different between two successive image frames of the aerial sequence. It computes a mask representing the changed pixels
* LucasKanade.py is used to track how an object has moved between frames. Given a bounding box representing the location of the object in the first frame, this function finds a good bounding box for the same object in the second frame based only on translating the box. It returns an x and y representing how the bounding box has moved between the frames.
* LucasKanadeAffine.py is used for the same purpose as LucasKanade.py except that the object in the bounding box can now be warped in an affine manner instead of just translated. It returns a matrix representing an affine warp of how the bounding box has moved between frames.
* LucasKanadeBasis.py is used for the same purpose as LucasKanade.py but is more robust to lighting changes and size changes because of the use of the basis. It tries to find the difference between the bases and your calculated warped object and returns an x and y representing how the box has moved between frames.
* InverseCompositinAffine.py calculates the same thing as LucasKanadeAffine.py but in a more computationally efficient way due to the Inverse Composition.

### Results
The results folder contains images of the tracking performance of each algorithm
<<<<<<< HEAD
* Files beginning with LK_(number) show images of tracking performance of the car driving with simple LucasKanade
* Files beginning with LK_aerial show images of tracking performance using a mask to represent changed pixels in the aerial sequence using LucasKanadeAffine
* Files beginning with LK_wcrt show images of tracking performance of the car driving with LucasKanade with template correction
* Files beginning with LKB show images of tracking performance of the Sylvester stuffed animal using LucasKanade with Basis
=======
* Files beginning with LK_ show images of tracking performance of the car driving with simple LucasKanade over frames 1, 100, 200, 300 and 400.
* Files beginning with LK_aerial show images of tracking performance using a mask to represent changed pixels in the aerial sequence using LucasKanadeAffine over frames 30, 60, 90, and 120.
* Files beginning with LK_wcrt show images of tracking performance of the car driving with LucasKanade with template correction over frames 1, 100, 200, 300, and 400.
* Files beginning with LKB show images of tracking performance of the Sylvester stuffed animal using LucasKanade with Basis over frames 1, 100, 200, 300, 350, and 400.
>>>>>>> 433a14c9bf929ddd16114ca561c67f3316b7c49e
