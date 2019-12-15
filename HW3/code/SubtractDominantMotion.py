import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from scipy.ndimage import affine_transform
from scipy.ndimage.morphology import binary_dilation
from InverseCompositionAffine import InverseCompositionAffine

""" Lucas Kanade implemented using inverse composition for affine tranforms
Input
	image1 (data type): template image at time t
	image 2(data type): template image at time t+1
Output
	mask (nxm matrix): mask representing which pixels have changed between image1 and image2
"""
def SubtractDominantMotion(image1, image2):
	mask = np.ones(image1.shape, dtype=bool)

	# only consider pixels in the middle of the images
	mask2 = np.zeros(image1.shape)
	mask2[15:image1.shape[0]-50, 50:image1.shape[1]-50] = 1.0

	threshold = 0.23
	# compute the affine transformation between the 2 images
	M = LucasKanadeAffine(image1, image2)

	#determine the mask representing the changed pixels
	warped_image = affine_transform(image1, M)
	new_im = np.absolute(image2 - warped_image)
	mask = (new_im > threshold) * mask2
	mask = binary_dilation(mask, np.ones((5, 5)))
	return mask
