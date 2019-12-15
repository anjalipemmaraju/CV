import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform

""" Lucas Kanade implemented using inverse composition for affine tranforms
Inputs
	It (data type): template image at time t
	It1 (data type): template image at time t+1

Returns
	M: (2x3 numpy array): the Affine warp matrix
"""
def InverseCompositionAffine(It, It1):
	M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
	threshold = 0.15

	It_spline = RectBivariateSpline(np.array(range(0, It.shape[0])), 
									np.array(range(0, It.shape[1])), 
									It)

	# XX corresponds to columns, YY corresponds to rows
	XX, YY = np.meshgrid(np.array(range(0, It.shape[0])), np.array(range(0, It.shape[1])))

	norm = threshold + 1
	# gradient of image in x and y direction
	It_x_grad = np.expand_dims(It_spline.ev(YY, XX, dx=1, dy=0).flatten(), axis=1)
	It_y_grad = np.expand_dims(It_spline.ev(YY, XX, dx=0, dy=1).flatten(), axis=1)

	valid_region = np.ones((It.shape[0], It.shape[1]))
	XX = XX.flatten()
	YY = YY.flatten()
	ones = np.ones((XX.shape))
	zeros = np.zeros((XX.shape))

	# represents warp jacobian
	warp_x = np.asarray(([XX, YY, ones, zeros, zeros, zeros]))
	warp_y = np.asarray(([zeros, zeros, zeros, XX, YY, ones]))

	A = It_x_grad * warp_x.T + It_y_grad * warp_y.T

	# loop until calculated affine matrix resembles the correct affine matrix
	while norm > threshold:
		It1_warp = affine_transform(It1, M)
		valid = affine_transform(valid_region, M)
		I_warp = affine_transform(It, M)
		valid_It1 = valid * It1_warp

		# comapre the warped template and warped image frame to each other
		b = (I_warp - It1_warp).flatten()
		delta_p, _, _, _ = np.linalg.lstsq(A, b)
		norm = np.linalg.norm(delta_p)

		# compute how to change M based on the difference between the warped template and the warped image frame
		m_delta_p = np.asarray(([1 + delta_p[0], delta_p[1], delta_p[2]], 
								[delta_p[3], 1+ delta_p[4], delta_p[5]], 
								[0, 0, 1]))
		M = np.dot(M, np.linalg.inv(m_delta_p))

	return M
