import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import time

'''
Input: 
	It: template image
	It1: Current image
Output:
	M: the Affine warp matrix [2x3 numpy array]
put your implementation here
'''
def LucasKanadeAffine(It, It1):
	# start with identity as M guess
	M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
	threshold = 0.15

	# create bivariate spline for interpolation between pixels
	It_spline = RectBivariateSpline(np.array(range(0, It.shape[0])), np.array(range(0, It.shape[1])), It)

	XX, YY = np.meshgrid(np.array(range(0, It.shape[0])), np.array(range(0, It.shape[1])))
	#XX corresponds to columns, YY corresponds to rows
	norm = threshold + 1
	Ix_grad = It_spline.ev(YY, XX, dx=1, dy=0)
	Iy_grad = It_spline.ev(YY, XX, dx=0, dy=1)

	# create valid region to only compare pixels where the two frames overlap
	valid_region = np.ones((It.shape[0], It.shape[1]))
	
	XX = XX.flatten()
	YY = YY.flatten()
	ones = np.ones((XX.shape))
	zeros = np.zeros((XX.shape))

	#represents warp jacobian
	warp_x = np.asarray(([XX, YY, ones, zeros, zeros, zeros]))
	warp_y = np.asarray(([zeros, zeros, zeros, XX, YY, ones]))
	while norm > threshold:
		#warp the gradients		
		Ix_warp = np.expand_dims(affine_transform(Ix_grad, M).flatten(), axis=1)
		Iy_warp = np.expand_dims(affine_transform(Iy_grad, M).flatten(), axis=1)
		I_warp = affine_transform(It, M)

		#must figure out overlap region here
		valid = affine_transform(valid_region, M)
		valid_It1 = valid * It1

		#construct A
		Ix = Ix_warp * warp_x.T
		Iy = Iy_warp * warp_y.T
		A = Ix + Iy
		b = (I_warp - valid_It1).flatten()

		delta_p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
		norm = np.linalg.norm(delta_p)
		
		# update affine warp matrix based on delta_p
		M[0][0] += delta_p[0]
		M[0][1] += delta_p[1]
		M[0][2] += delta_p[2]
		M[1][0] += delta_p[3]
		M[1][1] += delta_p[4]
		M[1][2] += delta_p[5]
	
	return M
