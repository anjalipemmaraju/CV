import numpy as np
from scipy.interpolate import RectBivariateSpline
'''
Input: 
	It: template image
	It1: Current image
	rect: Current position of the car
	(top left, bot right coordinates)
	p0: Initial movement vector [dp_x0, dp_y0]
Output:
	p: movement vector [dp_x, dp_y]
'''
def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	p = p0
	threshold = 0.01
	
	# create bivariate spline to help with interpolation of the pixels in the images
	It_spline = RectBivariateSpline(np.array(range(0, It.shape[0])), np.array(range(0, It.shape[1])), It)
	It1_spline = RectBivariateSpline(np.array(range(0, It1.shape[0])), np.array(range(0, It1.shape[1])), It1)
	
	width = np.array(range(int(rect[2]) - int(rect[0]) + 1))
	height = np.array(range(int(rect[3]) - int(rect[1]) + 1))
	
	XX, YY = np.meshgrid(width, height)

	#XX corresponds to columns, YY corresponds to rows
	XX = XX.astype(float)
	YY = YY.astype(float)

	#move box to initial guess to start
	XX += rect[0] + p[0]
	YY += rect[1] + p[1]
	
	XX_template, YY_template = np.meshgrid(width, height)
	XX_template = XX_template.astype(float)
	YY_template = YY_template.astype(float)
	XX_template += rect[0]
	YY_template += rect[1]
	
	#evaluate the before image at the known box location
	template = It_spline.ev(YY_template, XX_template)
	norm = threshold + 1

	while norm >= threshold:
		#p is col, rows
		#estimate gradients
		#XX and YY are already at the box for the first frame
		#warp is just a translation, so no need to do anything with the gradients because it's already 
		# estimated at your guess of the position and it should be the same size
		Ix_warp = It1_spline.ev(YY, XX, dx=1, dy=0)
		Iy_warp = It1_spline.ev(YY, XX, dx=0, dy=1)
		I_warp = It1_spline.ev(YY, XX)

		A = np.asarray((Iy_warp.flatten(), Ix_warp.flatten())).T
		b = (template - I_warp).flatten()
		delta_p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
		norm = np.linalg.norm(delta_p)
		p += delta_p

		#just want to move the box to our new guess
		XX = XX + delta_p[0]
		YY = YY + delta_p[1]

	return p
