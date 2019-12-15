import numpy as np
import helper
import submission
import matplotlib as plt

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

'''
Estimate all possible M2 and return the correct M2 and 3D points P
:param pred_pts1:
:param pred_pts2:
:param intrinsics:
:return: M2, the extrinsics of camera 2
			C2, the 3x4 camera matrix
			P, 3D points after triangulation (Nx3)
'''
def test_M2_solution(pts1, pts2, intrinsics):
	im1 = plt.pyplot.imread("../data/im1.png")
	im2 = plt.pyplot.imread("../data/im2.png")
	M = max((im1.shape[0], im1.shape[1], im2.shape[0], im1.shape[1]))
	F = submission.eightpoint(pts1, pts2, M)

	K1 = intrinsics['K1']
	K2 = intrinsics['K2']
	E = submission.essentialMatrix(F, K1, K2)

	M1 = np.asarray(([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]))
	M1 = M1.astype('float')
	M2 = helper.camera2(E)
	min_err = np.float('Inf')
	P = []
	M2_return = []
	
	# find the correct M2 (camera extrinsics) based on the triangulate function
	for i in range(0, 4):
		m2 = M2[:,:,i]
		C1 = np.matmul(K1, M1)
		C2 = np.matmul(K2, m2)
		w, err = submission.triangulate(C1, pts1, C2, pts2)
		wcol = w[:,2]
		mask = sum(wcol<0)
		if mask <= 0:
			P = w
			M2_return = m2
			break

	return M2_return, C2, P


if __name__ == '__main__':
	data = np.load('../data/some_corresp.npz')
	pts1 = data['pts1']
	pts2 = data['pts2']
	
	intrinsics = np.load('../data/intrinsics.npz')

	M2, C2, P = test_M2_solution(pts1, pts2, intrinsics)
	np.savez('q3_3', M2=M2, C2=C2, P=P)
