import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import submission
import findM2

# simple script to test all the functions written

im1 = plt.imread("../data/im1.png")
im2 = plt.imread("../data/im2.png")
M = max((im1.shape[0], im1.shape[1], im2.shape[0], im1.shape[1]))

intrinsics = np.load("../data/intrinsics.npz")
K1 = intrinsics['K1']
K2 = intrinsics['K2']

M1 = np.asarray(([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]))

noisy = np.load("../data/some_corresp_noisy.npz")
pts1 = noisy["pts1"]
pts2 = noisy["pts2"]
# run ransac to find the best fundamental matrix using noisy points
F, inliers = submission.ransacF(pts1, pts2, M)
# np.savez("ransacF.npz", F=F, inliers=inliers)

C1 = np.matmul(K1, M1)
idxs = np.nonzero(inliers)
pts1_inliers = pts1[idxs]
pts2_inliers = pts2[idxs]

# compare M2 solution after bundle adjustment using projected points error
M2, C2, P_init = findM2.test_M2_solution(pts1_inliers, pts2_inliers, intrinsics)
w, err_init = submission.triangulate(C1, pts1_inliers, C2, pts2_inliers)
M2_fin, P = submission.bundleAdjustment(K1, M1, pts1_inliers, K2, M2, pts2_inliers, P_init)
C2 = np.matmul(K2, M2_fin)
w, err = submission.triangulate(C1, pts1_inliers, C2, pts2_inliers)

print("M2_init=", M2)
print("error_init", err_init)
print("M2_fin=", M2_fin)
print("error_fin", err)

# plot reprojected points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P[:,0], P[:,1], P[:,2], c='r')
ax.scatter(P_init[:,0], P_init[:,1], P_init[:,2], c='b')
plt.show()
