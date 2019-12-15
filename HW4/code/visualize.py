'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import submission
import numpy as np
import matplotlib.pyplot as plt
import helper
import findM2
from mpl_toolkits.mplot3d import Axes3D

selected = np.load("../data/templeCoords.npz")
data = np.load("../data/some_corresp.npz")
print(data.files)
pts1 = data["pts1"]
pts2 = data["pts2"]

im1 = plt.imread("../data/im1.png")
im2 = plt.imread("../data/im2.png")
M = max((im1.shape[0], im1.shape[1], im2.shape[0], im1.shape[1]))
F = submission.eightpoint(pts1, pts2, M)

xs = selected['x1'].T[0]
ys = selected['y1'].T[0]
pts1 = np.asarray([xs,ys]).T
correspondences = []
for i in range(0,len(xs)):
    x = xs[i]
    y = ys[i]
    x_match, y_match = submission.epipolarCorrespondence(im1, im2, F, x, y)
    correspondences.append([x_match, y_match])

correspondences = np.asarray(correspondences)
intrinsics = np.load("../data/intrinsics.npz")
K1 = intrinsics['K1']
M1 = np.asarray(([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]))
C1 = np.matmul(K1, M1)

M2, C2, P = findM2.test_M2_solution(pts1, correspondences, intrinsics)

np.savez("q4_2.npz", F=F, M1=M1, M2=M2, C1=C1, C2=C2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P[:,0], P[:,1], P[:,2])
plt.show()


