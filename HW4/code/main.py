import submission
import helper
import numpy as np
import matplotlib as plt
import random

data = np.load("../data/some_corresp.npz")
print(data.files)
pts1 = data["pts1"]
pts2 = data["pts2"]

# find a good fundamental matrix based on the 8 point algorithm
im1 = plt.pyplot.imread("../data/im1.png")
im2 = plt.pyplot.imread("../data/im2.png")
M = max((im1.shape[0], im1.shape[1], im2.shape[0], im1.shape[1]))
F = submission.eightpoint(pts1, pts2, M)
np.savez("q2_1.npz", F=F, M=M)
np.savez("q4_1.npz", F=F, pts1=pts1, pts2=pts2)
helper.displayEpipolarF(im1, im2, F)


F_7 = np.asarray(([0,0,0],[0,0,0],[0,0,0]))
F_7 += 1
rtol = 1e-04
atol = 1e-04
close = np.allclose(F_7, F, rtol, atol)
# find a good fundamental matrix based on the 7 point algorithm based on how close it is to the 8 point solution
while not close:
    pts = random.sample(range(0, len(pts1)), 7)
    # GOOD POINTS 
    # pts = [50, 85, 29, 77, 93, 16, 14]
    pts1_7 = pts1[pts]
    pts2_7 = pts2[pts]
    f = submission.sevenpoint(pts1_7, pts2_7, M)
    for fundamental in f:
        close = np.allclose(fundamental, F, rtol, atol)
        if close:
            F_7 = fundamental
            np.savez("q2_2.npz", F=F_7, M=M, pts1=pts1, pts2=pts2)
            print("F7=", F_7)
            break

helper.displayEpipolarF(im1, im2, F_7)

intrinsics = np.load("../data/intrinsics.npz")
K1 = intrinsics['K1']
K2 = intrinsics['K2']
E = submission.essentialMatrix(F, K1, K2)

helper.epipolarMatchGUI(im1, im2, F)

