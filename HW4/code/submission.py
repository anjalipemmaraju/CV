"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper
import scipy.ndimage
import random
import math
#from scipy.optimize import minimize

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    pts1 = pts1 / M
    pts2 = pts2 / M
    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    x2 = pts2[:, 0]
    y2 = pts2[:, 1]
    ones = np.ones(len(y2))

    # construct A matrix and compute SVD and find fundamental matrix 
    A = np.array([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, ones])
    U,L,V = np.linalg.svd(A.T)
    V = V.T
    f = np.reshape(V[:,-1], (3, 3))
    f = helper._singularize(f)
    f = helper.refineF(f, pts1, pts2)
    T = np.array([[1/M, 0, 0],[0, 1/M, 0], [0, 0, 1]])
    f = np.matmul(np.matmul(T.T, f), T)

    return f


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    pts1 = pts1 / M
    pts2 = pts2 / M
    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    x2 = pts2[:, 0]
    y2 = pts2[:, 1]
    ones = np.ones(len(y2))
    
    A = np.array([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, ones])
    U,L,V = np.linalg.svd(A.T)

    f1 = V.T[:, -1]
    f2 = V.T[:, -2]
    f1 = np.reshape(f1, (3, 3))
    f2 = np.reshape(f2, (3, 3))

    # solve a polynomial equation based on the A matrix since there are only 7 DOF in the fundamental matrix
    fun = lambda a: np.linalg.det(a * f1 + (1 - a) * f2)
    a0 = fun(0)
    a1 = 2*(fun(1) - fun(-1)) / 3 - (fun(2) - fun(-2))/12
    a2 = 0.5*fun(1)+0.5*fun(-1) - fun(0)
    a3 = fun(1) - (a0+a1+a2)

    roots = np.roots([a3, a2, a1, a0])
    f = []
    T = np.array([[1/M, 0, 0],[0, 1/M, 0], [0, 0, 1]])
    thresh = 1e-5
    for root in roots:
        comp = np.imag(root)
        # only pick this root to construct fundamental matrix if it is not imaginary
        if np.abs(comp) < thresh:
            new_f = float(np.real(root))*f1 + (1-float(np.real(root)))*f2
            new_f = np.matmul(np.matmul(T.T, new_f), T)
            f.append(new_f)
    return f


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    E = np.matmul(np.matmul(K2.T, F), K1)
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    w = []
    A = np.zeros((4, 4))
    # for each point in 2D construct a w vector based on each camera's intrinstics
    for i in range(len(pts1)):
        A[0,:] = pts1[i, 1] * C1[2,:] - C1[1,:]
        A[1,:] = pts1[i, 0] * C1[2,:] - C1[0,:]
        A[2,:] = pts2[i, 1] * C2[2,:] - C2[1,:]
        A[3,:] = pts2[i, 0] * C2[2,:] - C2[0,:]

        U,L,V = np.linalg.svd(A)
        w_temp = V.T[:,-1]
        w_temp /= w_temp[-1]
        w.append(w_temp)
    w = np.asarray(w)
    err = 0
    
    # compute the error of the projected points with the 2D points
    for i in range(len(pts1)):
        proj1 = np.matmul(C1, w[i])
        # only take the x y 2D values (not z representing depth) to compute error
        proj1 = (proj1 / proj1[2])[:2]
        err1 = np.linalg.norm(proj1 - pts1[i])

        proj2 = np.matmul(C2, w[i])
        # only take the x y 2D values (not z representing depth) to compute error
        proj2 = (proj2 / proj2[2])[:2]
        err2 = np.linalg.norm(proj2 - pts2[i])

        err += err1 + err2
    w = w[:,:-1]
    return w, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    pt1_homo = np.asarray([[x1], [y1],[1]])
    l2 = np.matmul(pt1_homo.T, F.T)[0]
    # l2 represents a, b, c of the line in image 2
    # take a window in im2
    fun = lambda y: int((-1*l2[1] - l2[2])/l2[0])
    pts = []
    window = 7

    # only look at this x and y if they lie within the center portion of the picture
    for y in range(0, im2.shape[0]):
        x = fun(y)
        if x > window and x < im2.shape[1]-window:
            pts.append([x, y])  

    min_error = np.float('Inf')
    im1_patch = im1[int(y1-window):int(y1+window+1), int(x1-window):int(x1+window+1)]
    match = pts[0]
    for x,y in pts:
        # if the distance between the two points is small, 
        # compute the similarity of the pixels in a small patch around the point
        distance = np.sqrt((x-x1)**2 + (y-y1)**2) 
        if distance < 30:
            im2_patch = im2[y-window:y+window+1, x-window:x+window+1]
            diff = np.abs(im1_patch - im2_patch)

            # weight the difference with a gaussian to favor the center of the patch
            diff_gauss = np.sum(scipy.ndimage.gaussian_filter(diff, sigma=1))

            # find the point with the largest patch similarity
            if diff_gauss < min_error:
                min_error = diff_gauss
                match = x, y
    
    return match


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    num_pts = len(pts1)
    inliers = [False]*num_pts
    
    threshold = 0.001
    best_f = []

    # homogenize all the coordinates
    ones = np.ones(len(pts1))
    pts1 = np.asarray((pts1[:,0], pts1[:,1], ones)).T
    pts2 = np.asarray((pts2[:,0], pts2[:,1], ones)).T
    max_inliers = 0

    for iter in range(0,5000):
        # sample 7 points from each set of noisy points to run the 7 point algorithm
        indices = random.sample(range(0, num_pts), 7)
        pts1_7 = pts1[indices]
        pts2_7 = pts2[indices]
        f = sevenpoint(pts1_7, pts2_7, M)

        # pick the best fundamental matrix out of the ones returned by 7 point
        for fundamental in f:
            # pts2 * F * pts1 should be equal to 0 if F is correct based on the formula
            relation = np.abs(np.sum(np.multiply(np.matmul(pts2, fundamental), pts1), axis=1))

            # inliers are those close to 0
            curr_inliers = relation < threshold
            indices = np.nonzero(curr_inliers)[0]

            # find the best F based on which had the most inliers
            if np.sum(curr_inliers) > max_inliers:
                max_inliers = np.sum(curr_inliers)
                inliers = curr_inliers
                pts1_sample = pts1[indices,:2]
                pts2_sample = pts2[indices,:2]
                
                # refine the fundamental matrix now that you know which are the inliers
                best_f = helper.refineF(fundamental, pts1_sample, pts2_sample)

    return best_f, inliers
            
        
'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta == 0:
        R = np.identity(3)
    else:
        unit = r / theta
        unitcross = np.asarray(([0, -unit[2, 0], unit[1, 0]], [unit[2, 0], 0, -unit[0, 0]], [-unit[1, 0], unit[0, 0], 0]))
        R = np.identity(3)*math.cos(theta) + (1-math.cos(theta))*unit * unit.T + unitcross * math.sin(theta)
    return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    theta = math.acos((np.trace(R) - 1)/2)
    if theta == 0:
        w = np.zeros((3, 1))
    else:
        w = np.asarray([[R[2,1] - R[1,2]],[R[0,2] - R[2,0]], [R[1,0] - R[0,1]]]) / (2*math.sin(theta))
        w = w * theta
    return w

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    p_init = x[:-6]
    rod_rot = x[-6:-3].reshape(3,1)
    trans = x[-3:].reshape(3,1)

    rot = rodrigues(rod_rot)
    m2 = np.asarray(np.concatenate((rot, trans), axis=1))
    C1 = np.matmul(K1, M1)
    C2 = np.matmul(K2, m2)

    p = p_init.reshape(-1, 3)
    ones = np.ones(len(p)).reshape(len(p), 1)
    p_homo = np.asarray(np.concatenate((p, ones), axis=1))

    proj1 = np.matmul(C1, p_homo.T)
    proj1 /= proj1[-1]
    proj1 = proj1[:2]
    err1 = np.abs(proj1.T - p1).flatten()

    proj2 = np.matmul(C2, p_homo.T)
    proj2 /= proj2[-1]
    proj2 = proj2[:2]
    err2 = np.abs(proj2.T - p2).flatten()

    err = np.concatenate((err1, err2)).flatten()
    residuals = err.reshape(4*len(p1), 1)
    return residuals


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    rot = M2_init[:, :3]
    trans = M2_init[:,-1].reshape(3, 1)
    rod_rot = invRodrigues(rot)
    x = np.concatenate((P_init.flatten(), rod_rot.flatten(), trans.flatten())).flatten()
    func = lambda x: np.sum(rodriguesResidual(K1,M1,p1,K2,p2,x)**2)
    res = scipy.optimize.minimize(func,x,method='L-BFGS-B')
    p_final = res.x[:-6]
    rod_rot_final = res.x[-6:-3].reshape(3, 1)
    trans_final = res.x[-3:].reshape(3, 1)
    rot_final = rodrigues(rod_rot_final)
    M2 = np.concatenate((rot_final, trans_final), axis=1)
    print(M2 - M2_init)
    P = p_final.reshape(-1, 3)
    return M2, P
