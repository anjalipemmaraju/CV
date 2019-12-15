import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanadeBasis
import LucasKanade

# write your script here, we recommend the above libraries for making your animation
fig = plt.figure()
ax = fig.add_subplot(111)
rect_lkb = [101.0, 61.0, 155.0, 107.0]
rect_lk = [101.0, 61.0, 155.0, 107.0]

lk_rects = []
lkb_rects = []

lk_rects.append(rect_lk.copy())
lkb_rects.append(rect_lkb.copy())
frames = np.load("../data/sylvseq.npy")
bases = np.load("../data/sylvbases.npy")
p0 = np.zeros(2)

# run loop for every frame in the sequence
for idx in range(frames.shape[2] - 1):
    # run LK with bases because of lighting changes
    p = LucasKanadeBasis.LucasKanadeBasis(frames[:,:,idx], frames[:,:,idx+1], rect_lkb, bases)
    rect_lkb[0] += p[0]
    rect_lkb[2] += p[0]
    rect_lkb[1] += p[1]
    rect_lkb[3] += p[1]
    lkb_rects.append(rect_lkb.copy())

    # run LK without bases to see the difference
    p0 = LucasKanade.LucasKanade(frames[:,:,idx], frames[:,:,idx+1], rect_lk, p0)
    rect_lk[0] += p0[0]
    rect_lk[2] += p0[0]
    rect_lk[1] += p0[1]
    rect_lk[3] += p0[1]
    lk_rects.append(rect_lk.copy())

lkb_rects = np.asarray(lkb_rects)
np.save('sylvseqrects.npy', lkb_rects)

# for visualization
for idx in range(frames.shape[2] - 1):
    img = frames[:,:,idx].copy()

    rect = lk_rects[idx]
    width = rect[2] - rect[0] + 1
    height = rect[3] - rect[1] + 1
    rectangle = patches.Rectangle((rect[0], rect[1]), width, height, edgecolor='green', fill=False)
    ax.add_patch(rectangle)

    rect = lkb_rects[idx]
    width = rect[2] - rect[0] + 1
    height = rect[3] - rect[1] + 1
    rectangle = patches.Rectangle((rect[0], rect[1]), width, height, edgecolor='yellow', fill=False)
    ax.add_patch(rectangle)

    plt.imshow(img, cmap='gray')
    if idx in [1, 200, 300, 350, 400]:
        plt.savefig("LKB_"+str(idx)+".png")
    plt.pause(0.005)
    ax.clear()
