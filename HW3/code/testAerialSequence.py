import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation
fig = plt.figure()
ax = fig.add_subplot(111)
frames = np.load("../data/aerialseq.npy")
masks = np.zeros((frames.shape[0], frames.shape[1], 4))

# for each frame in the sequence do these steps
for idx in range(frames.shape[2] - 1):
    # find the mask representing pixels which have changed between frames
    mask = SubtractDominantMotion(frames[:,:,idx], frames[:,:,idx+1])
    img = frames[:,:,idx].copy()
    plt.imshow(img, cmap='gray')
    locs = np.where(mask==True)
    plt.plot(locs[1], locs[0], 'b.', markersize = 1, alpha=0.4)

    if idx in [30, 60, 90, 120]:
        plt.savefig("LK_aerial"+str(idx)+".png")
        masks = np.dstack((masks, mask))
    plt.pause(0.05)

    ax.clear()

print(masks.shape)
print(masks)
#np.save('aerialseqmasks.npy', masks)
