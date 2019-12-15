import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanade

# write your script here, we recommend the above libraries for making your animation
fig = plt.figure()
ax = fig.add_subplot(111)
rect = [59.0, 116.0, 145.0, 151.0]
frames = np.load("../data/carseq.npy")
carseqrects = []
carseqrects.append(rect)
p = [0, 0]

# for each frame in the dataset
for idx in range(frames.shape[2] - 1):
    print(idx)

    # find the template image in the current frame and update the rectangle bounding box accordingly
    p = LucasKanade.LucasKanade(frames[:,:,idx], frames[:,:,idx+1], rect, p)
    rect[0] += p[0]
    rect[2] += p[0]
    rect[1] += p[1]
    rect[3] += p[1]
    carseqrects.append(rect.copy())


carseqrects = np.asarray(carseqrects)
print(carseqrects.shape)
np.save('carseqrects.npy', carseqrects)

#for visualization
carseqrects = np.load('carseqrects.npy')
for idx in range(frames.shape[2] - 1):
    print(idx)  
    img = frames[:,:,idx].copy()
    rect = carseqrects[idx]
    width = rect[2] - rect[0] + 1
    height = rect[3] - rect[1] + 1
    rectangle = patches.Rectangle((rect[0], rect[1]), width, height, edgecolor='yellow', fill=False)
    ax.add_patch(rectangle)
    plt.imshow(img, cmap='gray')
    if idx in [1, 100, 200, 300, 400]:
        plt.savefig("LK_"+str(idx)+".png")
    plt.pause(0.005)
    ax.clear()