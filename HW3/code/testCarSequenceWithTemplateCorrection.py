import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanade

# write your script here, we recommend the above libraries for making your animation
fig = plt.figure()
ax = fig.add_subplot(111)
rect = [59.0, 116.0, 145.0, 151.0]
template_rect = np.copy(rect)
frames = np.load("../data/carseq.npy")
carseqrects_wcrt = []
carseqrects_wcrt.append(rect)
threshold = 30
carseqrects = np.load('carseqrects.npy')
pn = [0, 0]
for idx in range(frames.shape[2] - 1):
    print(idx)
    pn = LucasKanade.LucasKanade(frames[:,:,idx], frames[:,:,idx+1], rect, pn)

    #pn is the difference between current and next frame with given rectangle
    #since we are using the template_rect we need to correct for the different between template_rect and rect
    #because pn should be the difference 
    p_pass = (pn[0] + rect[0] - template_rect[0], pn[1] + rect[1] - template_rect[1])
    pn_star = LucasKanade.LucasKanade(frames[:,:,0], frames[:,:,idx+1], template_rect, p_pass)

    p_norm = np.linalg.norm(pn_star - p_pass)

    # if p_norm is less than the threshold then update with the template_rect, otherwise do nothing
    if (p_norm < threshold):
        rect[0] = template_rect[0] + pn_star[0]
        rect[2] = template_rect[2] + pn_star[0]
        rect[1] = template_rect[1] + pn_star[1]
        rect[3] = template_rect[3] + pn_star[1]

    carseqrects_wcrt.append(rect.copy()) 

    img = frames[:,:,idx].copy()

    # save the bounding boxes as rectangle patches for each frame
    rect_wcrt = carseqrects_wcrt[idx]
    width = rect_wcrt[2] - rect_wcrt[0] + 1
    height = rect_wcrt[3] - rect_wcrt[1] + 1
    rectangle_wcrt = patches.Rectangle((rect_wcrt[0], rect_wcrt[1]), width, height, edgecolor='green', fill=False)
    ax.add_patch(rectangle_wcrt)

    rect = carseqrects[idx]
    width = rect[2] - rect[0] + 1
    height = rect[3] - rect[1] + 1
    rectangle = patches.Rectangle((rect[0], rect[1]), width, height, edgecolor='yellow', fill=False)
    ax.add_patch(rectangle)
    plt.imshow(img, cmap='gray')
    if idx in [1, 100, 200, 300, 400]:
        plt.savefig("LK_wcrt"+str(idx)+".png")
    plt.pause(0.005)
    ax.clear()

carseqrects_wcrt = np.asarray(carseqrects_wcrt)
print(carseqrects_wcrt.shape)
np.save('carseqrects-wcrt.npy', carseqrects_wcrt)

#for visualization
carseqrects_wcrt = np.load('carseqrects_wcrt.npy')
carseqrects = np.load('carseqrects.npy')
for idx in range(frames.shape[2] - 1):
    print(idx)  
    img = frames[:,:,idx].copy()

    rect_wcrt = carseqrects_wcrt[idx]
    width = rect_wcrt[2] - rect_wcrt[0] + 1
    height = rect_wcrt[3] - rect_wcrt[1] + 1
    rectangle_wcrt = patches.Rectangle((rect_wcrt[0], rect_wcrt[1]), width, height, edgecolor='green', fill=False)
    ax.add_patch(rectangle_wcrt)

    rect = carseqrects[idx]
    width = rect[2] - rect[0] + 1
    height = rect[3] - rect[1] + 1
    rectangle = patches.Rectangle((rect[0], rect[1]), width, height, edgecolor='yellow', fill=False)
    ax.add_patch(rectangle)
    plt.imshow(img, cmap='gray')
    if idx in [1, 100, 200, 300, 400]:
        plt.savefig("LK_wcrt"+str(idx)+".png")
    plt.pause(0.005)
    ax.clear()