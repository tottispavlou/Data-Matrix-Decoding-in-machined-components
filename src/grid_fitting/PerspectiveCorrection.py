import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def perspective_warp(image: np.ndarray, transform: np.ndarray) -> Tuple[np.ndarray, int, int]:
        h, w = image.shape[:2]
        corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        print(corners_bef)
        corners_aft = cv2.perspectiveTransform(corners_bef, transform)
        print(corners_aft)
        xmin = math.floor(corners_aft[:, 0, 0].min())
        ymin = math.floor(corners_aft[:, 0, 1].min())
        xmax = math.ceil(corners_aft[:, 0, 0].max())
        ymax = math.ceil(corners_aft[:, 0, 1].max())
        x_adj = math.floor(xmin - corners_aft[0, 0, 0])
        y_adj = math.floor(ymin - corners_aft[0, 0, 1])
        translate = np.eye(3)
        print(translate)
        translate[0, 2] = -xmin
        translate[1, 2] = -ymin
        corrected_transform = np.matmul(translate, transform)
        return cv2.warpPerspective(image, corrected_transform, (math.ceil(xmax - xmin), math.ceil(ymax - ymin))), x_adj, y_adj

# Just like perspective_warp, but it also returns an alpha mask that can be used for blitting
def perspective_warp_with_mask(image: np.ndarray, transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    mask_in = np.empty(image.shape, dtype = np.uint8)
    mask_in.fill(255)
    output, x_adj, y_adj = perspective_warp(image, transform)
    mask, _, _ = perspective_warp(mask_in, transform)
    return output, mask, x_adj, y_adj

# alpha_blits src onto dest according to the alpha values in mask at location (x, y),
# ignoring any parts that do not overlap
def alpha_blit(dest: np.ndarray, src: np.ndarray, mask: np.ndarray, x: int, y: int) -> None:
    dl = max(x, 0)
    dt = max(y, 0)
    sl = max(-x, 0)
    st = max(-y, 0)
    sr = max(sl, min(src.shape[1], dest.shape[1] - x))
    sb = max(st, min(src.shape[0], dest.shape[0] - y))
    dr = dl + sr - sl
    db = dt + sb - st
    m = mask[st:sb, sl:sr]
    dest[dt:db, dl:dr] = (dest[dt:db, dl:dr].astype(np.float) * (255 - m) + src[st:sb, sl:sr].astype(np.float) * m) / 255

# blits a perspective-warped src image onto dest
def perspective_blit(dest: np.ndarray, src: np.ndarray, transform: np.ndarray) -> None:
    blitme, mask, x_adj, y_adj = perspective_warp_with_mask(src, transform)
    cv2.imwrite("blitme.png", blitme)
    alpha_blit(dest, blitme, mask, int(transform[0, 2] + x_adj), int(transform[1, 2] + y_adj))

img = cv2.imread(r"C:\Users\User\Desktop\1G9V68937920005_jpeg.rf.67d179c96046f19af6c0cd524567babe.jpg")
rows,cols,ch = img.shape

pts1 = np.float32([[234,150],[355,155],[406, 551],[285,549]])

ratio=1
cardH=math.sqrt((pts1[2][0]-pts1[1][0])*(pts1[2][0]-pts1[1][0])+(pts1[2][1]-pts1[1][1])*(pts1[2][1]-pts1[1][1]))
cardW=ratio*cardH
pts2 = np.float32([[pts1[0][0],pts1[0][1]], [pts1[0][0]+cardW, pts1[0][1]], [pts1[0][0]+cardW, pts1[0][1]+cardH], [pts1[0][0], pts1[0][1]+cardH]])

M = cv2.getPerspectiveTransform(pts1,pts2)

offsetSize=500
transformed = np.zeros((int(cardW+offsetSize), int(cardH+offsetSize)), dtype=np.uint8)
dst = cv2.warpPerspective(img, M, transformed.shape)
cv2.imwrite("image.jpg", dst)

# perspective_blit(img, img, M)

# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()
