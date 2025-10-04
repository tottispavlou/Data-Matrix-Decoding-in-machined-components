import cv2
import numpy as np
import os
from sklearn.linear_model import RANSACRegressor

def detect_dots(img_gray):
    """Detect circular blobs (dot peen marks)."""
    blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 200
    params.filterByCircularity = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(bw)

    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return pts


def fit_line_ransac(points):
    """Fit a robust line using sklearn RANSAC."""
    if len(points) < 2:
        return None, None

    X = points[:,0].reshape(-1,1)   # x
    y = points[:,1]                 # y

    ransac = RANSACRegressor(min_samples=2, residual_threshold=3, max_trials=200)
    ransac.fit(X, y)

    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_

    return slope, intercept


def refine_grid_with_ransac(crop):
    """Deskew crop using RANSAC-fitted lines."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    pts = detect_dots(gray)

    if len(pts) < 10:
        print("[WARN] Too few points detected.")
        return crop

    # Horizontal fit
    slope_h, _ = fit_line_ransac(pts)

    # Vertical fit (swap x/y)
    slope_v, _ = fit_line_ransac(pts[:,[1,0]])

    if slope_h is None or slope_v is None:
        print("[WARN] RANSAC failed")
        return crop

    angle_h = np.degrees(np.arctan(slope_h))
    angle_v = -90 + np.degrees(np.arctan(slope_v))

    angle = (angle_h + angle_v) / 2.0

    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    rectified = cv2.warpAffine(crop, M, (w,h), flags=cv2.INTER_LINEAR)

    return rectified


def read_yolo_obb(label_path, img_w, img_h):
    """Parse YOLO OBB label into quad coords."""
    with open(label_path, "r") as f:
        line = f.readline().strip().split()
    vals = list(map(float, line[1:]))
    pts = np.array(vals).reshape(-1,2)
    pts[:,0] *= img_w
    pts[:,1] *= img_h
    return pts.astype(np.float32)


def four_point_crop(img, pts, size=256):
    """Warp quad → rectified square crop."""
    dst = np.array([[0,0],[size,0],[size,size],[0,size]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (size,size))
    return warped


if __name__ == "__main__":
    img_path = "data/rectified_crops/4B9U53603750002_jpeg.rf.1ca40283e33fb52f2775da697fdeb6ce_det0.png"
    label_path = "runs/obb/predict/labels/example.txt"

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    quad = read_yolo_obb(label_path, w, h)
    crop = four_point_crop(img, quad, size=256)

    rectified = refine_grid_with_ransac(crop)

    cv2.imwrite("rectified_ransac.jpg", rectified)
    print("[OK] Saved rectified crop -> rectified_ransac.jpg")
