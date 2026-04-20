import cv2
import numpy as np
from src.preprocessing import preprocess
from src.detection import detect_markers

img_ref = cv2.imread("data/ref/my_photo_1.jpg", cv2.IMREAD_GRAYSCALE)
img_def = cv2.imread("data/img/my_photo_2.jpg", cv2.IMREAD_GRAYSCALE)

ref_proc = preprocess(img_ref)
def_proc = preprocess(img_def)
ref_markers, _ = detect_markers(ref_proc)

h, w = img_ref.shape
center = np.array([w/2.0, h/2.0])

def evaluate(win, level):
    pts_ref = ref_markers.reshape(-1, 1, 2).astype(np.float32)
    lk_params = dict(winSize=(win, win), maxLevel=level, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    pts_def, status, _ = cv2.calcOpticalFlowPyrLK(ref_proc, def_proc, pts_ref, None, **lk_params)
    pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(def_proc, ref_proc, pts_def, None, **lk_params)
    fb_error = np.linalg.norm(pts_back.reshape(-1, 2) - ref_markers, axis=1)
    valid = (status.flatten() == 1) & (status_back.flatten() == 1) & (fb_error < 2.0)
    
    pts_ref_valid = pts_ref.reshape(-1, 2)[valid]
    pts_def_valid = pts_def.reshape(-1, 2)[valid]
    
    disps = pts_def_valid - pts_ref_valid
    radial_dirs = pts_ref_valid - center
    norms = np.linalg.norm(radial_dirs, axis=1, keepdims=True)
    radial_dirs = radial_dirs / (norms + 1e-5)
    
    dot_prods = np.sum(disps * radial_dirs, axis=1)
    mag_disps = np.linalg.norm(disps, axis=1)
    
    inward_count = np.sum((dot_prods < -0.5) & (mag_disps > 1.0))
    print(f"win={win:2d}, level={level}, valid={valid.sum():3d}, inward_count={inward_count:3d}")

for win in [15, 21, 31, 41, 51, 61]:
    for level in [0, 1, 2, 3]:
        evaluate(win, level)

