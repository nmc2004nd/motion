import cv2
import numpy as np
from src.preprocessing import preprocess
from src.detection import detect_markers
from src.tracking import track_markers_lk
from src.visualization import visualize_flow_arrows

img_ref = cv2.imread("data/ref/my_photo_1.jpg", cv2.IMREAD_GRAYSCALE)
img_def = cv2.imread("data/img/my_photo_2.jpg", cv2.IMREAD_GRAYSCALE)

ref_proc = preprocess(img_ref)
def_proc = preprocess(img_def)
ref_markers, _ = detect_markers(ref_proc)

def test_params(win, level):
    pts_ref = ref_markers.reshape(-1, 1, 2).astype(np.float32)
    lk_params = dict(winSize=(win, win), maxLevel=level, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    pts_def, status, _ = cv2.calcOpticalFlowPyrLK(ref_proc, def_proc, pts_ref, None, **lk_params)
    pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(def_proc, ref_proc, pts_def, None, **lk_params)
    fb_error = np.linalg.norm(pts_back.reshape(-1, 2) - ref_markers, axis=1)
    valid = (status.flatten() == 1) & (status_back.flatten() == 1) & (fb_error < 2.0)
    
    vis = visualize_flow_arrows(def_proc, ref_markers, pts_def.reshape(-1, 2), valid, scale=1.0, save_path=f"test_arrow_{win}_{level}.png")
    print(f"win={win}, level={level}, valid={valid.sum()}")

test_params(21, 3)
test_params(31, 3)
test_params(41, 3)
test_params(21, 4)
test_params(31, 4)
test_params(31, 2)
