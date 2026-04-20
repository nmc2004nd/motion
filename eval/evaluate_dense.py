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

# Farneback
flow = cv2.calcOpticalFlowFarneback(ref_proc, def_proc, None, 0.5, 3, 15, 3, 5, 1.2, 0)
flow_pts = []
for pt in ref_markers:
    x, y = int(pt[0]), int(pt[1])
    flow_pts.append(pt + flow[y, x])
flow_pts = np.array(flow_pts)

# evaluate
disps = flow_pts - ref_markers
radial_dirs = ref_markers - center
norms = np.linalg.norm(radial_dirs, axis=1, keepdims=True)
radial_dirs = radial_dirs / (norms + 1e-5)

dot_prods = np.sum(disps * radial_dirs, axis=1)
mag_disps = np.linalg.norm(disps, axis=1)

inward_count = np.sum((dot_prods < -0.5) & (mag_disps > 1.0))
print(f"Farneback inward_count: {inward_count}")
