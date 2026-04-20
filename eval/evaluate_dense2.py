import cv2
import numpy as np
from src.preprocessing import preprocess
from src.detection import detect_markers
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

img_ref = cv2.imread("data/ref/my_photo_1.jpg", cv2.IMREAD_GRAYSCALE)
img_def = cv2.imread("data/img/my_photo_2.jpg", cv2.IMREAD_GRAYSCALE)

ref_proc = preprocess(img_ref)
def_proc = preprocess(img_def)
ref_markers, _ = detect_markers(ref_proc)
def_markers, _ = detect_markers(def_proc)

h, w = img_ref.shape

# Blur images heavily to track the envelope
ref_blur = cv2.GaussianBlur(ref_proc, (61, 61), 15)
def_blur = cv2.GaussianBlur(def_proc, (61, 61), 15)

flow = cv2.calcOpticalFlowFarneback(ref_blur, def_blur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
flow_pts = []
for pt in ref_markers:
    x, y = int(pt[0]), int(pt[1])
    flow_pts.append(pt + flow[y, x])
flow_pts = np.array(flow_pts)

# Nearest def marker to the flow-shifted ref marker!
D = distance_matrix(flow_pts, def_markers)
row_ind, col_ind = linear_sum_assignment(D)

disps = def_markers[col_ind] - ref_markers[row_ind]
mag_disps = np.linalg.norm(disps, axis=1)

center = np.array([w/2.0, h/2.0])
radial_dirs = ref_markers[row_ind] - center
norms = np.linalg.norm(radial_dirs, axis=1, keepdims=True)
radial_dirs = radial_dirs / (norms + 1e-5)
dot_prods = np.sum(disps * radial_dirs, axis=1)

inward_count = np.sum((dot_prods < -0.5) & (mag_disps > 1.0))
print(f"Farneback+Blur then Hungarian inward_count: {inward_count}")

import json
print(json.dumps({'max_disp': float(np.max(mag_disps))}))
