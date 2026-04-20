import cv2
import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from src.preprocessing import preprocess
from src.detection import detect_markers

img_ref = cv2.imread("data/ref/my_photo_1.jpg", cv2.IMREAD_GRAYSCALE)
img_def = cv2.imread("data/img/my_photo_2.jpg", cv2.IMREAD_GRAYSCALE)

ref_proc = preprocess(img_ref)
def_proc = preprocess(img_def)
ref_markers, _ = detect_markers(ref_proc)
def_markers, _ = detect_markers(def_proc)

D = distance_matrix(ref_markers, def_markers)
row_ind, col_ind = linear_sum_assignment(D)

disps = def_markers[col_ind] - ref_markers[row_ind]
mag_disps = np.linalg.norm(disps, axis=1)

h, w = img_ref.shape
center = np.array([w/2.0, h/2.0])
radial_dirs = ref_markers[row_ind] - center
norms = np.linalg.norm(radial_dirs, axis=1, keepdims=True)
radial_dirs = radial_dirs / (norms + 1e-5)
dot_prods = np.sum(disps * radial_dirs, axis=1)

inward_count = np.sum(dot_prods < -1.0)
outward_count = np.sum(dot_prods > 1.0)
print(f"inward: {inward_count}, outward: {outward_count}")

import json
print(json.dumps(dict(
    mean_disp=float(np.mean(mag_disps)),
    max_disp=float(np.max(mag_disps)),
)))
