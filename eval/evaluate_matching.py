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

D = distance_matrix(ref_markers, def_markers)
row_ind, col_ind = linear_sum_assignment(D)

disps = def_markers[col_ind] - ref_markers[row_ind]
mag_disps = np.linalg.norm(disps, axis=1)
print(f"Mean disp: {np.mean(mag_disps):.2f}, Max disp: {np.max(mag_disps):.2f}")

h, w = img_ref.shape
center = np.array([w/2.0, h/2.0])
radial_dirs = ref_markers[row_ind] - center
norms = np.linalg.norm(radial_dirs, axis=1, keepdims=True)
radial_dirs = radial_dirs / (norms + 1e-5)

dot_prods = np.sum(disps * radial_dirs, axis=1)

inward_count = np.sum((dot_prods < -0.5) & (mag_disps > 1.0))
print(f"nearest neighbor Hungarian inward_count: {inward_count}")

# Regularized LK test:
pts_ref = ref_markers.reshape(-1, 1, 2).astype(np.float32)
# we can use the def_markers[col_ind] to initialize optical flow!
pts_init = def_markers[col_ind].reshape(-1, 1, 2).astype(np.float32)
lk_params = dict(winSize=(15, 15), maxLevel=0, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01), flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
pts_def, status, _ = cv2.calcOpticalFlowPyrLK(ref_proc, def_proc, pts_ref, pts_init, **lk_params)

pts_def_valid = pts_def.reshape(-1, 2)
disps2 = pts_def_valid - ref_markers
dot_prods2 = np.sum(disps2 * radial_dirs, axis=1)
mag_disps2 = np.linalg.norm(disps2, axis=1)
inward_count2 = np.sum((dot_prods2 < -0.5) & (mag_disps2 > 1.0))
print(f"LK with Initial Flow inward_count: {inward_count2}")

