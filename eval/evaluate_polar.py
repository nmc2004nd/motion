import cv2
import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from src.preprocessing import preprocess
from src.detection import detect_markers
from src.visualization import visualize_flow_arrows

img_ref = cv2.imread("data/ref/my_photo_1.jpg", cv2.IMREAD_GRAYSCALE)
img_def = cv2.imread("data/img/my_photo_2.jpg", cv2.IMREAD_GRAYSCALE)

ref_proc = preprocess(img_ref)
def_proc = preprocess(img_def)
ref_markers, _ = detect_markers(ref_proc)
def_markers, _ = detect_markers(def_proc)

center_ref = np.mean(ref_markers, axis=0)
center_def = np.mean(def_markers, axis=0)

def to_polar(pts, c):
    d = pts - c
    r = np.linalg.norm(d, axis=1)
    theta = np.arctan2(d[:, 1], d[:, 0])
    return r, theta

r_ref, t_ref = to_polar(ref_markers, center_ref)
r_def, t_def = to_polar(def_markers, center_def)

# Distance metric: differences in r and theta
# We expect r_def > r_ref typically, but maybe not. The angles should be very close!
# Because the expansion is radial.
# Distance = w_theta * angular_dist + w_r * radial_dist
# For angular distance, handle periodicity:
D_theta = np.abs(t_ref[:, None] - t_def[None, :])
D_theta = np.minimum(D_theta, 2*np.pi - D_theta)

D_r = np.abs(r_ref[:, None] - r_def[None, :])

D = D_theta * 100.0 + D_r * 1.0  # High weight on angle!

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
print(f"Polar Hungarian inward_count: {inward_count}")

