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

h, w = img_ref.shape
center = np.array([w/2.0, h/2.0])

# Compute Euclidean distance matrix
D = distance_matrix(ref_markers, def_markers)

# Add physical prior penalty: inward motion is heavily penalized
radial_dirs = ref_markers - center
norms = np.linalg.norm(radial_dirs, axis=1, keepdims=True)
# Safe normalization
norms[norms < 1e-5] = 1.0
radial_dirs = radial_dirs / norms

# We can broadcast to compute displacements for all pairs
# disps[i, j] = def_markers[j] - ref_markers[i]
# It reshapes nicely: def_markers is (M, 2), ref_markers is (N, 2)
disps = def_markers[None, :, :] - ref_markers[:, None, :] # shape (N, M, 2)

# Dot products: shape (N, M)
# dot(disps[i, j], radial_dirs[i])
dot_prods = np.sum(disps * radial_dirs[:, None, :], axis=2)

# Penalize inward movement: any negative dot product gets a large penalty multiplier
# Increase distance cost if it points inward
inward_mask = dot_prods < -2.0 # Allow small inward noise (e.g. 2 pixels)
D[inward_mask] += np.abs(dot_prods[inward_mask]) * 5.0 # Add penalty proportional to inward distance

# Hungarian algorithm
row_ind, col_ind = linear_sum_assignment(D)

# Evaluate
final_disps = def_markers[col_ind] - ref_markers[row_ind]
mag_disps = np.linalg.norm(final_disps, axis=1)

final_dot_prods = np.sum(final_disps * radial_dirs[row_ind], axis=1)

inward_count = np.sum(final_dot_prods < -1.0)
outward_count = np.sum(final_dot_prods > 1.0)
valid_mask = mag_disps < 60.0

inward_valid = np.sum((final_dot_prods < -1.0) & valid_mask)
print(f"Radial Hungarian: inward: {inward_count} (valid: {inward_valid}), outward: {outward_count}, max_disp: {np.max(mag_disps[valid_mask]):.2f}")

