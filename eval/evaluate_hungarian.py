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

D = distance_matrix(ref_markers, def_markers)
row_ind, col_ind = linear_sum_assignment(D)

pts_def = def_markers[col_ind]
pts_ref = ref_markers[row_ind]
valid = np.ones(len(pts_ref), dtype=bool)

# Render
visualize_flow_arrows(def_proc, pts_ref, pts_def, valid, scale=3.0, save_path="test_hungarian.png")

