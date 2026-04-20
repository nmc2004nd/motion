"""
Tactile sensor marker tracking pipeline.
Fixes issues with uneven lighting, merged markers, and mismatched correspondences.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pathlib import Path

# =========================================================
# STEP 1: PREPROCESSING — fix uneven lighting (vignetting)
# =========================================================
def preprocess(img_gray):
    """
    Normalize illumination using background subtraction.
    The LED gives uneven lighting (bright center, dark edges).
    We estimate the background and subtract it.
    """
    # Large Gaussian blur approximates the background illumination
    background = cv2.GaussianBlur(img_gray, (101, 101), 0)
    # Subtract and normalize: markers (bright) remain, illumination bias is removed
    normalized = cv2.subtract(img_gray, background)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    # CLAHE for local contrast enhancement (helps in low-contrast regions)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized.astype(np.uint8))
    return enhanced

# =========================================================
# STEP 2: BLOB DETECTION with SimpleBlobDetector
# SimpleBlobDetector is much more robust than threshold+centroid
# because it filters by area, circularity, inertia, convexity.
# =========================================================
def create_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    # Markers are WHITE on DARK background -> invert threshold
    params.minThreshold = 50
    params.maxThreshold = 220
    params.thresholdStep = 10
    # Detect bright blobs
    params.filterByColor = True
    params.blobColor = 255
    # Area filter (tune to your marker size in pixels)
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 500
    # Circularity (markers are ~round)
    params.filterByCircularity = True
    params.minCircularity = 0.5
    # Inertia (how elongated - 1.0 = circle)
    params.filterByInertia = True
    params.minInertiaRatio = 0.3
    # Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.7
    return cv2.SimpleBlobDetector_create(params)

def detect_markers(img_processed):
    """Returns Nx2 array of marker (x, y) centers as subpixel floats."""
    detector = create_blob_detector()
    keypoints = detector.detect(img_processed)
    centers = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    return centers, keypoints

# =========================================================
# STEP 3: BUILD CORRESPONDENCE via optical flow
# Instead of re-detecting in frame 2 and hoping the order matches,
# we TRACK each reference marker using Lucas-Kanade optical flow.
# This guarantees a 1-to-1 correspondence.
# =========================================================
def track_markers_LK(img_ref, img_def, ref_points):
    """
    Track ref_points from img_ref to img_def using Lucas-Kanade.
    Returns tracked points and a validity mask.
    """
    lk_params = dict(
        winSize=(21, 21),       # Large enough to capture marker motion
        maxLevel=3,              # Multi-scale pyramid (handles larger motion)
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    pts_ref = ref_points.reshape(-1, 1, 2).astype(np.float32)
    pts_def, status, err = cv2.calcOpticalFlowPyrLK(
        img_ref, img_def, pts_ref, None, **lk_params
    )
    # Forward-backward check for robustness
    pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
        img_def, img_ref, pts_def, None, **lk_params
    )
    fb_error = np.linalg.norm(pts_back.reshape(-1, 2) - ref_points, axis=1)
    valid = (status.flatten() == 1) & (status_back.flatten() == 1) & (fb_error < 2.0)
    return pts_def.reshape(-1, 2), valid

# =========================================================
# STEP 4: VISUALIZE the flow field nicely
# =========================================================
def visualize_flow_arrows(img, ref_pts, def_pts, valid, scale=3.0, save_path=None):
    """Draw arrows from ref to deformed position (scaled)."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    for i in range(len(ref_pts)):
        if not valid[i]:
            continue
        p0 = ref_pts[i]
        p1 = def_pts[i]
        # Scale the displacement for visibility
        disp = p1 - p0
        mag = np.linalg.norm(disp)
        if mag < 0.3:  # Skip negligible motion
            cv2.circle(vis, tuple(p0.astype(int)), 2, (180, 180, 180), -1)
            continue
        end = (p0 + disp * scale).astype(int)
        start = p0.astype(int)
        # Color by magnitude (red = high)
        color_intensity = min(int(mag * 20), 255)
        color = (0, 255 - color_intensity, color_intensity)  # BGR: green->red
        cv2.arrowedLine(vis, tuple(start), tuple(end), color, 1, tipLength=0.3)
        cv2.circle(vis, tuple(start), 2, (255, 255, 0), -1)
    if save_path:
        cv2.imwrite(save_path, vis)
    return vis

def visualize_flow_hsv(ref_pts, def_pts, valid, img_shape, save_path=None):
    """
    HSV visualization like in your target image 3.
    Hue = direction, Saturation = magnitude.
    Produces dense flow field by interpolating from sparse markers.
    """
    h, w = img_shape[:2]
    disp = def_pts - ref_pts
    valid_pts = ref_pts[valid]
    valid_disp = disp[valid]

    # Create dense grid
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Inverse-distance weighted interpolation from sparse markers
    from scipy.spatial import cKDTree
    tree = cKDTree(valid_pts)
    k = 6
    dists, idxs = tree.query(grid_pts, k=k)
    weights = 1.0 / (dists + 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    dx_dense = (valid_disp[idxs, 0] * weights).sum(axis=1).reshape(h, w)
    dy_dense = (valid_disp[idxs, 1] * weights).sum(axis=1).reshape(h, w)

    # Convert to HSV
    mag, ang = cv2.cartToPolar(dx_dense, dy_dense, angleInDegrees=True)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)  # OpenCV hue is 0-180
    hsv[..., 1] = np.clip(mag * 25, 0, 255).astype(np.uint8)
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Overlay arrows on sparse markers
    for i in range(len(ref_pts)):
        if not valid[i]:
            continue
        p0 = ref_pts[i].astype(int)
        p1 = (ref_pts[i] + (def_pts[i] - ref_pts[i]) * 3).astype(int)
        cv2.arrowedLine(bgr, tuple(p0), tuple(p1), (0, 0, 0), 1, tipLength=0.3)
        cv2.circle(bgr, tuple(p0), 1, (0, 0, 0), -1)

    if save_path:
        cv2.imwrite(save_path, bgr)
    return bgr


# =========================================================
# MAIN PIPELINE
# =========================================================
def main():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    img_ref = cv2.imread('data/ref/my_photo_1.jpg', cv2.IMREAD_GRAYSCALE)
    img_def = cv2.imread('data/img/my_photo_2.jpg', cv2.IMREAD_GRAYSCALE)

    # Step 1: Preprocess
    ref_proc = preprocess(img_ref)
    def_proc = preprocess(img_def)
    cv2.imwrite(str(output_dir / '02_ref_preprocessed.png'), ref_proc)
    cv2.imwrite(str(output_dir / '02_def_preprocessed.png'), def_proc)
    print("[1] Preprocessing done")

    # Step 2: Detect markers in reference
    ref_markers, ref_kps = detect_markers(ref_proc)
    print(f"[2] Detected {len(ref_markers)} markers in reference")
    vis_ref = cv2.drawKeypoints(ref_proc, ref_kps, None, (0, 255, 0),
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(str(output_dir / '03_ref_markers.png'), vis_ref)

    # Also show what naive detection gives on deformed frame
    def_markers_naive, def_kps_naive = detect_markers(def_proc)
    print(f"    Naive detection on deformed: {len(def_markers_naive)} markers "
          f"(count mismatch = correspondence issue!)")
    vis_def_naive = cv2.drawKeypoints(def_proc, def_kps_naive, None, (0, 255, 0),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(str(output_dir / '03_def_markers_naive.png'), vis_def_naive)

    # Step 3: Track ref markers into deformed frame (GUARANTEED correspondence)
    def_markers_tracked, valid = track_markers_LK(ref_proc, def_proc, ref_markers)
    print(f"[3] Tracked {valid.sum()}/{len(ref_markers)} markers successfully")

    # Step 4: Displacement statistics
    displacements = def_markers_tracked - ref_markers
    mags = np.linalg.norm(displacements[valid], axis=1)
    print(f"[4] Displacement: mean={mags.mean():.2f}px, max={mags.max():.2f}px, "
          f"min={mags.min():.2f}px")

    # Step 5: Visualize
    vis_arrows = visualize_flow_arrows(def_proc, ref_markers, def_markers_tracked,
                                        valid, scale=3.0,
                                save_path=str(output_dir / '04_flow_arrows.png'))
    vis_hsv = visualize_flow_hsv(ref_markers, def_markers_tracked, valid,
                                  img_ref.shape,
                            save_path=str(output_dir / '05_flow_hsv.png'))
    print("[5] Visualizations saved")

    # Composite summary figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0, 0].imshow(img_ref, cmap='gray'); axes[0, 0].set_title('1. Raw reference')
    axes[0, 1].imshow(img_def, cmap='gray'); axes[0, 1].set_title('2. Raw deformed')
    axes[0, 2].imshow(ref_proc, cmap='gray'); axes[0, 2].set_title('3. After preprocessing (lighting fixed)')
    axes[1, 0].imshow(cv2.cvtColor(vis_ref, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'4. Detected markers in ref: {len(ref_markers)}')
    axes[1, 1].imshow(cv2.cvtColor(vis_arrows, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('5. Flow arrows (displacement ×3)')
    axes[1, 2].imshow(cv2.cvtColor(vis_hsv, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('6. Dense HSV flow (like your target)')
    for a in axes.ravel():
        a.axis('off')
    plt.tight_layout()
    plt.savefig(str(output_dir / '06_summary.png'), dpi=100, bbox_inches='tight')
    print("[6] Summary figure saved")

if __name__ == "__main__":
    main()
