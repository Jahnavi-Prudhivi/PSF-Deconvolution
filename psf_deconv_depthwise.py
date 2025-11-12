# align_and_crop_checkerboard_wide.py
# Step 1 only: align blurred -> sharp using checkerboard homography (ECC fallback),
# then crop BOTH to a WIDE ROI that includes: left checkerboard + center noise + right checkerboard.
#
# Input folders:
#   images/wo-phasemask/   -10.png ... 0.png ... 10.png   (sharp)
#   images/phasemask/      -10.png ... 0.png ... 10.png   (blurred)
#
# Output folder:
#   outputs/aligned_rois/
#     {label}_sharp_roi.png
#     {label}_blur_roi.png
#     {label}_diff.png       (normalized |sharp - blur|)
#
# Requirements: numpy, opencv-python, pillow

import os, glob
import numpy as np
import cv2
from PIL import Image

# ----------------------------- Config -----------------------------
SHARP_DIR = "images/wo-phasemask"
BLUR_DIR  = "images/phasemask"
OUT_DIR   = "outputs/aligned_rois"
LABELS    = [str(i) for i in range(-10, 11)]  # -10..10

# ⚠️ SET THIS to your checkerboard inner corners (cols, rows)!
# e.g., board with 10x7 squares => inner corners (9,6)
PATTERN_SIZE = (9, 6)

# Wide crop padding around the checkerboard bounding box (fractions of w/h)
PAD_X_FRAC = 0.60   # 60% of board width to left/right
PAD_Y_FRAC = 0.40   # 40% of board height to top/bottom
# You can instead hardcode pixels, e.g. PAD_X_PIX=220 / PAD_Y_PIX=160; set *_FRAC=None to use pixels.

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ----------------------------- IO helpers -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def find_image_path(folder: str, label: str):
    for ext in IMG_EXTS:
        p = os.path.join(folder, f"{label}{ext}")
        if os.path.isfile(p): return p
    hits = []
    for ext in IMG_EXTS:
        hits += glob.glob(os.path.join(folder, f"{label}*{ext}"))
    return hits[0] if hits else None

def load_u8_gray(path: str) -> np.ndarray:
    """Load as uint8 grayscale (OpenCV-friendly)."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.asarray(Image.open(path).convert("L"))
    return img

def to_f01(u8: np.ndarray) -> np.ndarray:
    return u8.astype(np.float32) / 255.0

def save_img01(arr: np.ndarray, path: str):
    x = np.clip(arr, 0.0, 1.0)
    Image.fromarray((x * 255.0 + 0.5).astype(np.uint8)).save(path)

# ------------------------ Alignment utils -----------------------------
def find_corners_gray(u8: np.ndarray, pattern_size):
    """Find and refine inner corners; returns (N,2) or None."""
    # For tough images, try cv2.findChessboardCornersSB(u8, pattern_size)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(u8, pattern_size, flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    corners = cv2.cornerSubPix(u8, corners, (5,5), (-1,-1), crit)
    return corners.reshape(-1, 2)  # (N,2)

def ecc_align(I: np.ndarray, J: np.ndarray, use_homography=True, iters=300, eps=1e-6):
    """Align J to I via ECC; returns warp and warped image in [0,1]."""
    H, W = I.shape
    if use_homography:
        warp_mode = cv2.MOTION_HOMOGRAPHY
        WARP = np.eye(3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
        try:
            _, WARP = cv2.findTransformECC(I, J, WARP, warp_mode, criteria, None, 5)
            Jw = cv2.warpPerspective((J*255).astype(np.uint8), WARP, (W, H), flags=cv2.INTER_LINEAR)
            return WARP, to_f01(Jw)
        except cv2.error:
            # fall back to affine
            warp_mode = cv2.MOTION_AFFINE
            WARP = np.eye(2,3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
            _, WARP = cv2.findTransformECC(I, J, WARP, warp_mode, criteria, None, 5)
            Jw = cv2.warpAffine((J*255).astype(np.uint8), WARP, (W, H), flags=cv2.INTER_LINEAR)
            return None, to_f01(Jw)
    else:
        warp_mode = cv2.MOTION_AFFINE
        WARP = np.eye(2,3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
        _, WARP = cv2.findTransformECC(I, J, WARP, warp_mode, criteria, None, 5)
        Jw = cv2.warpAffine((J*255).astype(np.uint8), WARP, (W, H), flags=cv2.INTER_LINEAR)
        return None, to_f01(Jw)

def wide_checkerboard_crop_bounds(corners_xy: np.ndarray, img_shape, pad_x_frac, pad_y_frac,
                                  pad_x_pix=None, pad_y_pix=None):
    """Compute a wide crop rectangle around checkerboard with generous margins."""
    Hh, Ww = img_shape
    x, y, w, h = cv2.boundingRect(corners_xy.astype(np.float32))
    if pad_x_frac is not None and pad_y_frac is not None:
        pad_x = int(pad_x_frac * w)
        pad_y = int(pad_y_frac * h)
    else:
        pad_x = int(pad_x_pix or 0)
        pad_y = int(pad_y_pix or 0)

    x0 = max(x - pad_x, 0)
    y0 = max(y - pad_y, 0)
    x1 = min(x + w + pad_x, Ww)
    y1 = min(y + h + pad_y, Hh)
    # Ensure at least some area:
    if x1 <= x0 + 10 or y1 <= y0 + 10:
        raise RuntimeError("Wide crop collapsed; check padding or corner detection.")
    return x0, y0, x1, y1

def homography_align_and_crop_wide(Iu8: np.ndarray, Ju8: np.ndarray,
                                   pattern_size,
                                   pad_x_frac=0.6, pad_y_frac=0.4,
                                   pad_x_pix=None, pad_y_pix=None):
    """
    Align J->I via checkerboard homography if possible; else ECC.
    Then crop BOTH to a WIDE ROI (checkerboard + noise + checkerboard).
    Returns I_roi_f01, J_roi_f01, method_str.
    """
    Ic = find_corners_gray(Iu8, pattern_size)
    Jc = find_corners_gray(Ju8, pattern_size)

    if Ic is not None and Jc is not None and Ic.shape == Jc.shape:
        # Homography J -> I
        H, _ = cv2.findHomography(Jc, Ic, method=cv2.RANSAC, ransacReprojThreshold=1.5)
        Hh, Ww = Iu8.shape
        Jw = cv2.warpPerspective(Ju8, H, (Ww, Hh), flags=cv2.INTER_LINEAR)

        # Wide ROI from sharp corners
        x0, y0, x1, y1 = wide_checkerboard_crop_bounds(
            Ic, (Hh, Ww), pad_x_frac, pad_y_frac, pad_x_pix, pad_y_pix
        )
        I_roi = Iu8[y0:y1, x0:x1]
        J_roi = Jw [y0:y1, x0:x1]
        return to_f01(I_roi), to_f01(J_roi), f"homography corners={Ic.shape[0]}"
    else:
        # Fallback: ECC (homography first, else affine)
        I = to_f01(Iu8); J = to_f01(Ju8)
        H, Jw = ecc_align(I, J, use_homography=True)

        # Build overlap mask to crop a wide, valid region (no exact corners here)
        Hh, Ww = Iu8.shape
        if H is not None and H.shape == (3,3):
            mask = cv2.warpPerspective(np.ones_like(J), H, (Ww, Hh), flags=cv2.INTER_NEAREST)
        else:
            # Affine overlap (approximate with a border to avoid empty edges)
            ones = (np.ones_like(J)*255).astype(np.uint8)
            mask = (ones > 0.5).astype(np.float32)

        ys, xs = np.where(mask > 0.5)
        if ys.size == 0:
            raise RuntimeError("ECC produced empty overlap; check inputs.")
        # Tight overlap box, then shrink slightly to avoid boundaries
        y0, y1 = ys.min(), ys.max()+1
        x0, x1 = xs.min(), xs.max()+1
        margin_y = int(0.02*(y1 - y0))
        margin_x = int(0.02*(x1 - x0))
        y0 = max(y0 + margin_y, 0); y1 = min(y1 - margin_y, Hh)
        x0 = max(x0 + margin_x, 0); x1 = min(x1 - margin_x, Ww)

        I_roi = I[y0:y1, x0:x1]
        J_roi = Jw[y0:y1, x0:x1]
        return I_roi, J_roi, "ECC"

# ----------------------------- Driver ---------------------------------
def main():
    ensure_dir(OUT_DIR)
    print(f"Using checkerboard inner corners: {PATTERN_SIZE}")
    print(f"Wide padding: PAD_X_FRAC={PAD_X_FRAC}, PAD_Y_FRAC={PAD_Y_FRAC}")

    for lbl in LABELS:
        sharp_path = find_image_path(SHARP_DIR, lbl)
        blur_path  = find_image_path(BLUR_DIR,  lbl)
        if sharp_path is None or blur_path is None:
            print(f"[skip] missing pair for label {lbl}")
            continue

        Iu8 = load_u8_gray(sharp_path)
        Ju8 = load_u8_gray(blur_path)

        try:
            I_roi, J_roi, how = homography_align_and_crop_wide(
                Iu8, Ju8, PATTERN_SIZE,
                pad_x_frac=PAD_X_FRAC, pad_y_frac=PAD_Y_FRAC
                # or: pad_x_frac=None, pad_y_frac=None, pad_x_pix=220, pad_y_pix=160
            )
            print(f"[{lbl}] aligned via {how}, roi={I_roi.shape}")
        except Exception as e:
            print(f"[warn] alignment failed for {lbl}: {e}")
            continue

        # Save aligned wide ROIs
        out_I = os.path.join(OUT_DIR, f"{lbl}_sharp_roi.png")
        out_J = os.path.join(OUT_DIR, f"{lbl}_blur_roi.png")
        save_img01(I_roi, out_I)
        save_img01(J_roi, out_J)

        # Difference image (normalized abs diff)
        diff = np.abs(I_roi - J_roi)
        if diff.max() > 0: diff = diff / diff.max()
        out_D = os.path.join(OUT_DIR, f"{lbl}_diff.png")
        save_img01(diff, out_D)

if __name__ == "__main__":
    main()
