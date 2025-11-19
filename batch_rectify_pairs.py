# batch_rectify_pairs_ecc_with_metrics.py
# Same as before, but also computes alignment metrics (MSE, PSNR, corr)
# for each depth label after ECC + overlap cropping.

import os, glob
import numpy as np
import cv2
from PIL import Image

# ---------------- Config ----------------
SHARP_DIR = "phasecam4_images/sharp-refocus"
BLUR_DIR  = "phasecam4_images/phasemask"
OUT_DIR   = "aligned_pairs_phasecam4_REFOCUS"

LABELS    = [str(i) for i in range(-10, 11)]
IMG_EXTS  = (".png", ".jpg")

#ECC - used for img alignment 
ECC_ITERS = 300
ECC_EPS   = 1e-6

def ensure_dir(p):
    #creates directory if it doesn't exist
    os.makedirs(p, exist_ok=True)


#searches for an img file in the specific folder w the matching label
def find_image_path(folder, label):
    for ext in IMG_EXTS:
        p = os.path.join(folder, f"{label}{ext}")
        if os.path.isfile(p):
            return p
    hits = []
    for ext in IMG_EXTS:
        hits += glob.glob(os.path.join(folder, f"{label}*{ext}"))
    return hits[0] if hits else None

#loads an img as grayscale
    #cv.imdecode: this is used to read the img from the path and decode into grayscale

def load_u8_gray(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    #my debugs^^^
    if img is None:
       img = np.asarray(Image.open(path).convert("L"))
    return img

#converts an int8 image into a floating point img.
def to_f01(u8):
    return (u8.astype(np.float32) / 255.0).clip(0, 1)

#saves an img in the range 0 to 1 after converting it back to a 8bit format.
def save_img01(arr, path):
    x = np.clip(arr, 0.0, 1.0)
    Image.fromarray((x * 255.0 + 0.5).astype(np.uint8)).save(path)

#fn takes two input imges: I and J (both are floating pt images) and accepts two arguments of iterations (300) and eps: tolerance.
def ecc_align_homography(I01, J01, iters=ECC_ITERS, eps=ECC_EPS):
    """
    Align J to I using ECC with homography; returns (J_aligned, mask).
    I01, J01 in [0,1], float32, same initial size.
    """
    #gives height and width of img
    H, W = I01.shape
    #specifies transformation should be a homography (general transformation)
    warp_mode = cv2.MOTION_HOMOGRAPHY
    #initializes the transformation matrix as a 3x3 identity matrix
    WARP = np.eye(3, dtype=np.float32)

    #flags used by openCV ecc algorithms to specify the stopping conditions, EPS: means algorithm will stop if the transformation change is smaller than the specified eps, COUNT: means the algorithmw will stop after a fixed number of iterations
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)

    #multiply to convert to 8 bit integer format
    Iu8 = (I01 * 255).astype(np.uint8)
    Ju8 = (J01 * 255).astype(np.uint8)

    try:
        #performs an algorithm to estimate the transformation that aligns to Iu8 and Ju8, warp is the initial transformation matrix that will be updated by the algorithm, and warpmode is to use homography transformation, crit speciesi the stopping criteria for the algorithm, and 5 is the pyramid layers used in ECC algorithm
        _, WARP = cv2.findTransformECC(Iu8, Ju8, WARP, warp_mode, crit, None, 5)
    except cv2.error:
        # Fallback: affine motion
        print("FALL BACK FOR DEBUG")
        warp_mode = cv2.MOTION_AFFINE
        #switches to a 2x3 matrix - allows for simpler transformation like rotations, scaling and translation but can't handle perspective distortions
        WARP2 = np.eye(2, 3, dtype=np.float32)
        _, WARP2 = cv2.findTransformECC(Iu8, Ju8, WARP2, warp_mode, crit, None, 5)
        Jw_u8 = cv2.warpAffine(Ju8, WARP2, (W, H), flags=cv2.INTER_LINEAR)
        Jw = to_f01(Jw_u8)
        mask = np.ones_like(I01, dtype=np.uint8)
        return Jw, mask

    # Homography warp
        #if the homography alignment is successful, the img Ju8 is transformed using the homography matrix and LINEAR is bilinear interpolation
    Jw_u8 = cv2.warpPerspective(Ju8, WARP, (W, H), flags=cv2.INTER_LINEAR)
    #converted back to a floating pt format.
    Jw = to_f01(Jw_u8)

    # build a mask of valid pixels
    mask = cv2.warpPerspective(np.ones_like(Ju8, dtype=np.uint8),
                               WARP, (W, H),
                               flags=cv2.INTER_NEAREST)
    return Jw, mask

#crops I01 and J01 to where the mask indicates valid overlap
def crop_to_valid_overlap(I01, J01, mask):
    """
    Crop I and J to the bounding box of mask>0 (valid overlap region).
    """
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        # no overlap, just center-crop to common min size as worst case
        Hmin = min(I01.shape[0], J01.shape[0])
        Wmin = min(I01.shape[1], J01.shape[1])
        syI = (I01.shape[0] - Hmin) // 2
        sxI = (I01.shape[1] - Wmin) // 2
        syJ = (J01.shape[0] - Hmin) // 2
        sxJ = (J01.shape[1] - Wmin) // 2
        I_c = I01[syI:syI+Hmin, sxI:sxI+Wmin]
        J_c = J01[syJ:syJ+Hmin, sxJ:sxJ+Wmin]
        return I_c, J_c

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    I_c = I01[y0:y1, x0:x1]
    J_c = J01[y0:y1, x0:x1]
    return I_c, J_c

#adjusts the intensity of img J so that its mean intensity matches that of img I
def match_intensity(J, I):
    mI = float(I.mean())
    mJ = float(J.mean())
    if mJ < 1e-6:
        return J
    alpha = mI / mJ
    return J * alpha

#computes MSE, PSNR and correlation coefficient (to evaluate how well aligned the images are)
def compute_alignment_metrics(I_c, J_c):
    # Low-pass filter to focus on large structures (not fine blur details) and standard dev of 5 to focus on larger structures
    I_lp = cv2.GaussianBlur(I_c, (21, 21), 5)
    J_lp = cv2.GaussianBlur(J_c, (21, 21), 5)

    # Match brightness
    #J_lp's brightness is adjusted to match I_lp
    J_lp = match_intensity(J_lp, I_lp)

    # Clip to [0,1] for safety to prevent pixel values outside this range.
    I_lp = np.clip(I_lp, 0.0, 1.0)
    J_lp = np.clip(J_lp, 0.0, 1.0)

    # MSE / PSNR (max_val=1.0 because both are in [0,1])
    #MSE is computed between I lp and J lp
    #PSNR is calculated from MSE with a large PSNR value indicating better alignment, if mse is too small, the PSNR is set to infinity
    resid = I_lp - J_lp
    mse = float(np.mean(resid**2))
    if mse < 1e-12:
        psnr = float("inf")
    else:
        psnr = 10.0 * np.log10(1.0 / mse)

    # Correlation coefficient is computed by flattening both images, centering them and calculating the normalized dot product 
    v1 = I_lp.ravel().astype(np.float64)
    v2 = J_lp.ravel().astype(np.float64)
    v1 -= v1.mean()
    v2 -= v2.mean()
    denom = np.sqrt((v1 @ v1) * (v2 @ v2)) + 1e-12
    corr = float((v1 @ v2) / denom)

    return mse, psnr, corr


#loops thru a list of img labels and loads corresponding sharp and blurred imgs from the disk
#if the images aren't the same size, the blurred img is resized to match the sharp img's dimensions
def main():
    ensure_dir(OUT_DIR)
    print("ECC alignment, homography -> overlap crop + metrics\n")

    metrics_lines = []
    metrics_lines.append("label, height, width, mse_lp, psnr_lp, corr_lp\n")

    for lbl in LABELS:
        sharp_path = find_image_path(SHARP_DIR, lbl)
        blur_path  = find_image_path(BLUR_DIR,  lbl)

        if sharp_path is None or blur_path is None:
            print(f"[skip] missing pair for label {lbl}")
            continue

        Iu8 = load_u8_gray(sharp_path)
        Ju8 = load_u8_gray(blur_path)

        # Resize blur to match sharp if needed
        if Iu8.shape != Ju8.shape:
            Ju8 = cv2.resize(Ju8, (Iu8.shape[1], Iu8.shape[0]),
                             interpolation=cv2.INTER_LINEAR)

        I01 = to_f01(Iu8)
        J01 = to_f01(Ju8)

        try:
            #tries to align the imgs using ECC homography - it alignment falls, it falls back to center cropping
            Jw, mask = ecc_align_homography(I01, J01)
            I_c, J_c = crop_to_valid_overlap(I01, Jw, mask)
            print(f"[{lbl}] aligned & cropped -> shape={I_c.shape}")
        except Exception as e:
            print(f"[warn] ECC failed for label {lbl}: {e}")
            # fallback: center crop to common size
            Hmin = min(I01.shape[0], J01.shape[0])
            Wmin = min(I01.shape[1], J01.shape[1])
            syI = (I01.shape[0]-Hmin)//2; sxI = (I01.shape[1]-Wmin)//2
            syJ = (J01.shape[0]-Hmin)//2; sxJ = (J01.shape[1]-Wmin)//2
            I_c = I01[syI:syI+Hmin, sxI:sxI+Wmin]
            J_c = J01[syJ:syJ+Hmin, sxJ:sxJ+Wmin]
            print(f"[{lbl}] fallback center-crop -> shape={I_c.shape}")

        #computes the alignment metrics MSE PSNR and correlation for the correped images
        mse_lp, psnr_lp, corr_lp = compute_alignment_metrics(I_c, J_c)
        print(f"    metrics: MSE_lp={mse_lp:.4e}, PSNR_lp={psnr_lp:5.2f} dB, corr_lp={corr_lp:6.3f}")
        metrics_lines.append(f"{lbl},{I_c.shape[0]},{I_c.shape[1]},"
                             f"{mse_lp:.6e},{psnr_lp:.3f},{corr_lp:.4f}\n")

        # save aligned sharp/blur
        save_img01(I_c, os.path.join(OUT_DIR, f"{lbl}_sharp.png"))
        save_img01(J_c, os.path.join(OUT_DIR, f"{lbl}_blur.png"))

        # saves diff for sanity check
        diff = np.abs(I_c - J_c)
        if diff.max() > 0:
            diff = diff / diff.max()
        save_img01(diff, os.path.join(OUT_DIR, f"{lbl}_diff.png"))

    # write metrics file
    metrics_path = os.path.join(OUT_DIR, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        f.writelines(metrics_lines)
    print(f"\nMetrics written to {metrics_path}")


if __name__ == "__main__":
    main()


#low MSE is good
#PSNR a higher is better like 40 is good
#correlation: +1 is good, 0 is no linear correlation, -1 is perfect - correlation
    # u want 1.

#This script processes pairs of sharp and blurred images by aligning them using ECC, cropping them to the valid overlapping region, computing quality metrics (MSE, PSNR, and correlation), and saving the results for further analysis. It handles cases where alignment fails by falling back to center cropping and provides an automated workflow for evaluating image alignment quality.