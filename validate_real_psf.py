import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# ---------- FFT "same" convolution (same as in auto.py) ----------

def conv_fft_same(I: np.ndarray, p: np.ndarray) -> np.ndarray:
    H, W = I.shape
    h, w = p.shape
    padH, padW = H + h - 1, W + w - 1

    FI = np.fft.rfft2(I, s=(padH, padW))
    Fp = np.fft.rfft2(p, s=(padH, padW))
    FY = FI * Fp
    Y  = np.fft.irfft2(FY, s=(padH, padW))

    sy = (h - 1) // 2
    sx = (w - 1) // 2
    return Y[sy:sy+H, sx:sx+W]


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.min()
    rng = np.ptp(x)
    if rng < 1e-8:
        return np.zeros_like(x)
    return x / rng


def validate_one_label(in_dir, out_dir, lbl):
    sharp_path = os.path.join(in_dir, f"{lbl}_sharp_patch.png")
    blur_path  = os.path.join(in_dir, f"{lbl}_blur_patch.png")
    psf_path   = os.path.join(in_dir, f"{lbl}_psf.npy")

    missing = []
    if not os.path.exists(sharp_path):
        missing.append("sharp_patch")
    if not os.path.exists(blur_path):
        missing.append("blur_patch")
    if not os.path.exists(psf_path):
        missing.append("psf")
    if missing:
        print(f"[{lbl}] missing: {', '.join(missing)} – skipping.")
        return None  # signal "skipped"

    # --- load data ---
    I = imageio.imread(sharp_path).astype(np.float64)
    J = imageio.imread(blur_path).astype(np.float64)
    if I.ndim == 3:
        I = I[..., 0]
    if J.ndim == 3:
        J = J[..., 0]
    p = np.load(psf_path).astype(np.float64)

    # --- reblur ---
    J_hat = conv_fft_same(I, p)

    # brightness scaling
    alpha = (J.sum() + 1e-8) / (J_hat.sum() + 1e-8)
    J_hat *= alpha

    # --- metrics ---
    resid = J_hat - J
    mse = float(np.mean(resid ** 2))
    max_val = float(np.max(J))
    if mse < 1e-12:
        psnr = float("inf")
    else:
        psnr = 10.0 * np.log10((max_val ** 2) / mse)

    print(f"[{lbl}] MSE={mse:.4e}, PSNR={psnr:.2f} dB")

    # --- visualization (5-panel: I, J, I conv p, |J−I conv p|, p) ---
    I_n    = normalize01(I)
    J_n    = normalize01(J)
    Jhat_n = normalize01(J_hat)
    diff_n = normalize01(np.abs(J_hat - J))
    psf_n  = normalize01(p)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3))
    titles = [
        "sharp patch I",
        "blur patch J",
        "reblur I⊗p",
        "|J − I⊗p|",
        "PSF p"
    ]
    imgs = [I_n, J_n, Jhat_n, diff_n, psf_n]

    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im, cmap="gray")
        ax.set_title(t, fontsize=9)
        ax.axis("off")

    fig.suptitle(
        f"PSF validation for label {lbl}  "
        f"(MSE={mse:.4e}, PSNR={psnr:.2f} dB)",
        fontsize=11
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{lbl}_validate.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    return mse, psnr


def main():
    IN_DIR  = "resultsCLIP/sharp-refocus_aligned"
    OUT_DIR = "validate/validate_sharp-no-focus_psfs_ALIGNEDNEW"
    labels = [str(i) for i in range(-10, 11)]

    print(f"Validating PSFs from '{IN_DIR}' → '{OUT_DIR}'")
    validated = []
    skipped   = []

    for lbl in labels:
        res = validate_one_label(IN_DIR, OUT_DIR, lbl)
        if res is None:
            skipped.append(lbl)
        else:
            validated.append(lbl)

    print("\nSummary:")
    print("  validated labels:", validated)
    print("  skipped labels  :", skipped if skipped else "none")
    print("ALLL DONEE.")


if __name__ == "__main__":
    main()
