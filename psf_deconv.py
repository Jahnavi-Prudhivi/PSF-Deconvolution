import numpy as np
import os
from PIL import Image

def conv(I: np.ndarray, p: np.ndarray) -> np.ndarray:
    # I conv p
        # should take in I sharp img
        # p: psf

        # should return blurred img (should be same as I)
                # I conv P  = F^-1 (F(I) * F (p))
        # performs 2d conv
    H, W = I.shape #img size
    h, w = p.shape #psf size
    padH, padW = H + h - 1, W + w - 1
    
    # from px to freq space
    FI = np.fft.rfft2(I, s=(padH, padW))
    Fp = np.fft.rfft2(p, s=(padH, padW))
    # mult in freq domain is conv operation.
    FY = FI * Fp
    #take inverse of F-1 to get it back into img
    # freq space to px space
    Y  = np.fft.irfft2(FY, s=(padH, padW))
    #see which padding to remove
    sy = (h - 1) // 2
    sx = (w - 1) // 2
    #remove padding         converts result back to same data type as sharp img.
    return Y[sy:sy+H, sx:sx+W].astype(I.dtype)


# this whole eqn returns a float bc its a number saying how different the sim blur is from real blur and how smooth the psf is and whether it sums to 1. this outputs energy/cost
def eqn15_energy(I: np.ndarray, J: np.ndarray, p: np.ndarray, lam: float, mu: float) -> float:
    #first term
    s = float(I.sum()) / float(J.sum())
    I_blur = conv(I, p)
    residual = I_blur - s * J
    data_term = float((residual * residual).sum()) #L2 term
    #second term
    gx = np.zeros_like(p)
    gy = np.zeros_like(p)
    gx[:, :-1] = p[:, 1:] - p[:, :-1]
    gy[:-1, :] = p[1:, :] - p[:-1, :]
    tv_term = lam * np.sqrt(gx*gx + gy*gy + 1e-12).sum()

    #third term
        #u don't need to write 1.T @ p when p.sum() basically does the dot product for us.
    energy_term = mu * (float(p.sum()) - 1.0)**2
    
    return float(data_term + tv_term + energy_term)


# loading image as grayscale
def load_img01(path: str) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("L"), np.float32) / 255.0
    return img

def main():
    IN_DIR = "outputs/aligned_pairs"
    labels = [str(i) for i in range(-10, 11)]

    psf_size = 15
    p = np.ones((psf_size, psf_size), np.float32)
    p /= p.sum()

    lam = 1e-3
    mu = 10.0

    for lbl in labels:
        sharp_path = os.path.join(IN_DIR, f"{lbl}_sharp.png")
        blur_path  = os.path.join(IN_DIR, f"{lbl}_blur.png")
        if not (os.path.exists(sharp_path) and os.path.exists(blur_path)):
            print(f"[skip] missing pair for {lbl}")
            continue

        I = load_img01(sharp_path)
        J = load_img01(blur_path)
        print(f"Loaded pair {lbl}: shape={I.shape}")

        E = eqn15_energy(I, J, p, lam, mu)
        print(f"[{lbl}] Energy = {E:.6f}")

if __name__ == "__main__":
    main()