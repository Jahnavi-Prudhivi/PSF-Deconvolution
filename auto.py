import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches


#  FFT "same" convolution + adjoint

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


# returns the gradient of conv wrt psf p (used in primal update step)
#calculates L2 loss between curr convolved img and target img in freq domain --> this helps update the PSF in the CP optimization. The idea is that to update the PSF so that when we convolve the img it gets closer to the target.
def conv_adj_fft_to_psf(I: np.ndarray, resid: np.ndarray, psf_shape) -> np.ndarray:
    #padding for conv.
    H, W = I.shape
    h, w = psf_shape
    padH, padW = H + h - 1, W + w - 1

    # gets the 2D FFT of the img I and resid 
    FI = np.fft.rfft2(I,     s=(padH, padW))
    FR = np.fft.rfft2(resid, s=(padH, padW))
    
    # compute the gradient of FFT - L2 loss between convolved img and target img.
    K  = np.fft.irfft2(np.conj(FI) * FR, s=(padH, padW))

    # need this bc the result will be larger than the PSF bc of padding - this crops the result to match the exact size of the PSF
    sy = H - 1 - (h - 1) // 2
    sx = W - 1 - (w - 1) // 2
    return K[sy:sy+h, sx:sx+w]


#how much the field spreads out or converges at each point
def divergence(qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    div = np.zeros_like(qx)
    div[:-1, :] += qy[:-1, :]
    div[1:,  :] -= qy[:-1, :]
    div[:, :-1] += qx[:, :-1]
    div[:, 1: ] -= qx[:, :-1]
    return div

#  Automatic checkerboard patch finder (left side)
def extract_checkerboard_patch(
    I: np.ndarray,
    J: np.ndarray,
    patch_h: int = 256,
    patch_w: int = 256,
    frac: float = 0.5,
    side: str = "left",   # "left" or "right"
):
    """
    Automatically pick a patch on the checkerboard by
    looking for a high-variance region in |I-J|.

    side = "left"  -> search left frac of the image
    side = "right" -> search right frac of the image
    """
    assert I.shape == J.shape, f"Sharp and blur must be same shape, got {I.shape} vs {J.shape}"
    H, W = I.shape

    # if patch bigger than image, just center-crop
    if (patch_h > H) or (patch_w > W):
        print("  [warn] patch larger than image; using center crop")
        cy, cx = H // 2, W // 2
        y0 = max(0, cy - patch_h // 2)
        x0 = max(0, cx - patch_w // 2)
        y0 = min(y0, H - patch_h)
        x0 = min(x0, W - patch_w)
        return I[y0:y0+patch_h, x0:x0+patch_w], \
               J[y0:y0+patch_h, x0:x0+patch_w], \
               (x0, y0)

    # difference image highlights edges (checkerboard)
    D = np.abs(I.astype(np.float64) - J.astype(np.float64))
    D = D - D.min()
    D /= (D.max() + 1e-8)

    # -------- pick horizontal search range depending on side --------
    frac = float(frac)
    frac = max(0.0, min(1.0, frac))  # clamp to [0,1]

    if side == "left":
        x_start = 0
        x_end   = int(W * frac)
        if x_end - x_start < patch_w:
            x_end = patch_w
    elif side == "right":
        x_end   = W
        x_start = int(W * (1.0 - frac))
        if x_end - x_start < patch_w:
            x_start = x_end - patch_w
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    # vertical search is full height
    y_start = 0
    y_end   = H

    y_max = y_end - patch_h
    x_max = x_end - patch_w

    if x_max < x_start or y_max < y_start:
        # fallback: center crop
        cy = H // 2
        cx = W // 2
        y0 = max(0, cy - patch_h // 2)
        x0 = max(0, cx - patch_w // 2)
        y0 = min(y0, H - patch_h)
        x0 = min(x0, W - patch_w)
        return I[y0:y0+patch_h, x0:x0+patch_w], \
               J[y0:y0+patch_h, x0:x0+patch_w], \
               (x0, y0)

    ys = np.arange(y_start, y_max + 1)
    xs = np.arange(x_start, x_max + 1)

    # integral image for fast rectangle sums
    S = D.cumsum(axis=0).cumsum(axis=1)
    S = np.pad(S, ((1,0),(1,0)), mode='constant', constant_values=0)

    Y0 = ys[:, None]
    X0 = xs[None, :]
    Y1 = Y0 + patch_h
    X1 = X0 + patch_w

    sum_map = S[Y1, X1] - S[Y0, X1] - S[Y1, X0] + S[Y0, X0]

    idx = np.argmax(sum_map)
    iy, ix = np.unravel_index(idx, sum_map.shape)
    y0 = int(ys[iy])
    x0 = int(xs[ix])

    I_patch = I[y0:y0+patch_h, x0:x0+patch_w]
    J_patch = J[y0:y0+patch_h, x0:x0+patch_w]

    return I_patch, J_patch, (x0, y0)


def save_overlay_triplet(I, J, x0, y0, ph, pw, out_path, title="patch"):
    #62 bc we want double precision, after necessary operations with higher precision (in float64), you typically don't need that level of precision for display or saving the image.

    #absolute difference between the sharp image I and the blurred image J pixel by pixel
    D = np.abs(I.astype(np.float64) - J.astype(np.float64))

    #normalize
    # shifts all the values of D so that the smallest value becomes zero
    # image processing are usually expected to have non-negative values/helps standardize the data
    D -= D.min()
    #normalizing the values in the array D to the range [0, 1].
    D /= (D.max() + 1e-8) #ensures denominator in the division is never exactly zero

    #Plots
    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    for ax, img, name in zip(axes,
                             [I, J, D],
                             ["sharp", "blur", "|sharp-blur|"]):
        ax.imshow(img, cmap="gray")
        ax.set_title(name)
        ax.axis("off")
        rect = patches.Rectangle(
            (x0, y0),
            pw, ph,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)



#  Chambolle–Pock for PSF
    #important notes: minp (F(Kp) + G(p))
        # F => how well the current PSF explains the blurred img observed
        # G -> reg term, this penalizes undesirable properties of the PSF like roughness
    
    #I - blurred img
    #J - sharp img, p- psf we're optimizing for and pbar - momentum term used for faster convergence



def chambolle_pock_psf(I, J, p_init,
                        lam=1e-5,
                        mu=10.0,
                        tau=0.05,
                        sigma=0.05,
                        theta=1.0,
                        max_iter=500):
    #converted images to I and J to avoid integer overflow and so we normalize them so both I and J are in between 0 and 1
    I = I.astype(np.float64)
    J = J.astype(np.float64)
    I = I - I.min()
    J = J - J.min()
    I /= (I.max() + 1e-8)
    J /= (J.max() + 1e-8)

    #p is the initial PSF and pbar is used to store the momentum of the PSF during updates. this is for faster convergence
    p = p_init.copy().astype(np.float64) #initial psf
    p_bar = p.copy() 
    h, w = p.shape

    #qx and qy are dual variables - represent the gradients of the PSF & updated in the loop.
    qx = np.zeros_like(p)
    qy = np.zeros_like(p)

    #makes sure the sum of px values in J and Ixp are equal
    s = I.sum() / (J.sum() + 1e-8)

    for it in range(max_iter):
        # DUALL UPDATEEE:
            #computes the gradients in x dir and y dir
        grad_x = np.zeros_like(p_bar)
        grad_y = np.zeros_like(p_bar)
        grad_x[:, :-1] = p_bar[:, 1:] - p_bar[:, :-1]
        grad_y[:-1, :] = p_bar[1:, :] - p_bar[:-1, :]

        #qx and qy variables represent dual var (like yk in my paper), we accumulate the gradients scaled ((which controls step size for updating dual var))
        qx += sigma * grad_x
        qy += sigma * grad_y

        #compute the norm to make sure the values don't get too large - this is the projection step in my paper --> keeps everything in check so the magnitude doesn't go beyond 1.
        norm = np.maximum(1.0, np.sqrt(qx*qx + qy*qy))
        qx /= norm
        qy /= norm

        # PRIMALLL UPDATEEE
            # optimizes psf p based on dual var and curr state of the system

        #compute the residual between convolved p and J

        #eqn 15 comes in here in the primal update!
        I_conv_p = conv_fft_same(I, p)    #data term eqn 15
        resid = I_conv_p - s * J   ## full data term of eqn 15

        #gradient of the data term: compute the gradient with respect to the psf p using convolution  ==> basically the L2 norm - it tells us how to adjust the PSF p to reduce the error between I conv p and s*j
            #convffttopsf=> computes the gradient of the conv wrt p (basically gives the update dir.)
        g_data = 2.0 * conv_adj_fft_to_psf(I, resid, (h, w))
        #divergence of the dual variables is addied to the primal update - forces a smooth solution
        div_q  = divergence(qx, qy)

        #our sum constraint -> makes sure the psf p sums to 1, mu controls strength of this reg.
        g_sum  = 2.0 * mu * (p.sum() - 1.0)

        #new psf after the primal update, adjust p by subtracting gradients from data term, reg and constraints and scale it by tau
        p_new = p - tau * (g_data + g_sum - lam * div_q)

        # normalizes and makes sure the PSF remains non-negative and normalized sum = 1(this is kinda like projection)
        p_new = np.maximum(p_new, 0.0)
        ps = p_new.sum()
        if ps > 0:
            p_new /= ps

        #adding extrapolation step to speed up convergence by adding some momentum
        p_bar = p_new + theta * (p_new - p)
        p = p_new

        # just for debugging so i can see the progress not too important ig
        if (it + 1) % 50 == 0:
            resid_dbg = conv_fft_same(I, p) - s * J
            data_term = (resid_dbg * resid_dbg).mean()
            print(f"    iter {it+1:4d} | data_term={data_term:.6e}, psf_max={p.max():.4e}")

    return p.astype(np.float32)

#  Main


def main():
    IN_DIR  = "phasecam4_images/sharp-refocus"   # from batch.py
    OUT_DIR = "results/auto_psfs_phasecam4_REfocus"
    os.makedirs(OUT_DIR, exist_ok=True)

    # "nominal" desired patch + PSF sizes
    PATCH_H_TARGET = 128   
    PATCH_W_TARGET = 128   
    PSF_SIZE = 21

    labels = [str(i) for i in range(-10, 11)]

    for lbl in labels:
        sharp_path = os.path.join(IN_DIR, f"{lbl}.jpg")
        blur_path  = os.path.join(IN_DIR, f"{lbl}.jpg")

        if not (os.path.exists(sharp_path) and os.path.exists(blur_path)):
            print(f"[{lbl}] missing images, skipping. ({sharp_path} / {blur_path})")
            continue

        I_full = imageio.imread(sharp_path).astype(np.float32)
        J_full = imageio.imread(blur_path).astype(np.float32)

        # wants grayscale i dont want to deal w rgb yet
        # only the first channel (I_full[..., 0] and J_full[..., 0]) is used converting the images to grayscale
        if I_full.ndim == 3:
            I_full = I_full[..., 0]
        if J_full.ndim == 3:
            J_full = J_full[..., 0]

        print(f"[{lbl}] full shape = {I_full.shape}, blur shape = {J_full.shape}")

        # safety: enforce same shape
        #if I_full.shape != J_full.shape:
           # print(f"  [warn] shape mismatch after alignment: {I_full.shape} vs {J_full.shape}")
           # Hc = min(I_full.shape[0], J_full.shape[0])
          #  Wc = min(I_full.shape[1], J_full.shape[1])
          #  I_full = I_full[:Hc, :Wc]
          #  J_full = J_full[:Hc, :Wc]
          #  print(f"  [fix] cropped to {I_full.shape}")

        H_img, W_img = I_full.shape

        # adapt patch size to image size, but keep >= PSF_SIZE
        patch_h = min(PATCH_H_TARGET, H_img)
        patch_w = min(PATCH_W_TARGET, W_img)

        # the patch needs to fit and should be less than the PSF so it can fit
        if patch_h < PSF_SIZE or patch_w < PSF_SIZE:
            print(f"[{lbl}] image too small for PSF size {PSF_SIZE}, skipping.")
            continue
        print(f"[{lbl}] using patch size = ({patch_h}, {patch_w})")

        #extract best checkerboard patch on left 
        I_patch, J_patch, (x0, y0) = extract_checkerboard_patch(
            I_full, J_full,
            patch_h=patch_h,
            patch_w=patch_w,
            frac=0.4,       # search 40% of the width
            side="right"    # <--- THIS moves the red box to the right
        )
        #prints on console which patch im using
        print(f"[{lbl}] patch @ (x0={x0}, y0={y0}), shape={I_patch.shape}")

        # debug overlay
        overlay_path = os.path.join(OUT_DIR, f"{lbl}_overlay.png")
        #sharp img I, blurred img J, (x,y position of the patch inside the original image)
        # height and width of the patch that was extracted
        #overlay path where the overlay image will be saved
        #title that will be used for the image when it’s displayed
        save_overlay_triplet(I_full, J_full, x0, y0, patch_h, patch_w,
                             overlay_path, title=f"label {lbl}")


        # normalize patches for saving
        # sharp and blurred patches so that their pixel values are scaled between 0 and 1
        I_range = np.ptp(I_patch) #calculates the difference between the maximum and minimum values in an array
        # hand, ensure that all images you process have pixel values within the same range (e.g., [0, 1]), which leads to more predictable results
        J_range = np.ptp(J_patch)
        I_norm = (I_patch - I_patch.min()) / (I_range + 1e-6)
        J_norm = (J_patch - J_patch.min()) / (J_range + 1e-6)

        imageio.imwrite(os.path.join(OUT_DIR, f"{lbl}_sharp_patch.png"),
                        (I_norm * 255).astype(np.uint8))
        imageio.imwrite(os.path.join(OUT_DIR, f"{lbl}_blur_patch.png"),
                        (J_norm * 255).astype(np.uint8))

        # ----- initial PSF: small centered Gaussian -----
        #x and y are the coordinates of each pixel in the kernel
        h = w = PSF_SIZE
        #creates a 2d grid for x and y - these are where the fn itself is done
            ##xx x-coordinates from 0 to w-1, same w y
        yy, xx = np.mgrid[:h, :w]
        # center of the Gaussian kernel is defined by the coordinates
            #cx = center in the horizontal direction, cy as vertical dir
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.
        #standard dev controls how spread out the Gaussian & multiply bc standard deviation is proportional to the smallest dimension of the kernel
        sigma_psf = 0.25 * min(h, w)
        #formula in my paper
        p_init = np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2.0 * sigma_psf**2))
        #makes sure there are no negative values in the kernel
        p_init = np.maximum(p_init, 0.0)
        #Gaussian kernel should have a total sum of 1 to preserve the image intensity during convolution
        p_init /= (p_init.sum() + 1e-8)

        # ----- run Chambolle–Pock -----
        print(f"[{lbl}] running CP PSF estimation...")
        p_est = chambolle_pock_psf(
            I_patch, J_patch, p_init,
            lam=1e-5, mu=1.0,
            tau=0.05, sigma=0.05,
            max_iter=300
        )

        print(f"[{lbl}] PSF sum={p_est.sum():.6f}, max={p_est.max():.4e}")

        # save PSF
        np.save(os.path.join(OUT_DIR, f"{lbl}_psf.npy"), p_est)

        plt.figure(figsize=(5,5))
        plt.imshow(p_est, cmap="gray", interpolation="bicubic")
        plt.title(f"PSF {lbl}")
        plt.axis("off")
        plt.savefig(os.path.join(OUT_DIR, f"{lbl}_psf.png"),
                    dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()

    print("Done.")


if __name__ == "__main__":
    main()



#more notes:
    # primal update: updates p (primal var) using feedback from dual var y and the reg term
    #dual update: updates the dual var y which encodes how well the curr solution p fits the data and constraints
    #extrapolation - speeds up the convergence by adding some momentum to the primal var

