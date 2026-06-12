import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import rgb2lab, lab2rgb
from scipy.ndimage import uniform_filter
 
 
def load_image(path):
    return Image.open(path).convert("RGB")
 
 
# ── Pre-processing ────────────────────────────────────────────────────────────
 
def _sample_bg_color(arr, s=5):
    corners = np.concatenate([
        arr[:s, :s].reshape(-1, 3), arr[:s, -s:].reshape(-1, 3),
        arr[-s:, :s].reshape(-1, 3), arr[-s:, -s:].reshape(-1, 3),
    ])
    return np.median(corners, axis=0)
 
 
def remove_background(image, threshold=30.0, feather=2):
    arr  = np.array(image.convert("RGB")).astype(float)
    dist = np.sqrt(np.sum((arr - _sample_bg_color(arr)) ** 2, axis=2))
    mask = Image.fromarray((dist > threshold).astype(np.uint8) * 255, "L")
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
        mask = Image.fromarray((np.array(mask) > 128).astype(np.uint8) * 255, "L")
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    return rgba
 
 
def autocrop(image, padding=8):
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    bbox = image.split()[-1].getbbox()
    if bbox is None:
        return image
    w, h = image.size
    return image.crop((max(0, bbox[0]-padding), max(0, bbox[1]-padding),
                       min(w, bbox[2]+padding), min(h, bbox[3]+padding)))
 
 
def preprocess(image, remove_bg=True, crop=True, threshold=30.0,
               padding=12, bg_color=(240, 235, 220)):
    rgba = image.convert("RGBA")
    if remove_bg:
        rgba = remove_background(rgba, threshold=threshold)
    if crop:
        rgba = autocrop(rgba, padding=padding)
    bg = Image.new("RGBA", rgba.size, bg_color + (255,))
    bg.paste(rgba, mask=rgba.split()[-1])
    return bg.convert("RGB")
 
 
# ── Baselines ─────────────────────────────────────────────────────────────────
 
def baseline_pixelate(image, pixel_size=50):
    w, h = image.size
    return image.resize((pixel_size, pixel_size), Image.NEAREST).resize((w, h), Image.NEAREST)
 
 
def kmeans_pixelate(image, pixel_size=50, n_colors=16):
    w, h   = image.size
    small  = image.resize((pixel_size, pixel_size), Image.LANCZOS)
    pixels = np.array(small).reshape(-1, 3).astype(float)
    km     = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = km.fit_predict(pixels)
    palette = km.cluster_centers_.astype(np.uint8)
    return Image.fromarray(
        palette[labels].reshape(pixel_size, pixel_size, 3).astype(np.uint8), "RGB"
    ).resize((w, h), Image.NEAREST)
 
 
# ── SLIC Pixel Art (Gerstner 2012) ────────────────────────────────────────────
 
def slic_pixelate(
    image,
    pixel_size: int = 50,
    n_colors: int = 16,
    compactness: float = 20.0,
    bilateral_sigma: float = 1.5,
    n_iter: int = 3,
):
    """
    Pixel art via SLIC superpixels + palette optimisation.
    Loosely follows Gerstner et al. 2012 "Pixelated Image Abstraction".
 
    The core idea: instead of downsampling first (which blends colours),
    we segment the full-resolution image into exactly pixel_size² superpixels
    using SLIC.  Each superpixel corresponds to one output pixel; its colour
    is determined by the pixels inside it, not by any averaging across the
    grid boundary.  High-contrast features like eye-whites form their own
    superpixels naturally because SLIC penalises crossing colour boundaries.
 
    Pipeline
    --------
    1. SLIC segmentation in LAB colour space.
       n_segments = pixel_size²; compactness controls the colour/spatial
       trade-off (higher = blockier, lower = more shape-following).
 
    2. Per-superpixel colour: mean of contained pixels in LAB, then
       bilaterally smoothed across the output grid to suppress gradient
       noise while preserving sharp boundaries (Gerstner §4.2).
 
    3. Palette refinement loop (simplified MCDA → K-Means):
       a. K-Means on superpixel colours → palette of n_colors.
       b. Re-assign each superpixel to nearest palette colour.
       c. Re-fit K-Means from those assignments.
       Iterate n_iter times so the palette stabilises around the actual
       quantised colours rather than the raw per-superpixel means.
 
    4. Map superpixel → output grid cell via centroid, render, scale up.
 
    Parameters
    ----------
    pixel_size    : Output grid resolution.
    n_colors      : Palette size (16–32 typical).
    compactness   : SLIC spatial regularisation.  10–30 for cartoons,
                    higher for photos.  Lower = superpixels follow colour
                    boundaries more closely.
    bilateral_sigma : Smoothing radius for superpixel colour smoothing.
    n_iter        : Palette refinement iterations.
    """
    w, h      = image.size
    img_arr   = np.array(image)          # (H, W, 3) uint8
    img_lab   = rgb2lab(img_arr)         # (H, W, 3) float, LAB
 
    # ── 1. SLIC segmentation ─────────────────────────────────────────────────
    n_segments = pixel_size * pixel_size
    print(f"[slic] Segmenting into {n_segments} superpixels…")
    labels_map = slic(
        img_arr,
        n_segments=n_segments,
        compactness=compactness,
        convert2lab=True,
        start_label=0,
        sigma=0,           # no pre-smoothing; we want sharp boundaries
    )
    # labels_map: (H, W), int, values in [0, n_actual-1]
    n_actual = labels_map.max() + 1
 
    # ── 2. Per-superpixel mean colour (LAB) and centroid (x, y) ─────────────
    print(f"[slic] Computing {n_actual} superpixel colours & centroids…")
    sp_color    = np.zeros((n_actual, 3), dtype=float)   # LAB
    sp_centroid = np.zeros((n_actual, 2), dtype=float)   # (row, col)
    sp_count    = np.zeros(n_actual, dtype=int)
 
    rows, cols = np.indices((h, w))
    for s in range(n_actual):
        mask        = labels_map == s
        cnt         = mask.sum()
        if cnt == 0:
            continue
        sp_count[s]    = cnt
        sp_color[s]    = img_lab[mask].mean(axis=0)
        sp_centroid[s, 0] = rows[mask].mean()
        sp_centroid[s, 1] = cols[mask].mean()
 
    # ── 2b. Bilateral-style smoothing of superpixel colours ──────────────────
    # Build a wout×hout grid of mean colours, smooth it, read back.
    # This is Gerstner's bilateral filter step: suppresses gradient noise
    # in flat regions while respecting edges (because edges → separate SPs).
    grid_lab = np.zeros((pixel_size, pixel_size, 3), dtype=float)
    grid_cnt = np.zeros((pixel_size, pixel_size), dtype=int)
 
    cell_r = (sp_centroid[:, 0] / h * pixel_size).clip(0, pixel_size - 1).astype(int)
    cell_c = (sp_centroid[:, 1] / w * pixel_size).clip(0, pixel_size - 1).astype(int)
 
    for s in range(n_actual):
        if sp_count[s] == 0:
            continue
        r, c = cell_r[s], cell_c[s]
        grid_lab[r, c] += sp_color[s]
        grid_cnt[r, c] += 1
 
    # Average cells that got multiple superpixels, fill empties via blur
    nonzero = grid_cnt > 0
    grid_lab[nonzero] /= grid_cnt[nonzero, np.newaxis]
    # Fill sparse cells with local average
    filled = uniform_filter(grid_lab, size=(int(bilateral_sigma * 2 + 1), int(bilateral_sigma * 2 + 1), 1))
    grid_lab[~nonzero] = filled[~nonzero]
 
    # Write smoothed colours back to superpixels
    for s in range(n_actual):
        if sp_count[s] == 0:
            continue
        sp_color[s] = grid_lab[cell_r[s], cell_c[s]]
 
    # ── 3. Palette refinement loop ───────────────────────────────────────────
    print(f"[slic] Refining palette ({n_colors} colours, {n_iter} iterations)…")
    valid = sp_count > 0
    sp_rgb = (lab2rgb(sp_color[valid]) * 255).clip(0, 255)  # (n_valid, 3)
 
    km = KMeans(n_clusters=min(n_colors, len(sp_rgb)), random_state=42, n_init=10)
    km.fit(sp_rgb)
 
    for _ in range(n_iter - 1):
        # Re-assign each SP to nearest palette colour, re-fit
        palette_rgb = km.cluster_centers_
        dists = np.sum((sp_rgb[:, None, :] - palette_rgb[None, :, :]) ** 2, axis=2)
        hard_labels = dists.argmin(axis=1)
        # Update palette as mean of assigned SPs
        new_palette = np.array([
            sp_rgb[hard_labels == k].mean(axis=0) if (hard_labels == k).any()
            else palette_rgb[k]
            for k in range(len(palette_rgb))
        ])
        km.cluster_centers_ = new_palette
 
    palette_rgb = km.cluster_centers_.astype(np.uint8)
    sp_palette_label = km.predict(sp_rgb)  # final assignment
 
    # ── 4. Render output grid ─────────────────────────────────────────────────
    print("[slic] Rendering output grid…")
    # Map valid SP index → palette colour
    valid_indices = np.where(valid)[0]
    sp_final_color = np.zeros((n_actual, 3), dtype=np.uint8)
    for i, s in enumerate(valid_indices):
        sp_final_color[s] = palette_rgb[sp_palette_label[i]]
 
    # Fill output grid: each cell gets colour of the superpixel whose
    # centroid falls in it.  Multiple → last write wins (minor artefact).
    out_grid = np.zeros((pixel_size, pixel_size, 3), dtype=np.uint8)
    for s in range(n_actual):
        if sp_count[s] == 0:
            continue
        out_grid[cell_r[s], cell_c[s]] = sp_final_color[s]
 
    # Fill any empty cells by nearest-neighbour propagation (simple dilation)
    from scipy.ndimage import distance_transform_edt, label as ndlabel
    empty = (out_grid.sum(axis=2) == 0)
    if empty.any():
        # Paint empty cells with colour of nearest non-empty cell
        _, nn_idx = distance_transform_edt(empty, return_indices=True)
        out_grid[empty] = out_grid[nn_idx[0][empty], nn_idx[1][empty]]
 
    result = Image.fromarray(out_grid, "RGB").resize((w, h), Image.NEAREST)
    return result
 
 
# ── Comparison ────────────────────────────────────────────────────────────────
 
def compare_results(image_path, pixel_size=50, n_colors=16,
                    remove_bg=True, crop=True):
    original  = load_image(image_path)
    processed = preprocess(original, remove_bg=remove_bg, crop=crop)
 
    print("Baseline…");   b   = baseline_pixelate(processed, pixel_size)
    print("K-Means…");    km  = kmeans_pixelate(processed, pixel_size, n_colors)
    print("SLIC…");       sp  = slic_pixelate(processed, pixel_size, n_colors)
 
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for ax, img, title in zip(axes,
        [processed, b, km, sp],
        ["Pre-processed",
         "Baseline\n(resize only)",
         "K-Means\n(downsample→quantize)",
         "SLIC ✦\n(Gerstner 2012)"]):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.suptitle(f"Pixel Art — {pixel_size}×{pixel_size}, {n_colors} colours", fontsize=13)
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
    print("Saved comparison.png")
 
 
def save_result(image, path):
    image.save(path)
    print(f"Saved {path}")
 
 
if __name__ == "__main__":
    original  = load_image("input.jpg")
    processed = preprocess(original)
    compare_results("input.jpg", pixel_size=50, n_colors=16)
    save_result(slic_pixelate(processed, 50, 16), "output_slic.png")