import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import rgb2lab, lab2rgb


def load_image(path):
    """Load image and convert to RGB."""
    return Image.open(path).convert("RGB")


# ── Pre-processing ────────────────────────────────────────────────────────────

def _sample_bg_color(arr, s=5):
    corners = np.concatenate([
        arr[:s, :s].reshape(-1, 3), arr[:s, -s:].reshape(-1, 3),
        arr[-s:, :s].reshape(-1, 3), arr[-s:, -s:].reshape(-1, 3),
    ])
    return np.median(corners, axis=0)


def remove_background(image, threshold=30.0, feather=2):
    """Corner-sampled background colour distance thresholding."""
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
    """Crop to the alpha channel's bounding box."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    bbox = image.split()[-1].getbbox()
    if bbox is None:
        return image
    w, h = image.size
    return image.crop((max(0, bbox[0] - padding), max(0, bbox[1] - padding),
                        min(w, bbox[2] + padding), min(h, bbox[3] + padding)))


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

def baseline_pixelate(image, pixel_size=70):
    """Naive resize down then back up. Colour averaging destroys fine detail."""
    w, h = image.size
    small = image.resize((pixel_size, pixel_size), Image.NEAREST)
    return small.resize((w, h), Image.NEAREST)


def kmeans_pixelate(image, pixel_size=70, n_colors=32):
    """
    Original approach: downsample first (LANCZOS), then K-Means quantise.

    Known flaw, kept here only as a comparison baseline: LANCZOS averaging
    blends high-contrast neighbours (e.g. white eye-whites against a black
    body -> grey), so small bright details are lost before clustering even
    runs.
    """
    w, h   = image.size
    small  = image.resize((pixel_size, pixel_size), Image.LANCZOS)
    pixels = np.array(small).reshape(-1, 3).astype(float)

    km      = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    km.fit(pixels)
    palette = km.cluster_centers_.astype(np.uint8)
    labels  = km.labels_

    quantized = palette[labels].reshape(pixel_size, pixel_size, 3)
    return Image.fromarray(quantized, "RGB").resize((w, h), Image.NEAREST)


# ── Palette-first ────────────────────────────────────────────────────────────

def palette_first_pixelate(image, pixel_size=70, n_colors=32,
                            contrast_bias: float = 2.0):
    """
    Detail-preserving pixel art via palette-first quantisation.

    The key insight: colour averaging during downsampling destroys small
    high-contrast features (thin lines, eye-whites surrounded by black).
    By discretising colours BEFORE downsampling, the blending problem never
    has a chance to happen.

    Pipeline
    --------
    1. K-Means on the full-resolution image -> build a palette of n_colors.
    2. Map every pixel in the original to its nearest palette entry.
       The image is now fully discrete -- no intermediate greys exist.
    3. Downsample via MODE voting with contrast bias: for each output
       pixel, look at the corresponding source block and vote for the
       palette colour that appears most. Blocks with high internal
       contrast get their minority colour's vote weight boosted, so
       minority-but-important colours (eye-whites, line edges) are not
       drowned out by the majority colour.
    4. Scale back up with nearest-neighbour (no new blending).

    This is the strongest method here for raw colour fidelity, since the
    palette is fit directly on untouched source pixels. Its weakness is
    structural: because every block is decided independently, boundaries
    follow the source image's exact (and often irregular) contours rather
    than the straighter, more regular edges a human pixel artist would
    draw. See slic_pixelate() for the boundary-regularised alternative.
    """
    w, h = image.size

    # Step 1: fit palette on full-resolution image
    pixels_full = np.array(image).reshape(-1, 3).astype(float)
    n_sample = min(len(pixels_full), 50_000)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pixels_full), n_sample, replace=False)
    km = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    km.fit(pixels_full[idx])
    palette = km.cluster_centers_.astype(np.uint8)   # (n_colors, 3)

    # Step 2: map every original pixel to its nearest palette colour
    diffs = pixels_full[:, None, :] - palette[None, :, :].astype(float)
    label_map = np.argmin(np.sum(diffs ** 2, axis=2), axis=1)
    quantized_full = label_map.reshape(h, w)

    # Step 3: mode-vote downsample with contrast bias
    out_labels = np.zeros((pixel_size, pixel_size), dtype=np.int32)
    block_h = h / pixel_size
    block_w = w / pixel_size

    for row in range(pixel_size):
        for col in range(pixel_size):
            r0, r1 = int(row * block_h), int((row + 1) * block_h)
            c0, c1 = int(col * block_w), int((col + 1) * block_w)
            block = quantized_full[r0:r1, c0:c1].ravel()
            if len(block) == 0:
                out_labels[row, col] = 0
                continue

            counts = np.bincount(block, minlength=n_colors).astype(float)

            if contrast_bias > 1.0 and counts.sum() > 0:
                present = np.where(counts > 0)[0]
                if len(present) > 1:
                    present_colors = palette[present].astype(float)
                    brightness = present_colors.mean(axis=1)
                    contrast = brightness.max() - brightness.min()
                    norm_contrast = contrast / 255.0
                    majority_label = counts.argmax()
                    minority_mask = np.ones(n_colors, dtype=bool)
                    minority_mask[majority_label] = False
                    counts[minority_mask] *= (1.0 + norm_contrast * (contrast_bias - 1.0))

            out_labels[row, col] = counts.argmax()

    # Step 4: colour the output grid and scale up
    out_rgb = palette[out_labels]
    small = Image.fromarray(out_rgb.astype(np.uint8), "RGB")
    return small.resize((w, h), Image.NEAREST)


# ── SLIC pixel art with boundary regularisation (Gerstner et al. 2012) ───────

def _compute_centroids(label_map, n_sp, rows, cols):
    centroids = np.zeros((n_sp, 2))
    counts = np.zeros(n_sp, dtype=int)
    for s in range(n_sp):
        mask = label_map == s
        c = mask.sum()
        counts[s] = c
        if c:
            centroids[s, 0] = rows[mask].mean()
            centroids[s, 1] = cols[mask].mean()
    return centroids, counts


def _build_adjacency(label_map, n_sp):
    """4-connected adjacency between superpixel labels, vectorised."""
    a, b = label_map[:, :-1].ravel(), label_map[:, 1:].ravel()
    c, d = label_map[:-1, :].ravel(), label_map[1:, :].ravel()
    pairs = np.concatenate([np.stack([a, b], axis=1), np.stack([c, d], axis=1)])
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    adjacency = [[] for _ in range(n_sp)]
    for x, y in pairs:
        adjacency[x].append(y)
        adjacency[y].append(x)
    return adjacency


def _joint_reassign(label_map, centroids, sp_color, adjacency, img_lab,
                     rows, cols, valid, compactness, spacing):
    """
    Reassign every pixel to the best centre among itself and its adjacent
    superpixels, using SLIC's own joint colour+space distance:

        D = sqrt(d_colour**2 + (d_space / spacing)**2 * compactness**2)

    This is the fix for the colour-blind version: a candidate centre with
    very different colour gets a large d_colour penalty, so it loses out
    to the pixel's current, colour-correct centre even when the centre's
    *position* just got nudged closer by Laplacian smoothing. A strong
    colour edge resists being dragged across; a weak/uniform area, where
    every candidate's colour is similar, offers little resistance and
    gets smoothed freely.
    """
    new_label_map = label_map.copy()
    for s in np.where(valid)[0]:
        mask = label_map == s
        if not mask.any():
            continue
        candidates = [s] + [n for n in adjacency[s] if valid[n]]
        cand_pos = centroids[candidates]                                  # (k,2)
        cand_col = sp_color[candidates]                                   # (k,3)

        px_pos = np.stack([rows[mask], cols[mask]], axis=1).astype(float)  # (m,2)
        px_col = img_lab[mask]                                             # (m,3)

        d_space = np.sqrt(np.sum((px_pos[:, None, :] - cand_pos[None, :, :]) ** 2, axis=2))
        d_color = np.sqrt(np.sum((px_col[:, None, :] - cand_col[None, :, :]) ** 2, axis=2))
        d = np.sqrt(d_color ** 2 + (d_space / spacing) ** 2 * compactness ** 2)

        winner = np.argmin(d, axis=1)
        new_label_map[mask] = np.array(candidates)[winner]
    return new_label_map


def slic_pixelate(
    image,
    pixel_size: int = 50,
    n_colors: int = 16,
    compactness: float = 20.0,
    n_iter: int = 3,
    smoothing: float = 0.3,
    color_sigma: float = 5.0,
):
    """
    Pixel art via SLIC superpixels with iterative boundary regularisation,
    following Gerstner et al. 2012, "Pixelated Image Abstraction" (NPAR).

    The paper's algorithm is not a single call to SLIC. It initialises
    superpixels on a regular grid, then iterates two steps: (1) reassign
    pixels to superpixels in a joint colour + position space, and (2)
    apply Laplacian smoothing to each superpixel centre's *position* --
    moving it part-way toward the average position of its 4-connected
    neighbours. That smoothing step is what straightens jagged, organic
    boundaries into the cleaner, more regular shapes a human pixel artist
    tends to draw; without it boundaries follow the source image's exact
    contours, which is why eyes and leaves came out lumpy in
    output_palette_first.png and the earlier SLIC output.

    Pipeline
    --------
    1. SLIC segmentation in LAB colour space gives the initial, colour-
       aware superpixel boundaries (n_segments = pixel_size**2).
    2. Repeat n_iter times:
       a. Laplacian-smooth each superpixel centre's position toward a
          COLOUR-WEIGHTED average of its neighbours' positions (see
          color_sigma below) -- this is what keeps a thin, minority-
          coloured region like the eye-white ring from having its own
          position estimate dragged off by its mostly differently-
          coloured neighbours, which otherwise undermined the colour
          protection in step b even though that step itself was correct.
       b. Re-assign every pixel to the best centre among itself and its
          ADJACENT superpixels, using SLIC's own joint colour+space
          distance, not pure spatial nearest-neighbour. A candidate whose
          colour is very different gets a large penalty in that distance,
          so a strong colour edge resists being dragged across even
          though its neighbouring centres just moved closer in position;
          a weak, low-contrast boundary has little colour penalty either
          way and gets smoothed freely.
       c. Recompute centroids and adjacency for the next iteration.
    3. Final palette: K-Means fit on the ORIGINAL full-resolution pixels,
       not on the smoothed superpixel means -- fitting on superpixel means
       was the earlier mistake that made colours muddy.
    4. Each superpixel's mean colour is matched to its nearest palette
       entry by CIE Lab Delta-E (perceptual distance), consistent with the
       MARD221 colour matching used elsewhere in this project.
    5. Render at full resolution, then downsample to the pixel_size grid
       via MODE VOTING over each block (the dominant superpixel wins) --
       not point-sampling, which is fragile to organic superpixel
       boundaries not aligning to the output grid.

    Parameters
    ----------
    pixel_size  : Output grid resolution.
    n_colors    : Palette size (16-32 typical).
    compactness : SLIC's colour/spatial trade-off, also used as the weight
                  in the joint-distance reassignment above. Higher = more
                  weight on position, boundaries snap to a blockier grid.
                  Lower = more weight on colour, boundaries follow colour
                  edges more closely. Back to the standard 20.0 default --
                  with colour_sigma below now protecting thin features,
                  compactness no longer has to do double duty, so it's
                  free to control overall boundary regularisation strength
                  on its own.
    n_iter      : Number of Laplacian-smoothing + reassignment rounds.
                  0 disables regularisation entirely (= plain SLIC).
    smoothing   : Fraction (0-1) each centre moves toward its neighbours'
                  average position per iteration. Higher = straighter,
                  more grid-like boundaries, but less faithful to the
                  source image's actual contours.
    color_sigma : Lab Delta-E scale for weighting neighbours in the
                  position-smoothing average. A neighbour whose colour
                  differs from this superpixel's by much more than
                  color_sigma contributes almost nothing to the average
                  (its weight decays as a Gaussian in colour distance);
                  a similarly-coloured neighbour contributes fully. This
                  is what stops a thin, minority-coloured superpixel (the
                  eye-white ring) from having its position estimate
                  dragged off by its mostly differently-coloured
                  neighbours. Defaults to 5.0 rather than a looser value
                  like 12: the cat's two eyes have genuinely different
                  amounts of white highlight in the source art (measured
                  at roughly 18% of the left eye's bounding box vs 12% of
                  the right eye's), so the thinner right-eye ring needed
                  tighter colour weighting to render as a continuous band
                  rather than scattered white cells inside the brown
                  outline. Lower = more protective of thin coloured detail
                  but less smoothing propagates between regions of
                  different colour; higher approaches the unweighted
                  (colour-blind) average.
    """
    w, h = image.size
    img_arr = np.array(image)
    img_lab = rgb2lab(img_arr)
    rows, cols = np.indices((h, w))

    # Step 1: initial SLIC segmentation
    n_segments = pixel_size * pixel_size
    label_map = slic(img_arr, n_segments=n_segments, compactness=compactness,
                      convert2lab=True, start_label=0, sigma=0)
    n_sp = label_map.max() + 1

    centroids, counts = _compute_centroids(label_map, n_sp, rows, cols)
    valid = counts > 0
    cur_label_map = label_map
    cur_centroids = centroids

    # Step 2: iterative colour-weighted Laplacian smoothing + joint reassignment
    spacing = np.sqrt(h * w / n_sp)
    for _ in range(n_iter):
        adjacency = _build_adjacency(cur_label_map, n_sp)

        # current representative colour of each superpixel, computed first so
        # the smoothing step below can use it to weight neighbours by colour
        sp_color = np.zeros((n_sp, 3))
        for s in np.where(valid)[0]:
            mask = cur_label_map == s
            sp_color[s] = img_lab[mask].mean(axis=0)

        # 2a. Laplacian-smooth positions toward a COLOUR-WEIGHTED neighbour
        # average: a neighbour whose colour is very different from this
        # superpixel's own colour contributes almost nothing to the average,
        # while a similarly-coloured neighbour contributes fully. Without
        # this, a thin, minority-coloured superpixel (like the eye-white
        # ring) gets its position dragged equally by every neighbour
        # regardless of colour -- mostly black/brown ones, since it has few
        # white neighbours of its own -- which pulls its *position estimate*
        # away from where its actual pixels are, even though the
        # colour-aware reassignment in step 2b would otherwise protect it.
        # This decouples "how much to protect thin coloured detail" (this
        # weighting) from "how much to straighten boundaries overall"
        # (compactness + smoothing), which used to be the same knob.
        new_centroids = cur_centroids.copy()
        for s in np.where(valid)[0]:
            neighbours = [n for n in adjacency[s] if valid[n]]
            if not neighbours:
                continue
            d_color = np.linalg.norm(sp_color[neighbours] - sp_color[s], axis=1)
            weights = np.exp(-(d_color ** 2) / (2 * color_sigma ** 2))
            if weights.sum() < 1e-9:
                continue
            avg_pos = (cur_centroids[neighbours] * weights[:, None]).sum(axis=0) / weights.sum()
            new_centroids[s] = (1 - smoothing) * cur_centroids[s] + smoothing * avg_pos

        # 2b. reassign pixels using the joint colour+space distance (the fix
        # from the previous round: a candidate whose colour is very
        # different gets a large penalty in this distance)
        cur_label_map = _joint_reassign(cur_label_map, new_centroids, sp_color,
                                         adjacency, img_lab, rows, cols, valid,
                                         compactness, spacing)
        cur_centroids = new_centroids

        # 2c. recompute centroids for the next iteration
        centroids, counts = _compute_centroids(cur_label_map, n_sp, rows, cols)
        valid = counts > 0
        cur_centroids = centroids

    # Final per-superpixel mean colour (LAB)
    final_color_lab = np.zeros((n_sp, 3))
    for s in np.where(valid)[0]:
        mask = cur_label_map == s
        final_color_lab[s] = img_lab[mask].mean(axis=0)

    # Step 3: palette fit on ORIGINAL full-resolution pixels
    pixels_full = img_arr.reshape(-1, 3).astype(float)
    n_sample = min(len(pixels_full), 60_000)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pixels_full), n_sample, replace=False)
    km = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    km.fit(pixels_full[idx])
    palette_rgb = km.cluster_centers_.astype(np.uint8)
    palette_lab = rgb2lab(palette_rgb.reshape(1, -1, 3) / 255.0).reshape(-1, 3)

    # Step 4: match each superpixel to nearest palette colour via Lab Delta-E
    valid_ids = np.where(valid)[0]
    diffs = final_color_lab[valid_ids][:, None, :] - palette_lab[None, :, :]
    nearest = np.argmin(np.sum(diffs ** 2, axis=2), axis=1)
    sp_final_rgb = np.zeros((n_sp, 3), dtype=np.uint8)
    sp_final_rgb[valid_ids] = palette_rgb[nearest]

    # Step 5: render to the output grid via MODE VOTING, not point-sampling.
    # Point-sampling (take the one pixel at each cell's centre) is fragile:
    # superpixel boundaries are organic and don't align to the pixel_size
    # grid, so a sliver of a neighbouring superpixel can poke into a cell's
    # footprint, and if the single sampled point lands on that sliver the
    # cell gets an isolated wrong colour -- this is exactly the speckle
    # noise seen in early output. Voting across the whole cell (as in
    # palette_first_pixelate) is robust to that: one stray pixel can't
    # outvote the cell's dominant superpixel.
    small = np.zeros((pixel_size, pixel_size, 3), dtype=np.uint8)
    block_h = h / pixel_size
    block_w = w / pixel_size
    for row in range(pixel_size):
        for col in range(pixel_size):
            r0, r1 = int(row * block_h), int((row + 1) * block_h)
            c0, c1 = int(col * block_w), int((col + 1) * block_w)
            block_labels = cur_label_map[r0:r1, c0:c1].ravel()
            vals, cnts = np.unique(block_labels, return_counts=True)
            winner = vals[np.argmax(cnts)]
            small[row, col] = sp_final_rgb[winner]
    return Image.fromarray(small, "RGB").resize((w, h), Image.NEAREST)


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_results(image_path, pixel_size=70, n_colors=16,
                    remove_bg=True, crop=True):
    original  = load_image(image_path)
    processed = preprocess(original, remove_bg=remove_bg, crop=crop)

    print("Baseline…");      b  = baseline_pixelate(processed, pixel_size)
    print("K-Means…");       km = kmeans_pixelate(processed, pixel_size, n_colors)
    print("Palette-First…"); pf = palette_first_pixelate(processed, pixel_size, n_colors)
    print("SLIC…");          sp = slic_pixelate(processed, pixel_size, n_colors)

    fig, axes = plt.subplots(1, 5, figsize=(26, 6))
    for ax, img, title in zip(axes,
        [processed, b, km, pf, sp],
        ["Pre-processed",
         "Baseline\n(resize only)",
         "K-Means\n(downsample→quantize)",
         "Palette-First\n(quantize→downsample)",
         "SLIC + Laplacian ✦\n(Gerstner 2012)"]):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.suptitle(f"Pixel Art — {pixel_size}×{pixel_size}, {n_colors} colours", fontsize=13)
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
    print("Saved comparison.png")

    return {
        "processed": processed,
        "baseline": b,
        "kmeans": km,
        "palette_first": pf,
        "slic": sp,
    }


def save_result(image, path):
    image.save(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    results = compare_results("input.jpg", pixel_size=70, n_colors=16)
    save_result(results["slic"], "output_slic.png")
    save_result(results["palette_first"], "output_palette_first.png")
