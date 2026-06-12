# Pixel Art Generator with K-Means Color Quantization

Turn any image into pixel art — with cleaner, more consistent colors using machine learning.

## The Problem

Naively resizing an image to 50×50 pixels produces noisy, inconsistent color blocks that don't look like real pixel art. Each pixel ends up a slightly different shade, making the result look muddy rather than stylized.

## The Solution

By applying **K-Means clustering** after resizing, we reduce the entire image to a limited color palette (e.g. 32 colors). Each pixel is then replaced by its nearest cluster color, producing the clean, flat color blocks characteristic of pixel art.

## Methods Compared

| Method | How it works | Quality |
|---|---|---|
| Baseline | Resize to NxN, scale back up | Noisy colors |
| K-Means | Resize + cluster colors into palette | Clean blocks |
| Edge-Enhanced | Edge detection + K-Means | Sharper outlines |

## Usage

```bash
pip install -r requirements.txt
```

```python
from pixelate import compare_results, kmeans_pixelate, load_image, save_result

# Compare all methods side by side
compare_results("your_image.jpg", pixel_size=50, n_colors=32)

# Or just get the result
original = load_image("your_image.jpg")
result = kmeans_pixelate(original, pixel_size=50, n_colors=32)
save_result(result, "output.png")
```

## Parameters

- `pixel_size` — grid resolution (lower = more pixelated)
- `n_colors` — palette size (lower = more stylized, higher = more detailed)

## Example Results

| Original | Baseline | K-Means (32 colors) |
|---|---|---|
| *(your image)* | Noisy resize | Clean pixel art |

## Why K-Means?

K-Means is an unsupervised machine learning algorithm that groups similar colors together. Applied to image pixels, it finds the most representative colors in the image and maps every pixel to the nearest one — effectively learning a custom palette for each image.

## Future Ideas

- Custom palette input (e.g. GameBoy palette, NES palette)
- Dithering for smoother gradients
- Super-resolution upscaling after pixelation
