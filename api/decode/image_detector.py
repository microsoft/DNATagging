from typing import Any, List, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.vq import kmeans
from skimage.measure import label, regionprops
from skimage.measure._regionprops import RegionProperties
from skimage.morphology import closing, diameter_opening, opening, remove_small_holes
from skimage.transform import AffineTransform

GRID_SIZE = 5


def dumb_resize(img: NDArray[Any], size: int):
    """
    Quick and dirty strided resize to get a (very) roughly constant image size
    """
    # Maybe resize it in a smarter way
    min_size = min(img.shape[:2])
    stride = max(round(min_size / size), 1)
    return img[::stride, ::stride].copy(), stride


def dumb_grayscale(img: NDArray[Any]):
    assert len(img.shape) in (2, 3), "Expecting RGB or Grayscale image"
    if len(img.shape) == 3:
        return img.mean(axis=-1)
    return img


def get_capture_area(img: NDArray[Any], noise_fp: int):
    u = np.quantile(img.reshape(-1), 0.5)
    ymin, xmin, ymax, xmax = sorted(
        regionprops(label(diameter_opening(remove_small_holes(img > u), noise_fp))),
        key=lambda x: x.area,
    )[-1].bbox
    return xmin, ymin, xmax, ymax


def get_roi(img: NDArray[Any], noise_fp: int) -> Tuple[int, int, int, int, AffineTransform]:
    """
    Get region of interest that has the circles
    """
    q = max(np.quantile(img.reshape(-1), 0.1).item(), 3)
    prep_ar = opening(remove_small_holes(img < q), np.ones((noise_fp, noise_fp)))
    region = sorted(
        [
            p
            for p in regionprops(label(prep_ar))
            if 0.8 < p.major_axis_length / p.minor_axis_length < 1 / 0.8
        ],
        key=lambda x: x.area,
    )[-1]
    h, w = img.shape
    corners = (
        np.abs(np.array([[0, 0], [0, w], [h, 0], [h, w]])[:, None] - region.coords)
        .sum(axis=-1)
        .argmin(axis=1)
    )
    corners = region.coords[corners]
    tr = AffineTransform()
    assert tr.estimate(
        corners[:, ::-1], np.array([[0, 0], [512, 0], [0, 512], [512, 512]])
    ), "Failed to get RoI transform"

    ymin, xmin, ymax, xmax = region.bbox
    return xmin, ymin, xmax, ymax, tr


def label_circles(
    img: NDArray[Any], thr: int, noise_fp: int, axis_ratio: float, min_area: float
) -> List[RegionProperties]:
    """
    Extract circle regions in the image
    """
    fp = np.ones((noise_fp, noise_fp))
    labels = label(closing(opening(img > thr, fp), fp), background=0, connectivity=2)
    regions = [
        prop
        for prop in regionprops(labels, img)
        if axis_ratio < prop.major_axis_length / prop.minor_axis_length < 1 / axis_ratio
        and prop.area > min_area
        and 0.8
        < prop.area / (np.pi * np.square((prop.major_axis_length + prop.minor_axis_length) / 4))
        < 1.2
    ]
    return regions


def brightness_grid(
    regions: List[RegionProperties], tr: AffineTransform, thr: int, refinement_ratio: float
) -> NDArray[Any]:
    """
    Compute average brightness for each circle after trying to refine the region a little bit
    """
    avg_brightness = []
    for region in regions:
        # Try to refine the circle by eliminating regions with brightness lower
        # than 1/3 of the median
        mask = (
            region.intensity_image
            > np.median(region.intensity_image[region.intensity_image > thr]) * refinement_ratio
        )
        avg_brightness.append(region.intensity_image[mask].mean())

    centroids = np.array([p.centroid for p in regions])
    tr2 = AffineTransform(rotation=tr.rotation)
    rot_centroids = tr2(centroids[:, ::-1])[:, ::-1]
    # Use kmeans to do scalar quantization and find row and column positions
    row, col = [
        np.abs(c - kmeans(c, np.linspace(c.min(), c.max(), GRID_SIZE))[0][:, None]).argmin(axis=0)
        for c in rot_centroids.T
    ]
    brightness_mat = np.zeros((GRID_SIZE, GRID_SIZE))
    brightness_mat[row, col] = avg_brightness
    return brightness_mat


def get_brightness_grid(
    image: NDArray,
    size: int = 512,
    noise_fp: int = 5,
    thr: int = 3,
    axis_ratio: float = 1.5,
    min_area: float = 400,
    refinement_ratio: float = 1 / 3,
    plot_things: bool = False,
):
    """
    Open image and extract brightness grid
    """
    assert axis_ratio > 0, "axis_ratio should be positive"
    if axis_ratio > 1:  # We always pass the minimum ratio
        axis_ratio = 1 / axis_ratio
    # ar = io.imread(filepath)
    gray = dumb_grayscale(image)
    small, stride = dumb_resize(gray, size)
    # 1. Crop to capture area
    xmin, ymin, xmax, ymax = map(lambda x: x * stride, get_capture_area(small, noise_fp))
    # 2. Crop to tag
    cap, stride = dumb_resize(gray[ymin:ymax, xmin:xmax], size)
    xmin2, ymin2, xmax2, ymax2, tr = map(
        lambda x: x * stride if isinstance(x, int) else x, get_roi(cap, noise_fp)
    )
    tr = cast(AffineTransform, tr)
    ymin2 += ymin
    ymax2 += ymin
    xmin2 += xmin
    xmax2 += xmin
    crop, _ = dumb_resize(gray[ymin2:ymax2, xmin2:xmax2], size)
    # Scale min area to image size
    adjusted_area = np.prod([s / size for s in crop.shape]) * min_area
    # Get circle regions
    regions = label_circles(crop, thr, noise_fp, axis_ratio, float(adjusted_area))
    if not regions:
        raise ValueError("Did not detect any circles")
    if plot_things:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(gray)
        plt.plot([xmin2, xmax2, xmax2, xmin2, xmin2], [ymin2, ymin2, ymax2, ymax2, ymin2], "r")
        plt.subplot(1, 2, 2)
        plt.imshow(crop)
        centroids = np.array([r.centroid for r in regions])
        plt.plot(*centroids.T[::-1], "ro")
        plt.show()
    # Compute brightness matrix
    return brightness_grid(regions, tr, thr, refinement_ratio)
