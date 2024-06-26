from typing import Any

from api.decode.image_detector import get_brightness_grid
import numpy as np
from numpy.typing import NDArray

# How large of a difference is required to determine a 0 vs 1 bit
BRIGHTNESS_THRESHOLD = 0.38

def decode(before_img: NDArray[Any], after_img: NDArray[Any]) -> NDArray[Any]:
    """Decode the image pair into a sequence of bits"""
    before_grid = get_brightness_grid(before_img)
    after_grid = get_brightness_grid(after_img)
    normalized_brightness_diffs = get_brightness_diffs(before_grid, after_grid)
    return np.where(normalized_brightness_diffs > BRIGHTNESS_THRESHOLD, 1, 0)


def get_brightness_diffs(before_grid: NDArray[Any], after_grid: NDArray[Any]) -> NDArray[Any]:
    """Interpret spots from pairs images to a normalized luminescence decay matrix
    
    Args:
        before_img: grid of avg brightness of spots on reporter ticket before reaction
        after_img: grid of avg brightness of spots on reporter ticket after reaction
    
    Returns:
        A matrix of normalized brightness differences between reporter spots in the two images
    """
    #Convert images to brightness grids for each reporter spot
    invert = False
    diffs = before_grid - after_grid
    if diffs.min() < 0:
        # after is brighter than before
        diffs = diffs - diffs.min()
        invert = True

    # Normalize the differences to a range of [0, 1]
    diffs = (diffs - diffs.min()) / (diffs.max() - diffs.min())

    if invert:
        diffs = 1 - diffs

    np.round(diffs, 2, diffs)
    return diffs
