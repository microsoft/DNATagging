import logging
from test.TestData import check_decoding, get_test_data, subdirs

import numpy as np
import pytest
from api.decode import bit_decoder, image_detector
from PIL import Image

"""
Tests to verify the image to bit string decoding algorithm works given a before and after image.
"""
@pytest.fixture
def max_errors(request):
    return request.config.getoption("--max_errors")

@pytest.fixture
def threshold(request):
    return request.config.getoption("--threshold")

@pytest.fixture
def replace_xn(request):
    return request.config.getoption("--replace_xn")

@pytest.mark.parametrize('test_case', subdirs)
def test_image_decoding(get_test_data, test_case, max_errors, threshold, replace_xn):
    """
    Test whether the algorithm is able to generate a correct readout
    for each pair of images. Crawls folder name for all subdirs that
    contain pairs of jpg images. Outputs debug data from intermediate steps.
    """
    test_data = get_test_data(test_case)

    before_np = np.array(Image.open(test_data.before_path))
    after_np = np.array(Image.open(test_data.after_path))
    
    before_brightness = image_detector.get_brightness_grid(before_np)
    after_brightness = image_detector.get_brightness_grid(after_np)
    expected = test_data.expected
    if replace_xn:
        before_brightness, after_brightness, expected = remove_xn(before_brightness, after_brightness, expected)
    matrix = bit_decoder.get_brightness_diffs(before_brightness, after_brightness)
    output = np.where(np.asarray(matrix) > threshold, 1, 0)
    errors = check_decoding(output, expected)
    logging.error(f"Test case: {test_data.test_name}")
    logging.error(f"Num errors: {len(errors)}")
    logging.error(f"Errors: {errors}")
    logging.error(f"Computed: {output}")
    logging.error(f"Expected: {expected}")
    logging.error(f"Before brightness: \n{before_brightness}")
    logging.error(f"After brightness: \n{after_brightness}")
    logging.error(f"Normalized diffs: \n{matrix}")

    assert len(errors) <= max_errors


def remove_xn(before_brightness, after_brightness, expected):
    """Remove spots that do not contain any DNA reporter from the analysis;
    these spots exhibit a different fluorescence decay than spots with DNA reporters."""
    mask = np.char.isnumeric(expected)
    return before_brightness[mask], after_brightness[mask], expected[mask]