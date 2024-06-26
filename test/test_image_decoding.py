import logging
from test.TestData import TestData, check_decoding, get_image_data

import numpy as np
import pytest
from api.decode import bit_decoder, image_detector
from PIL import Image

"""
Tests to verify the image to bit string decoding algorithm works given a before and after image.
"""

def get_max_errors(request):
    return int(request.config.getoption("--max_errors"))


def get_threshold(request):
    return float(request.config.getoption("--threshold"))


def pytest_generate_tests(metafunc):
    """Generates test cases from all subdirs of <project_root>/test/images"""
    if "test_case" in metafunc.fixturenames:
        test_case_generator = get_image_data("images", metafunc.config.getoption("--replace_xn"))
        metafunc.parametrize("test_case", test_case_generator)


def test_image_decoding(test_case: TestData, request):
    """
    Test whether the algorithm is able to generate a correct readout
    for each pair of images. Crawls folder name for all subdirs that
    contain pairs of jpg images. Outputs debug data from intermediate steps.
    """
    max_errors = get_max_errors(request)
    threshold = get_threshold(request)

    before_np = np.array(Image.open(test_case.before_path))
    after_np = np.array(Image.open(test_case.after_path))
    
    before_brightness = image_detector.get_brightness_grid(before_np)
    after_brightness = image_detector.get_brightness_grid(after_np)
    expected = test_case.expected
    if test_case.replace_xn:
        before_brightness, after_brightness, expected = remove_xn(before_brightness, after_brightness, expected)
    matrix = bit_decoder.get_brightness_diffs(before_brightness, after_brightness)
    output = np.where(np.asarray(matrix) > threshold, 1, 0)
    errors = check_decoding(output, expected)
    logging.error(f"Test case: {test_case.test_name}")
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