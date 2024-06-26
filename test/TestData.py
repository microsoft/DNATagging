import os
from typing import Iterable

import numpy as np
from numpy import genfromtxt, ndarray


class TestData:
    def __init__(
        self,
        before_path: str,
        after_path: str,
        test_name: str,
        expected: ndarray,
        replace_xn: bool = True,
    ):
        self.before_path: str = before_path
        self.after_path: str = after_path
        self.test_name: str = test_name
        self.expected: ndarray = expected
        self.replace_xn = replace_xn


def get_image_data(img_dir, replace_xn) -> Iterable[TestData]:
    """Return the path of all jpg images located at any level under img_dir

    Args:
        img_dir: the directory to crawl for sets of files in the form
                 of (before.jpg, after.jpg, expected.csv)
        replace_xn: where or not to replace 'x' and 'n' characters in
                    the expected result with '0'

    Returns:
        Yields a single TestData which will lazily load data into memory
    """
    test_dir = os.path.dirname(os.path.realpath(__file__))
    img_dir = os.path.join(test_dir, img_dir)
    for dirpath, _, filenames in os.walk(img_dir):
        if not filenames:
            continue
        img_paths = []
        csv = None
        error_file = None
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if str.lower(filename).endswith(".jpg"):
                img_paths.append(path)
            else:
                if "Errors" in filename:
                    error_file = path
                elif "Decay" not in filename:
                    csv = path
        if len(img_paths) >= 2 and csv is not None and error_file is not None:
            # rely on lexicographical ordering unless otherwise labeled
            before = img_paths[0]
            after = img_paths[1]

            # check if the images are labeled "before" and "after"
            lower = img_paths[0].lower()
            if "after" in lower:
                after = img_paths[0]
                before = img_paths[1]
            expected = genfromtxt(csv, delimiter=",", dtype=str)
            yield TestData(before, after, dirpath.split(os.sep)[-1], expected, replace_xn)


def check_decoding(matrix, expected_values):
    """Determine the positions in which matrix and true_matrix differ

    Args:
        matrix: detected readout
        true_matrix: expected readout

    Returns:
        Positions of errors in matrix
    """
    diff_coord = []
    for i in range(len(matrix)):
        expected = str(expected_values[i])
        actual = str(matrix[i])
        if expected != actual:
            diff_coord.append({"index": i, "expected": expected, "actual": actual})
    return diff_coord


def correct_errors(expected, errors):
    row_vals = "ABCDE"
    col_vals = "12345"
    for error in errors:
        row = row_vals.index(error[0])
        col = col_vals.index(error[1])
        expected[row][col] = "0" if expected[row][col] == "1" else "1"
