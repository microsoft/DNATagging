import os

import pytest
from numpy import genfromtxt

test_dir = os.path.dirname(os.path.realpath(__file__))
img_dir = os.path.join(test_dir, "images")

# This collects all subdirectories in the images directory
subdirs = next(os.walk(img_dir))[1]

class TestData:
    def __init__(self, test_name):
        self.test_name = test_name
        self.before_path = None
        self.after_path = None
        self.expected = None

    def load_data(self):
        img_paths = []
        csv = None
        test_path = os.path.join(img_dir, self.test_name)
        for file in os.listdir(test_path):
            path = os.path.join(test_path, file)
            if str.lower(file).endswith(".jpg"):
                img_paths.append(path)
            else:
                if "Codeword" in file:
                    csv = path
                elif "Decay" not in file and "Errors" not in file:
                    csv = path

        # rely on lexicographical ordering unless otherwise labeled
        self.before_path = img_paths[0]
        self.after_path = img_paths[1]

        # check if the images are labeled "before" and "after"
        lower = img_paths[0].lower()
        if "after" in lower:
            self.after_path = img_paths[0]
            self.before_path = img_paths[1]
        self.expected = genfromtxt(csv, delimiter=",", dtype=str)


@pytest.fixture(scope="session")
def get_test_data():
    def _get_test_data(test_name):
        test_data = TestData(test_name)
        test_data.load_data()
        return test_data
    return _get_test_data

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
