def pytest_addoption(parser):
    parser.addoption("--max_errors", action="store", default=3, help="Max errors for a pass")
    parser.addoption("--threshold", action="store", default=0.38, help="Threshold value to use for binarizaion")
    parser.addoption("--replace_xn", action="store", default=True, help="Remove unused spots from analysis")

