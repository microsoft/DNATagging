# Testing
## Running tests
You can run tests via VSCode or from the command line. Install the DNATagging project locally with pip using ```pip install -e .```. To run from the command line, simply run ```pytest``` from the project root.

Configure VS Code to discover Pytest tests following instructions here: https://code.visualstudio.com/docs/python/testing. Tests can be annotated to run on specific images or image sets.

Use conftest.py to update variables used in testing including
* threshold: The luminesence decay value used to determine if a bit is 0 or 1 (default=0.38)
* max_errors: The number of bit errors allowed before failing the binary decoding test (default=3)
* replace_xn: Exclude spots on tickets which do not contain DNA reporters (default=True)