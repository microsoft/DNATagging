# Testing
## Pulling images
Test images are stored in Azure Blob Storage rather than in git. You will need permission to read and list images in the blob storage account to download them. You can pull these images by running ```python download_test_images.py```

This code with will create a new images/ directory in this folder and save the test images there. Images are expected to be in the ```<project_root>/test/images/``` directory.


## Running tests
You can run tests via VSCode or from the command line. Install the DNATagging project locally with pip using ```pip install -e .```. To run from the command line, simply run ```pytest``` from the project root.

Configure VS Code to discover Pytest tests following instructions here: https://code.visualstudio.com/docs/python/testing. Tests can be annotated to run on specific images or image sets.

Use conftest.py to update variables used in testing including
* threshold: The luminesence decay value used to determine if a bit is 0 or 1 (default=0.38)
* max_errors: The number of bit errors allowed before failing the binary decoding test (default=3)
* replace_xn: Exclude spots on tickets which do not contain DNA reporters (default=True)