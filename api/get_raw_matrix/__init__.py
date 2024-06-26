import json
import logging

import azure.functions as func
import numpy as np
from api.common.data_utils import decimal_to_binary, string_to_matrix
from api.decode import bit_decoder, image_detector
from PIL import Image


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Get the raw signal decay matrix from the two provided images"""
    logging.info("get_raw_matrix is processing a request.")
    if req.files is None:
        return func.HttpResponse("No files found", status_code=400)

    before = req.files.get("before")
    after = req.files.get("after")

    if before is None or after is None:
        return func.HttpResponse("Images could not be read", status_code=400)

    before_np = np.array(Image.open(before))
    after_np = np.array(Image.open(after))

    before_grid = image_detector.get_brightness_grid(before_np)
    after_grid = image_detector.get_brightness_grid(after_np)
    diffs = bit_decoder.get_brightness_diffs(before_grid, after_grid)

    response_body = json.dumps(diffs.flatten().tolist())
    return func.HttpResponse(body=response_body)
