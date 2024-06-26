import json
import logging
import os
import stat
from subprocess import check_output

import azure.functions as func
import numpy as np
from api.common.data_utils import binary_to_decimal, decoder_dir, getGenMatrix, prefix
from api.decode import bit_decoder
from PIL import Image


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Decode data based on two input images: before and after reaction

    Returns:
        An Http response object with the decoded data
    """
    logging.info("classify is processing a request.")
    if req.files is None:
        return func.HttpResponse("No files found", status_code=400)

    before = req.files.get("before")
    after = req.files.get("after")

    if before is None or after is None:
        return func.HttpResponse("Images could not be read", status_code=400)

    before_np = np.array(Image.open(before))
    after_np = np.array(Image.open(after))

    bit_str = decode_image_pair(before_np, after_np)
    decoded = decode_binary_string(bit_str)
    response_body = json.dumps(decoded)
    return func.HttpResponse(body=response_body)


def decode_image_pair(before, after):
    """Resolve a pair of images to a binary string"""
    output = bit_decoder.decode(before, after)
    return "".join(str(bit) for bit in output.flatten())


def decode_binary_string(code_msg):
    """
    Decode a binary code message into farm id
    :param code_msg: 01 matrix
    :return: decoded farm Id
    """
    length = len(code_msg)
    matrix, fpath = getGenMatrix(length, 12)  # TODO: k now is fixed 12
    print("length", length)
    if matrix is None:
        return "unsupported length of the code message"

    if length == 24:
        decoder_file = "decoder24-linux"
    else:  # length == 25:
        decoder_file = "decoder25-linux"
    decoder_path = os.path.join(prefix, decoder_dir, decoder_file)

    st = os.stat(decoder_path)
    os.chmod(decoder_path, st.st_mode | stat.S_IEXEC)
    result = check_output([decoder_path, code_msg, fpath]).decode("utf-8")
    json_result = result[result.find("{") :]
    result = json.loads(json_result)
    return binary_to_decimal(result["message"])
