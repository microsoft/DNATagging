import json
import logging
import os

import azure.functions as func
import numpy as np
from api.common.data_utils import (
    convertTypeStrToInt,
    decimal_to_binary,
    matrix_dir,
    prefix,
    string_to_matrix,
)


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Encode the given farm id into a bit pattern with specified width and height"""
    logging.info("Encode received a request.")
    if req.form is None:
        return func.HttpResponse("No form found", status_code=400)
    farm_id = int(req.form["farm_id"])
    width = int(req.form["width"])
    height = int(req.form["height"])

    code_length = width * height
    if code_length != 24 and code_length != 25:
        return func.HttpResponse("Invalid code length", status_code=400)
    
    msg = encode(farm_id, code_length)
    res = convert_to_matrix(msg.tolist()[0], width)
    body = json.dumps(res)
    return func.HttpResponse(body, status_code=200)


def encode(farm_id, length):
    """
    encode farm Id into matrix
    :param farm_id: farm Id
    :param length: total bits of the resulting matrix
    :return:
    """
    bin_farm_id = string_to_matrix(decimal_to_binary(farm_id))
    matrix_file = os.path.join(prefix, matrix_dir, "gen_matrix_%i_12.txt" % length)
    matrix = convertTypeStrToInt(np.loadtxt(matrix_file, dtype=str))
    return np.matmul(bin_farm_id, matrix) % 2


def convert_to_matrix(encoding, width):
    """Format the matrix list into string with width elements per line

    Args:
        encoding: the data encoding
        width: the number of elements to place per row

    Returns:
        String copy of the encoding with width number of bits per line
    """
    i = 0
    res = ""
    str_lst = [str(x) for x in encoding]
    while i < len(str_lst):
        start = i
        end = min(i + width, len(str_lst))
        res += "".join(str_lst[start:end]) + "\n"
        i += width
    return res
