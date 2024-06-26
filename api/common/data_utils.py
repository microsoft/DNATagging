import os

import numpy as np

prefix = os.path.abspath(os.path.dirname(__file__))
decoder_dir = "decoder"
matrix_dir = "matrix"
# first 4msgs tested with 5/11/13, 5th tested with 13 only
seed = 13  # Also Tested with 11,13 (for generating error_location combos)
np.random.seed(seed)


def padColumnRand(matrix):
    # Adds a random bit (0/1) at the end of each row of matrix:
    new_column = np.random.randint(2, size=(matrix.shape[0], 1))
    new_mat = np.concatenate((matrix, new_column), axis=1)
    return new_mat


def getGenMatrix(length, parity_bits):
    """Get the generator matrix for this data size

    Args:
        length: total number of bits
        parity_bits: number of parity bits

    Returns:
        Matrix and filepath to resulting txt file
    """
    # Get generator matrix for ECC:
    generator_matrix_file = os.path.join(prefix, matrix_dir, "gen_matrix_24_12_8_golay_Apr22.txt")
    matrix = np.loadtxt(generator_matrix_file, dtype=str)
    fpath = os.path.join(prefix, matrix_dir, "gen_matrix_%i_%i.txt" % (length, parity_bits))

    matrix = convertTypeStrToInt(matrix)

    res = None
    if parity_bits == 12:
        if length == 24:
            res = matrix
        elif length == 25:
            res = padColumnRand(matrix)

    if res is not None:
        np.savetxt(fpath, res, fmt="%d", delimiter="")

    return res, fpath


def binary_to_decimal(bin):
    """
    convert binary into decimal
    :param bin: binary number
    :return: decimal number
    """
    return int(bin, 2)


def decimal_to_binary(num):
    """
    convert decimal into binary of length 12 bits
    :param num: decimal number
    :return: binary number
    """
    return f"{num:012b}"  # of length 12 bits


def string_to_matrix(s):
    """
    convert string matrix into int matrix
    :param s: string matrix
    :return: int matrix
    """
    array = []
    for bit in s:
        array.append(int(bit))
    return np.array([array])


def convertMatToStr(matrix):
    """
    convert matrix into a line of string
    :param matrix: matrix
    :return: a code string
    """
    return "".join(map(str, matrix[0]))


def convertTypeStrToInt(matrix):
    """
    convert string matrix into int matrix
    :param matrix: string matrix
    :return: int matrix
    """
    rows = [[int(bit) for bit in row] for row in matrix]
    return np.array(rows)
