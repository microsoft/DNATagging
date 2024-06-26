import json
import mimetypes
import urllib.parse
import uuid
from codecs import encode
from test.TestData import get_image_data

from api import decode as dna_decode
from api import encode as dna_encode
from api import get_raw_matrix
from azure.functions import HttpRequest

"""
Tests to ensure API endpoints are functioning as expected.
"""

def test_dna_encode():
    """Test that the encode function returns a valid bit string and
    the decoder can retrieve the original id"""
    input_id = 1234
    width = 5
    height = 5
    
    form_data = {
        "farm_id": input_id,
        "width": width,
        "height": height,
    }
    form_data_encoded = urllib.parse.urlencode(form_data)
    body_bytes = encode(form_data_encoded, "utf-8")

    api_request = HttpRequest(
        method="POST",
        url="/api/encode",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        body=body_bytes,
    )
    response = dna_encode.main(api_request)

    assert response.status_code == 200
    # ensure response is a binary string of the correct length
    response_content = json.loads(response.get_body())
    binary_string = response_content.replace("\n", "")
    assert len(binary_string) == width * height
    assert set(binary_string) <= {"0", "1"}
    # ensure the data can be decoded back to the original id
    response_id = dna_decode.decode_binary_string(binary_string)
    assert response_id == input_id


def test_dna_decode():
    """Test that the response for a query is successful and returns an int id"""
    test_case = next(get_image_data("images", False))
    headers, body = generate_payload(test_case)
    api_request = HttpRequest(
        method="POST",
        url="/api/decode",
        headers=headers,
        body=body,
    )
    response = dna_decode.main(api_request)

    assert response.status_code == 200
    response_content = json.loads(response.get_body())
    # ids are stored as ints
    assert isinstance(response_content, int)

def test_get_raw_matrix():
    """Test that the response for a query is successful and returns a list of floats.
    This test checks that the endpoint is functional, not that the response is correct"""
    test_case = next(get_image_data("images", False))
    headers, body = generate_payload(test_case)
    api_request = HttpRequest(
        method="POST",
        url="/api/get_raw_matrix",
        headers=headers,
        body=body,
    )
    response = get_raw_matrix.main(api_request)

    assert response.status_code == 200
    response_content = json.loads(response.get_body())
    assert isinstance(response_content, list)
    assert all(isinstance(x, float) for x in response_content)


def generate_payload(test_case):
    """Generate the headers and body for an http request using this test case"""
    dataList = []
    boundary = str(uuid.uuid4())
    add_file = add_file_context(dataList, boundary)
    add_file(test_case.before_path, "before")
    add_file(test_case.after_path, "after")

    dataList.append(encode("--" + boundary + "--"))
    dataList.append(encode(""))
    body = b"\r\n".join(dataList)
    headers = {"Content-type": f"multipart/form-data; boundary={boundary}"}
    return headers, body


def add_file_context(dataList, boundary):
    """Returns a function to add files to the existing dataList using the specified boundary

    Args:
        dataList: list of data to add as payload to http request
        boundary: data boundary to use
    """

    def add_file_helper(path, name):
        """Adds the file at path with the specified name to the dataList

        Args:
            path: path to file which should be added to request
            name: name to give file in request
        """
        dataList.append(encode("--" + boundary))
        dataList.append(encode(f"Content-Disposition: form-data; name={name}; filename={path}"))

        fileType = mimetypes.guess_type(path)[0] or "application/octet-stream"
        dataList.append(encode(f"Content-Type: {fileType}"))
        dataList.append(encode(""))

        with open(path, "rb") as f:
            dataList.append(f.read())

    return add_file_helper