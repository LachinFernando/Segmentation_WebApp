import streamlit as st
import base64
import requests
import json
from PIL import Image
from io import BytesIO


ENDPOINT_URL = "http://localhost:5000/inference"
IMAGE_NAME = "uploaded.png"


def bytes_to_image(data):
    """
    Get the base64 encoded data of an image and
    convert it to a bytearray
    """
    image = base64.b64decode(data)
    payload = bytearray(image)
    stream = BytesIO(payload)
    img = Image.open(stream)
    return img


def convert_base64(path: str):
    """
    convert the image into base64 encoded format and
    create the payload structure
    """
    with open(path, 'rb') as image_file:
        payload = base64.b64encode(image_file.read())
    
    payload_structure = {
        "body": {
            "image": payload.decode('utf-8')
        }
    }

    return payload_structure
    

def get_mask(path: str, url: str = ENDPOINT_URL):
    """
    send a post REST api request to flask backend
    """
    image_payload = convert_base64(path)
    headers = {'Content-Type': 'application/json'}
    endpoint_response = requests.post(url, data = json.dumps(image_payload), headers = headers)
    response = getattr(endpoint_response, '_content').decode("utf-8")
    final_response = json.loads(response)

    return final_response["data"]["prediction"]


st.title("Segmentation Web App")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Open the uploaded file using PIL
    image = Image.open(uploaded_file)

    # Display the image
    image.save(IMAGE_NAME)

    # get predictions
    with st.spinner("Getting Predictions......"):
        mask_response = get_mask(IMAGE_NAME)
        mask = bytes_to_image(mask_response)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image)

        with col2:
            st.subheader("Mask")
            st.image(mask)
