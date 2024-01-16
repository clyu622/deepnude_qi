import requests
import json
from io import BytesIO
from PIL import Image
import base64
import cv2
import mediapipe as mp
import numpy as np
import os


def initialize_segmentation_model():
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    return selfie_segmentation


def get_segmentation_mask(mp, image, index):
    # Load and resize the image
    original_image = image
    resized_image = cv2.resize(original_image, (1024, 1024))

    # Process the image for segmentation
    results = mp.process(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

    # Create the mask image from segmentation results
    mask_image = (results.segmentation_mask > 0.5) * 255
    mask_image = mask_image.astype(np.uint8)
    # mask_image_rgb = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)
    mask_image = Image.fromarray(mask_image)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    resized_image = Image.fromarray(resized_image)
    mask_image.save(f"./tmp_img/mask_image_{index}.jpg")
    resized_image.save(f"./tmp_img/original_image_{index}.jpg")
    print(f"Saving the mask image to ./tmp_img/mask_image_{index}.jpg")
    print(f"Saving the original image to./tmp_img/original_image_{index}.jpg")

    return resized_image, mask_image


# if __name__ == "__main__":
#     mp_selfie_segmentation = mp.solutions.selfie_segmentation
#     selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
#     image = cv2.imread("./test_img/IMG_0712.jpg")
#     get_segmentation_mask(selfie_segmentation, image, 1)
