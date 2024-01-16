import cv2
import numpy as np
import time
from ultralytics import YOLO
import os
from installation_helper import (
    get_segmentation_mask,
    initialize_segmentation_model,
)
import webuiapi
import shutil
from datetime import datetime
from PIL import Image
from collections import deque
import flickr_api


######### Please Check These Important Parameters Before Running #########
STILL_DURATION = 3  # seconds to the audience stillness check
THRESHOLD = 6  # pixels to the audience stillne"ss check
FLICKR_KEY = "e133d51350fbf111787d8d0e7e833a57"
FLICKR_SECRET = "a762ce7ba5b6dd15"
HOST_IP = "136.38.105.217"  # The IP address of the Stable Diffusion server
HOST_PORT = 33681
USER_NAME = "qi"  # The username of the Stable Diffusion server
USER_PASSWORD = "10121012"  # The password of the Stable Diffusion server
DISTANCE_HISTORY_SIZE = 5
TRACKING_CONFIDENCE = 0.3

######################################################################
# Initialize a buffer to store the distance history
distance_history = deque(maxlen=DISTANCE_HISTORY_SIZE)

# Set environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Set flickr api
flickr_api.set_keys(api_key=FLICKR_KEY, api_secret=FLICKR_SECRET)
flickr_api.set_auth_handler("./flickr_auth.txt")
user = flickr_api.test.login()
photosets = user.getPhotosets()
# Load YOLOv8 model
model = YOLO("yolov8n.pt")


def move_images_with_timestamp(source_dir, dest_dir):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            # Construct the full file path
            source_file = os.path.join(source_dir, filename)
            # Add timestamp to the filename
            dest_file = os.path.join(dest_dir, f"{timestamp}_{filename}")
            # Move the file
            shutil.move(source_file, dest_file)


def on_closing():
    move_images_with_timestamp("./tmp_img", "./storage_img")


def detect_humans(frame):
    results = model.track(
        frame,
        persist=True,
        conf=TRACKING_CONFIDENCE,
        classes=[0],
        imgsz=320,
        max_det=1,
        verbose=False,
    )
    if results[0].boxes:
        box = results[0].boxes.xyxy[0].numpy()
        return box
    return None


def is_human_still(current_box, previous_box, threshold=THRESHOLD):
    if previous_box is None:
        return False

    center_current = (
        (current_box[0] + current_box[2]) / 2,
        (current_box[1] + current_box[3]) / 2,
    )
    center_previous = (
        (previous_box[0] + previous_box[2]) / 2,
        (previous_box[1] + previous_box[3]) / 2,
    )
    distance = np.sqrt(
        (center_current[0] - center_previous[0]) ** 2
        + (center_current[1] - center_previous[1]) ** 2
    )

    # Add the distance to the deque
    distance_history.append(distance)

    # Calculate the median of distances
    median_distance = np.median(list(distance_history))
    print(
        f"Median distance over last {DISTANCE_HISTORY_SIZE} frames: {median_distance}"
    )

    return median_distance < threshold


def cam_capture_loop():
    # Initialize MediaPipe Selfie Segmentation
    print("Initializing MediaPipe Selfie Segmentation...")
    selfie_segmentation = initialize_segmentation_model()
    # Initalize WebSD APi
    api = webuiapi.WebUIApi(host=HOST_IP, port=HOST_PORT)
    api.set_auth(USER_NAME, USER_PASSWORD)
    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera stream not available.")
        exit()

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"format": "XRGB8888", "size": (1920, 1080)}
        )
    )
    picam2.start()
    previous_box = None
    stillness_start_time = None
    stillness_duration = STILL_DURATION  # Time in seconds to check for stillness
    image_counter = 0

    try:
        while True:
            frame = picam2.capture_array()
            h, w, _ = frame.shape
            start = int((w - h) / 2)
            cropped_frame = frame[:, start : start + h]

            frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

            human_box = detect_humans(frame_rgb)
            if human_box is not None:
                if is_human_still(human_box, previous_box):
                    if stillness_start_time is None:
                        stillness_start_time = time.time()
                    elif time.time() - stillness_start_time >= stillness_duration:
                        img, mask_img = get_segmentation_mask(
                            selfie_segmentation, cropped_frame, image_counter
                        )
                        inpainting_result = api.img2img(
                            images=[img],
                            mask_image=mask_img,
                            inpainting_fill=1,
                            steps=40,
                            seed=3661460071,
                            prompt="lion,fur,animal,nudity,female",
                            cfg_scale=8.0,
                            denoising_strength=0.7,
                        )
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        temp_image_path = f"./tmp_img/{timestamp}_{image_counter}.jpg"
                        inpainting_result.image.save(temp_image_path)
                        uploaded_photo = flickr_api.upload(
                            photo_file=temp_image_path,
                            title=f"DEEPNUDE_{timestamp}_{image_counter}",
                            description=f"DEEPNUDE_{timestamp}",
                        )
                        photoset = flickr_api.Photoset(id="72177720314085382")
                        photoset.addPhoto(photo=uploaded_photo)
                        image_counter += 1
                        stillness_start_time = None
                else:
                    stillness_start_time = None

                    previous_box = human_box

    finally:
        cap.release()
        print("Closing camera stream...")


if __name__ == "__main__":
    cam_capture_loop()
