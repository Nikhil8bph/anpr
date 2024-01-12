import torch
import cv2
from wrappodnet.wpodnet.backend import Predictor
from wrappodnet.wpodnet.model import WPODNet
from wrappodnet.wpodnet.stream import ImageStreamer
from PIL import Image
import numpy as np
import base64

def get_car(plate_detection,vehicle_detections_):
    plate_x1, plate_y1, plate_x2 ,plate_y2 ,plate_score ,plate_class_id = plate_detection
    for vehicle_detection in vehicle_detections_.boxes.data.tolist():
        vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, vehicle_track_id, vehicle_score, vechile_class_id = vehicle_detection
        if plate_x1 > vehicle_x1 and plate_y1 > vehicle_y1 and plate_x2 < vehicle_x2 and plate_y2 < vehicle_y2:
            return vehicle_detection 
    return -1, -1, -1, -1, -1, -1, -1

def read_plate(reader, license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        return text, score
    return "text_not_clear",-1

def wrap_transform_plate(source, wpodnet_model, scale = 1.0):
    cropped_image = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    predictor = Predictor(wpodnet_model)
    prediction = predictor.predict(Image.fromarray(cropped_image), scaling_ratio=scale)
    warped = prediction.warp()
    numpy_image = np.array(warped)
    warped_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return warped_image

def convertBase64ToImage(frame_data):
    # image_data = frame_data.decode('utf-8').replace('data:image/jpeg;base64,', '')
    image_data = frame_data.split(',')[1]
    # Decode the base64 string to bytes
    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def convertImageToBase64(image):
    # Encode the image as a base64 string
    retval, buffer = cv2.imencode('.jpg', image)
    bytes_image = base64.b64encode(buffer)
    base64_image = "data:image/jpeg;base64,"+str(bytes_image.decode('utf-8'))
    return base64_image