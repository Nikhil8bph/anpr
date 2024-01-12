from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import cv2
from utils import *
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import easyocr
import time
from utils import get_car, read_plate, wrap_transform_plate
import utils
from wrappodnet.wpodnet.model import WPODNet
import torch
from flask_socketio import SocketIO, emit

#load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
boolean_device = True if torch.cuda.is_available() else False
vehicles = [2, 3, 5, 7]
vehicle_detector = YOLO('./models/yolov8n.onnx',task='detect')
license_plate_detector = YOLO('./models/license_plate_detector.onnx',task='detect')
reader = easyocr.Reader(['en'], gpu=boolean_device)
wpodnet_model = WPODNet()
wpodnet_model.to(device)
wpodnet_checkpoint = torch.load('./models/wpodnet.pth')
wpodnet_model.load_state_dict(wpodnet_checkpoint)
frame_nmr = -1
video_path_ip = None
cap = None   
ret = False 

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('video_feed')
def handle_video_feed(image):
    if image is not None:
        img = convertBase64ToImage(image)
        # image_out = main.video_implementation_on_socket(True,img)
        emit('video_feed', convertImageToBase64(img), broadcast=True)

@socketio.on('video_feed_ip')
def handle_video_feed_ip(workMode):
    time.sleep(5)
    results = {}
    global ret
    while ret:
        global frame_nmr, cap
        frame_nmr += 1
        ret, frame = cap.read()
        if ret and workMode=="1":
            print("Hey This is True/False : ",workMode)
            results[frame_nmr] = {}
            # detect vehicles
            vehicle_detections_ = vehicle_detector.track(frame,classes=vehicles,persist=True)[0]
            plate_detections_ = license_plate_detector(frame)[0]
            for plate_detection in plate_detections_.boxes.data.tolist():
                vehicle_plate = get_car(plate_detection, vehicle_detections_)
                vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, vehicle_track_id, vehicle_score, vechile_class_id = vehicle_plate
                if vehicle_x1>0 or vehicle_y1>0 or vehicle_x2>0 or vehicle_y2>0:
                    plate_x1, plate_y1, plate_x2 ,plate_y2 ,plate_score ,plate_class_id = plate_detection
                    wrapped_plate = wrap_transform_plate(frame[int(vehicle_y1):int(vehicle_y2), int(vehicle_x1): int(vehicle_x2)], wpodnet_model)
                    plate_text, plate_text_score = read_plate(reader, wrapped_plate)
                    plate_org = (int(plate_x1), int(plate_y1) - 10)
                    cv2.rectangle(frame, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (0, 255, 0), 2)
                    cv2.putText(frame, plate_text, plate_org, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                    results[frame_nmr][vehicle_track_id] = {'car': {'bbox': [vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2]},
                                                            'license_plate': {'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                                                                        'text': plate_text,
                                                                        'bbox_score': plate_score,
                                                                        'text_score': plate_text_score},
                                                            }
            for vehicle_detection in vehicle_detections_.boxes.data.tolist():
                vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, vehicle_track_id, vehicle_score, vechile_class_id = vehicle_detection
                car_org = (int(vehicle_x1), int(vehicle_y1) - 10)
                cv2.putText(frame, ("vehicle id : "+str(vehicle_track_id)), car_org, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                cv2.rectangle(frame, (int(vehicle_x1), int(vehicle_y1)), (int(vehicle_x2), int(vehicle_y2)), (255, 0, 0), 2)    
            emit('video_feed_ip', convertImageToBase64(frame), broadcast=True)
    return 'Stop emitting'

@app.route('/updatesource', methods=['POST'])
def updateSource():
    global video_path_ip,cap, ret
    print(request)
    data1 = request.json.get('videoSource')
    data2 = request.json.get('fps')
    print(data1)
    print(data2)
    video_path_ip = data1
    ret = True
    cap = cv2.VideoCapture(video_path_ip)
    cap.set(cv2.CAP_PROP_FPS, int(data2))
    return f'Source Data Received : {video_path_ip}'

@app.route('/stopsource', methods=['GET'])
def stopSource():
    global video_path_ip,cap, ret
    video_path_ip = None
    ret = False
    cap = None
    return f'Source Stopped : {video_path_ip}'

@app.route('/initialize', methods=['POST'])
def initializeAnprPc():
    data = request.data.decode('utf-8')
    return f'Initialized ANPR'

if __name__ == '__main__':
    socketio.run(app, debug=True)
