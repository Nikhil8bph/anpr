from ultralytics import YOLO
import cv2
import easyocr
from utils import get_car, read_plate
# from sort.sort import *

#load models
results = {}
vehicles = [2, 3, 5, 7]
vehicle_detector = YOLO('./models/yolov8n.onnx',task='detect')
# mot_tracker = Sort()
license_plate_detector = YOLO('./models/license_plate_detector.onnx',task='detect')
reader = easyocr.Reader(['en'], gpu=False)

# load video
cap = cv2.VideoCapture('./test-data/sample.mp4')

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        vehicle_detections_ = vehicle_detector.track(frame,classes=vehicles,persist=True)[0]
                   
        plate_detections_ = license_plate_detector(frame)[0]
        for plate_detection in plate_detections_.boxes.data.tolist():
            vehicle_plate = get_car(plate_detection, vehicle_detections_)
            vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, vehicle_track_id, vehicle_score, vechile_class_id = vehicle_plate
            if vehicle_x1>0 or vehicle_y1>0 or vehicle_x2>0 or vehicle_y2>0:
                plate_x1, plate_y1, plate_x2 ,plate_y2 ,plate_score ,plate_class_id = plate_detection
                license_plate_crop = frame[int(plate_y1):int(plate_y2), int(plate_x1): int(plate_x2)]
                plate_text, plate_text_score = read_plate(reader, license_plate_crop)
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

        cv2.imshow("YOLOv8 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # break
    else:
        break
