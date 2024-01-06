from ultralytics import YOLO
import cv2
import easyocr
from utils import get_car, read_plate, wrap_transform_plate
from wrappodnet.wpodnet.model import WPODNet
import torch

class ANPR:
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

    def __init__(self, video_path_ip=None):
        self.video_path_ip = video_path_ip
        if video_path_ip is not None:
            self.cap = cv2.VideoCapture(self.video_path_ip)
        
    def video_implementation_on_socket(self,ret_val,frame_data):
        # read frames    
        results = {}
        frame_nmr = -1
        while ret:
            frame_nmr += 1
            ret, frame = ret_val,frame_data
            if ret:
                results[frame_nmr] = {}
                # detect vehicles
                vehicle_detections_ = self.vehicle_detector.track(frame,classes=self.vehicles,persist=True,verbose=False)[0]
                plate_detections_ = self.license_plate_detector(frame, verbose=False)[0]
                for plate_detection in plate_detections_.boxes.data.tolist():
                    vehicle_plate = get_car(plate_detection, vehicle_detections_)
                    vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, vehicle_track_id, vehicle_score, vechile_class_id = vehicle_plate
                    if vehicle_x1>0 or vehicle_y1>0 or vehicle_x2>0 or vehicle_y2>0:
                        plate_x1, plate_y1, plate_x2 ,plate_y2 ,plate_score ,plate_class_id = plate_detection
                        wrapped_plate = wrap_transform_plate(frame[int(vehicle_y1):int(vehicle_y2), int(vehicle_x1): int(vehicle_x2)], self.wpodnet_model)
                        plate_text, plate_text_score = read_plate(self.reader, wrapped_plate)
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

    def video_implementation_on_ip(self):
        # read frames    
        results = {}
        frame_nmr = -1
        ret = True
        while ret:
            frame_nmr += 1
            ret, frame = self.cap.read()
            if ret:
                results[frame_nmr] = {}
                # detect vehicles
                vehicle_detections_ = self.vehicle_detector.track(frame,classes=self.vehicles,persist=True)[0]
                plate_detections_ = self.license_plate_detector(frame)[0]
                for plate_detection in plate_detections_.boxes.data.tolist():
                    vehicle_plate = get_car(plate_detection, vehicle_detections_)
                    vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, vehicle_track_id, vehicle_score, vechile_class_id = vehicle_plate
                    if vehicle_x1>0 or vehicle_y1>0 or vehicle_x2>0 or vehicle_y2>0:
                        plate_x1, plate_y1, plate_x2 ,plate_y2 ,plate_score ,plate_class_id = plate_detection
                        wrapped_plate = wrap_transform_plate(frame[int(vehicle_y1):int(vehicle_y2), int(vehicle_x1): int(vehicle_x2)], self.wpodnet_model)
                        plate_text, plate_text_score = read_plate(self.reader, wrapped_plate)
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
