from ultralytics import YOLO
import cv2
import easyocr

from sort.sort import *

#load models
results = {}
vehicles = [2, 3, 5, 7]
coco_model = YOLO('./models/yolov8n.onnx')
mot_tracker = Sort()
license_plate_detector = YOLO('./models/license_plate_detector.onnx')
reader = easyocr.Reader(['en'], gpu=False)

def read_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        print("text detected : "+text)
        return text, score
    return "text_not_clear",-1

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
        detections = coco_model(frame,classes=vehicles)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))
        
        for track in track_ids:
            carx1, cary1, carx2, cary2, car_id = track
            car_org = (int(carx1), int(cary1) - 10)
            car_cropped = frame[int(cary1):int(cary2), int(carx1): int(carx2), :]
            license_plates = license_plate_detector(car_cropped)[0]
            for license_plate in license_plates.boxes.data.tolist():
                platex1, platey1, platex2, platey2, platescore, plateclass_id = license_plate
                license_plate_crop = car_cropped[int(platey1):int(platey2), int(platex1): int(platex2)]
                plate_height = platey2 - platey1
                plate_width = platex2 - platex1
                plate_org = (int(carx1 + platex1), int(cary1 + platey1) - 10)
                # read license plate number
                license_plate_text, license_plate_text_score = read_plate(license_plate_crop)
                cv2.rectangle(frame, (int(carx1 + platex1), int(cary1 + platey1)), (int(carx1 + platex1 + plate_width), int(cary1 + platey1 + plate_height)), (0, 255, 0), 2)
                cv2.putText(frame, license_plate_text, plate_org, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                results[frame_nmr][car_id] = {'car': {'bbox': [carx1, cary1, carx2, cary2]},
                                                  'license_plate': {'bbox': [platex1, platey1, platex2, platey2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': platescore,
                                                                    'text_score': license_plate_text_score}}
            cv2.putText(frame, ("vehicle id : "+str(car_id)), car_org, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            cv2.rectangle(frame, (int(carx1), int(cary1)), (int(carx2), int(cary2)), (255, 0, 0), 2)
        cv2.imshow("YOLOv8 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # break
    else:
        break
