import torch
import cv2
import numpy as np
import settings
import pickle



model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
class_names = model.module.names if hasattr(model, 'module') else model.names
allowed_classes = ['car', 'motorcycle', 'bus', 'truck']



cap = cv2.VideoCapture(1)

with open(settings.CALIBRATION_FILE_NAME_WEBCAM, 'rb') as f:
    calib_data = pickle.load(f)
    cam_matrix = calib_data["cam_matrix"]
    dist_coeffs = calib_data["dist_coeffs"]

def POINTS(event, x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE :
        colorsBRG = [x, y]
        print(colorsBRG)

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', POINTS)

while True:
    ret, frame = cap.read()
    frame = cv2.undistort(frame, cam_matrix, dist_coeffs)
    frame = cv2.resize(frame, (1366, 768))
    results = model(frame)
    frame = np.squeeze(results.render())
    print(results.pandas().xyxy[0])
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        b = str(row['name'])
        
        bounding_box = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, b, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()