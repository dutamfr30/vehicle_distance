import os
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator
import cv2
import time
import numpy as np 
import pickle

from moviepy.editor import VideoFileClip
from settings import CALIBRATION_FILE_NAME, PERSPECTIVE_FILE_NAME, ORIGINAL_SIZE, UNWARPED_SIZE
from lane_finder import Lane_Finder


class DigitalFilter: 
    def __init__(self, vector, b, a):
        self.len = len(vector)
        self.b = b.reshape(-1, 1)
        self.a = a.reshape(-1, 1)
        self.input_history = np.tile(np.array(vector, dtype=np.float64), (len(self.b), 1))
        self.output_history = np.tile(np.array(vector, dtype=np.float64), (len(self.a), 1))
        self.old_output = np.copy(self.output_history[0])

    def output(self):
        return self.output_history[0]
    
    def speed(self):
        return self.output_history[0] - self.output_history[1]
    
    def new_point(self, vector):
        self.input_history = np.roll(self.input_history, 1, axis = 0)
        self.old_output = np.copy(self.output_history[0])
        self.output_history = np.roll(self.output_history, 1, axis=0)
        self.input_history[0] = vector
        self.output_history[0] = (np.matmul(self.b.T, self.input_history) - np.matmul(self.a[1:].T, self.output_history[1:]))/self.a[0]
        return self.output()
    
    def skip_one(self):
        self.new_point(self.output())

def area(bbox):
    return float((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))

class Car:
    def __init__(self, bounding_box, first=False, warped_size=None, transform_matrix=None, pixel_per_meter=None):
        self.warped_size = warped_size
        self.transform_matrix = transform_matrix
        self.pixel_per_meter = pixel_per_meter
        self.has_position = self.warped_size is not None \
                            and self.transform_matrix is not None \
                            and self.pixel_per_meter is not None
        self.filtered_bbox = DigitalFilter(bounding_box, 1/21*np.ones(21, dtype=np.float32), np.array([1.0, 0]))
        self.position = DigitalFilter(self.calculate_position(bounding_box), 1/21*np.ones(21, dtype=np.float32), np.array([1.0, 0]))
        self.found = True
        self.num_lost = 0
        self.num_found = 0
        self.display = first
        self.fps = 25

    def calculate_position(self, bbox):
        if (self.has_position):
            pos = np.array((bbox[0]/2+bbox[2]/2, bbox[3])).reshape(1, 1, -1)
            dst = cv2.perspectiveTransform(pos, self.transform_matrix).reshape(-1, 1)
            return np.array((self.warped_size[1]-dst[1])/self.pixel_per_meter)
        else:
            return np.array([0])
        
    def get_window(self):
        return self.filtered_bbox.output()
    
    def one_found(self):
        self.num_lost = 0
        if not self.display:
            self.num_found += 1
            if self.num_found > 5:
                self.display = True

    def one_lost(self):
        self.num_found = 0
        self.num_lost += 1
        if self.num_lost > 5:
            self.found = False

    def update_car(self, bboxes):
        current_window = self.filtered_bbox.output()
        intersection = np.zeros(4, dtype=np.float32)
        for idx, bbox in enumerate(bboxes):
            intersection[0:2] = np.maximum(current_window[0:2], bbox[0:2])
            intersection[2:4] = np.minimum(current_window[2:4], bbox[2:4])
            if (area(bbox) > 0) and area(current_window) and ((area(intersection)/area(current_window) > 0.8) or (area(intersection)/area(bbox) > 0.8)):
                self.one_found()
                self.filtered_bbox.new_point(bbox)
                self.position.new_point(self.calculate_position(bbox))
                bboxes.pop(idx)
                return
            
        self.one_lost()
        self.filtered_bbox.skip_one()
        self.position.skip_one()

    def draw(self, img, color=(0, 255, 0), thickness=2):
        if self.display:
            window = self.filtered_bbox.output().astype(np.uint32)
            cv2.rectangle(img, (window[0], window[1]), (window[2], window[3]), color, thickness)
            if self.has_position:
                cv2.putText(img, "RPos : {:6.2f}m".format(self.position.output()[0]), (int(window[0]), int(window[1]-5)),
                           cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=3, color=(255, 255, 255))    
                cv2.putText(img, "RPos : {:6.2f}m".format(self.position.output()[0]), (int(window[0]), int(window[1]-5)),
                           cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=2, color=(0, 0, 0))
                cv2.putText(img, "RVel: {:6.2f}km/h".format(self.position.speed()[0]*self.fps*3.6), (int(window[0]), int(window[3]+20)),
                           cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=3, color=(255, 255, 255))    
                cv2.putText(img, "RVel: {:6.2f}km/h".format(self.position.speed()[0]*self.fps*3.6), (int(window[0]), int(window[3]+20)),
                           cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=2, color=(0, 0, 0))   

class CarDetector:
    def __init__(self,  warped_size=None, transform_matrix=None, pixel_per_meter=None, cam_matrix=None, dist_coeffs=None):
        self.warped_size = warped_size
        self.transform_matrix = transform_matrix
        self.pixel_per_meter = pixel_per_meter
        self.cam_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs
        self.cars = []
        self.first = True
        

    # def POINTS(event, x,y,flags,param):
    #     if event == cv2.EVENT_MOUSEMOVE :
    #         colorsBRG = [x, y]
    #         print(colorsBRG)

    # cv2.namedWindow('YOLO V8')
    # cv2.setMouseCallback('YOLO V8', POINTS)

    def detect(self, img):
        car_windows = []
        frame = cv2.undistort(img, self.cam_matrix, self.dist_coeffs)
        results = model.predict(frame, classes=classes, conf=conf)  
        bboxes = [] 
        for r in results:
            # annotator = Annotator(frame)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                # c = box.cls
                bbox = np.array([np.int(b[0]), np.int(b[1]), np.int(b[2]), np.int(b[3])], dtype=np.int64)
                # annotator.box_label(b, model.names[int(c)], color=(0, 255, 0), txt_color=(255, 9, 9))
                print('b', b)
                # print('c', c)
                print('bbox', bbox)
                bboxes.append(bbox)

        print('bboxes', bboxes)
        # frame = annotator.result()
        cv2.imshow('YOLO V8', frame)

        for car in self.cars:
            car.update_car(bboxes)

        for bbox in bboxes:
            self.cars.append(Car(bbox, self.warped_size, self.pixel_per_meter, self.transform_matrix))

        tmp_cars = []
        for car in self.cars:
            if car.found:
                tmp_cars.append(car)
        self.cars = tmp_cars
        self.first = False

    def draw(self, img):
        i2 = np.copy(img)
        for car in self.cars:
            car.draw(i2)
        return i2


if __name__ == "__main__":
    mode = 'predict'
    model = YOLO('yolov8n.pt')
    conf = 0.50
    classes = 2,3,5,7
    
    cam = cv2.VideoCapture('challenge_video.mp4')
    video_files = ['challenge_video.mp4']
    output_path = 'output_videos'
    
    with open(CALIBRATION_FILE_NAME, 'rb') as f:
        calib_data = pickle.load(f)
        cam_matrix = calib_data["cam_matrix"]
        dist_coeffs = calib_data["dist_coeffs"]
        img_size = calib_data["img_size"]

    with open(PERSPECTIVE_FILE_NAME, 'rb') as f:
        perspective_data = pickle.load(f)
        perspective_transform = perspective_data["perspective_transform"]
        pixels_per_meter = perspective_data["pixels_per_meter"]
        orig_points = perspective_data["orig_points"]

    def process_image(img, car_detector, lane_finder, cam_matrix, dist_coeffs, transform_matrix, pixel_per_meter, reset=False):
        img = cv2.undistort(img, cam_matrix, dist_coeffs)
        car_detector.detect(img)
        lane_finder.find_lane(img, distorted=False, reset=reset)
        return lane_finder.draw_lane_weighted(car_detector.draw(img))

    for file in video_files:
        lf = Lane_Finder(img_size=ORIGINAL_SIZE, warped_size=UNWARPED_SIZE, cam_matrix=cam_matrix, dist_coeffs=dist_coeffs, 
                         transform_matrix=perspective_transform, pixels_per_meter=pixels_per_meter, warning_icon='warning.png')
        cd = CarDetector(warped_size=UNWARPED_SIZE, transform_matrix=perspective_transform, 
                         pixel_per_meter=pixels_per_meter, cam_matrix=cam_matrix, 
                         dist_coeffs=dist_coeffs)
        video_capture = cv2.VideoCapture(file)
        while True:
            ret, image = video_capture.read()
            if not ret:
                break
            # process_image(image, cd, lf, cam_matrix, dist_coeffs, perspective_transform, pixels_per_meter)
            output = process_image(image, cd, lf, cam_matrix, dist_coeffs, perspective_transform, pixels_per_meter)
            cv2.imshow('YOLO V8', process_image(image, cd, lf, cam_matrix, dist_coeffs, perspective_transform, pixels_per_meter))
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()
        # output = os.path.join(output_path, "detect_cars_"+file)
        # clip2 = VideoFileClip(file)
        # challenge_clip = clip2.fl_image(lambda x: process_image(x, cd, lf, cam_matrix, dist_coeffs, perspective_transform, pixels_per_meter))
        # challenge_clip.write_videofile(output, audio=False)


        # bbox [1056  389 1279  527]
# box [800 360 959 519]
# bbox [1056  389 1279  527]
# bboxes [array([800, 360, 959, 496], dtype=int64), array([1056,  389, 1279,  523], dtype=int64)]