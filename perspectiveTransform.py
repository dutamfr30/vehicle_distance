import matplotlib.pyplot as plt
import matplotlib.image as pimg
import settings
import numpy as np
import cv2 as cv
import pickle
# import yaml

# image used to find the fanishing point 
straight_images = ["test_images/straight_lines1.jpg", "test_images/straight_lines2.jpg"]
roi_points = np.array([[0, settings.ORIGINAL_SIZE[1]-50], [settings.ORIGINAL_SIZE[0],settings.ORIGINAL_SIZE[1]-50], [settings.ORIGINAL_SIZE[0]//2,settings.ORIGINAL_SIZE[1]//2+50]], dtype=np.int32)
roi = np.zeros((settings.ORIGINAL_SIZE[1], settings.ORIGINAL_SIZE[0]), dtype=np.uint8)
cv.fillPoly(roi, [roi_points], 1)

with open(settings.CALIBRATION_FILE_NAME, 'rb') as f:
    calib_data = pickle.load(f)
    cam_matrix = calib_data["cam_matrix"]
    dist_coeffs = calib_data["dist_coeffs"]

Lhs = np.zeros((2,2), dtype=np.float32)
Rhs = np.zeros((2,1), dtype=np.float32)

for img_path in straight_images:
    img = pimg.imread(img_path)
    img = cv.undistort(img, cam_matrix, dist_coeffs)
    img_hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    edges = cv.Canny(img_hls[:, :, 1], 200, 100)
    lines = cv.HoughLinesP(edges*roi, 0.5, np.pi/180, 20, None, 180, 120)
    for line in lines:
        for x1, y1, x2, y2 in line:
            normal = np.array([[-(y2-y1)], [x2-x1]], dtype=np.float32)
            normal /= np.linalg.norm(normal)
            point = np.array([[x1],[y1]], dtype=np.float32)
            outer = np.matmul(normal, normal.T)
            Lhs += outer
            Rhs += np.matmul(outer, point)
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)

# Calculate the vanishing point 
vanishing_point = np.matmul(np.linalg.inv(Lhs), Rhs)
print("vp", vanishing_point[0])

top = vanishing_point[1] + 60
bottom = settings.ORIGINAL_SIZE[1]-35
width = 530
def on_line(p1, p2, ycoord):
    return [p1[0] + (p2[0]-p1[0])/float(p2[1]-p1[1])*(ycoord-p1[1]), ycoord]

# Define source and destination targets
p1 = [vanishing_point[0] - width/2, top]
p2 = [vanishing_point[0] + width/2, top]
p3 = on_line(p2, vanishing_point, bottom)
p4 = on_line(p1, vanishing_point, bottom)

print("p1", p1)
print("p2", p2)
print("p3", p3)
print("p4", p4)

src_points = np.array([[373.24634, 479.86014], [903.24634, 479.86014], [1809.2808, 685], [-532.788, 685]], dtype=np.float32)
dst_points = np.array([[0, 0], [settings.UNWRAPED_SIZE[0], 0],
                       [settings.UNWRAPED_SIZE[0], settings.UNWRAPED_SIZE[1]],
                       [0, settings.UNWRAPED_SIZE[1]]], dtype=np.float32)

print("src", src_points)
print("dst", dst_points)

# Draw the trapezoid
cv.polylines(img, [src_points.astype(np.int32)], True, (0, 0, 255), thickness=5)

# Find the projection matrix
M = cv.getPerspectiveTransform(src_points, dst_points)
min_wid = 1000

for img_path in straight_images:
    img = pimg.imread(img_path)
    img = cv.undistort(img, cam_matrix, dist_coeffs)
    img = cv.warpPerspective(img, M, settings.UNWRAPED_SIZE)
    img_hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    mask = img_hls[:, :, 1] > 128
    mask[:, :50] = 0
    mask[:, -50:] = 0
    mom = cv.moments(mask[:, :settings.UNWRAPED_SIZE[0]//2].astype(np.uint8))
    x1 = mom["m10"]/mom["m00"]
    mom = cv.moments(mask[:, settings.UNWRAPED_SIZE[0]//2:].astype(np.uint8))
    x2 = settings.UNWRAPED_SIZE[0]//2 + mom["m10"]/mom["m00"]
    cv.line(img, (int(x1), 0), (int(x1), settings.UNWRAPED_SIZE[1]), (255, 0, 0), 3)
    cv.line(img, (int(x2), 0), (int(x2), settings.UNWRAPED_SIZE[1]), (0, 0, 255), 3)
    if (x2-x1<min_wid):
        min_wid = x2-x1

meter_per_foot = 1/3.28084
pixel_per_meter_x = min_wid/(12* meter_per_foot)
Lh = np.linalg.inv(np.matmul(M, cam_matrix))
pixel_per_meter_y = pixel_per_meter_x * np.linalg.norm(Lh[:,0]) / np.linalg.norm(Lh[:,1])
print(pixel_per_meter_x, pixel_per_meter_y)

plt.imshow(img)
plt.show

perspective_data = {'perspective_transform': M,
                    'pixels_per_meter': (pixel_per_meter_x, pixel_per_meter_y),
                    'orig_points': src_points}

with open(settings.PERSPECTIVE_FILE_NAME, 'wb') as f:
    pickle.dump(perspective_data, f)

# with open("perspective_transform.yaml", 'w') as f:
#     yaml.dump(perspective_data, f)
