import cv2 as cv
import matplotlib.pyplot as plt 
import numpy as np
import glob
import yaml
import pickle
import settings 

n_x = 9
n_y = 6
frameSize = (settings.ORIGINAL_SIZE[0], settings.ORIGINAL_SIZE[1]) 

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# setup object points
objp = np.zeros((n_y * n_x, 3), np.float32)
objp[:, :2] = np.mgrid[0:n_x, 0:n_y].T.reshape(-1, 2)

# Arrays to store objecy points and image points from all the images
image_points = [] # 3D point in real world space
object_points = [] # 2D point in image plane

source_path = "D:/TUGAS AKHIR DUTA/vehicle_distance"

images = [f for f in glob.glob(source_path+'/camera_cal_webcam/*.jpg')]

found = 0
# loop through provided images
for image in images :
    # print(image)
    img = cv.imread(image)
    img = cv.resize(img, frameSize)
    cv.imshow('img', img)
    cv.waitKey(1)
    print(image)
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
   
    ret, corners = cv.findChessboardCorners(img_gray, (n_x, n_y), None)
    
    if ret == True:
        # make fine adjustments to the corners so higher precision can be obtained before
        # appending them to the list
        object_points.append(objp)
        corners2 = cv.cornerSubPix(img_gray, corners, (11,11), (-1, -1), criteria)
        image_points.append(corners2)
        
        # Draw and display the corners 
        img = cv.drawChessboardCorners(img, (n_x, n_y), corners2, ret)
        found += 1
        cv.imshow('chessboard', img)
        cv.waitKey(0)

print("number of images used for calibration:", found)

# perform the calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(object_points, image_points, img_gray.shape[::-1], None, None)
img_size = img.shape
print("Camera Calibrated: ", ret)
print("\nCamera Matrix:\n", mtx)
print("\nDistortion Parameters:\n", dist)
print("\nRotation Vectors:\n", rvecs)
print("\nTranslation Vectors:\n", tvecs)

# Transformation the matrix distortion coefficients to writeable lists
data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeffs': np.asarray(dist).tolist()}
print(mtx)
print(dist)

# save data with yaml
with open("webcam_calibration_matrix.yaml", "w") as f:
    yaml.dump(data, f)

# pickle the data and save it to a file to used in perspective transform
calib_data = {'cam_matrix': mtx, 
              'dist_coeffs': dist,
              'img_size': img_size}
with open(settings.CALIBRATION_FILE_NAME_WEBCAM, "wb") as f:
    pickle.dump(calib_data, f)

# Undistortion
for image in images:
    print(image)
    img = cv.imread(image)
    h, w = img.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w,h))

    # Undistort
    dst = cv.undistort(img, mtx, dist, None, newCameraMatrix)
    cv.imshow('undistort', dst)
    cv.waitKey(1)

    # Crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    # cv.imwrite('CalResult1.png', dst)
    cv.imshow('undistort2', dst)
    cv.waitKey(1)

cv.destroyAllWindows()

# # undistort with Remapping
# mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newCameraMatrix, (w, h), 5)
# dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# # Crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('CalResult2.png', dst)


# # Reprojection Error
# mean_error = 0
# for i in range(len(object_points)):
#     image_points2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv2.norm(image_points[i], image_points2, cv2.NORM_L2)/len(image_points2)
#     mean_error += error

# print("\ntotal error: {}".format(mean_error/len(object_points)))
# print("\n\n\n")

# [[1.05410380e+03 0.00000000e+00 7.15218900e+02]
#  [0.00000000e+00 1.40857644e+03 5.52757505e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# [[-3.83539547e-01  9.40152828e-02  2.03238857e-04 -6.14793914e-04
#    2.97137693e-03]]



# [[1.40424392e+03 0.00000000e+00 9.52225559e+02]
#  [0.00000000e+00 1.40699390e+03 5.52804431e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# [[-3.83384815e-01  9.67820418e-02  9.38332445e-05 -5.57076132e-04
#   -1.35046384e-03]]