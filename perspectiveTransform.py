import matplotlib.pyplot as plt
import matplotlib.image as pimg
import settings
import numpy as np
import cv2 as cv
import pickle
import yaml

# image used to find the fanishing point 
straight_images = ["test_images/straight_lines1.jpg", "test_images/straight_lines2.jpg"]
# menentukan roi 
roi_points = np.array([[0, settings.ORIGINAL_SIZE[1]-50], [settings.ORIGINAL_SIZE[0],settings.ORIGINAL_SIZE[1]-50], [settings.ORIGINAL_SIZE[0]//2,settings.ORIGINAL_SIZE[1]//2+50]], dtype=np.int32)
roi = np.zeros((settings.ORIGINAL_SIZE[1], settings.ORIGINAL_SIZE[0]), dtype=np.uint8)
cv.fillPoly(roi, [roi_points], 1)

with open(settings.CALIBRATION_FILE_NAME_WEBCAM, 'rb') as f:
    calib_data = pickle.load(f)
    cam_matrix = calib_data["cam_matrix"]
    dist_coeffs = calib_data["dist_coeffs"]

# membuat matriks 2x2 dan diinisiasi dengan nilai nol 
Lhs = np.zeros((2,2), dtype=np.float32)
# membuat matriks 2x1 dan diinisiasi dengan nilai nol
Rhs = np.zeros((2,1), dtype=np.float32)

for img_path in straight_images:
    img = pimg.imread(img_path)
    img = cv.undistort(img, cam_matrix, dist_coeffs)
    img_hls = cv.cvtColor(img, cv.COLOR_RGB2HLS) # mengubah warna RGB ke HLS (Hue, Lightness, Saturation)
    edges = cv.Canny(img_hls[:, :, 1], 200, 100) # deteksi tepi algoritma canny pada saluran lightness dengan parameter batas atas dan bawah 200 dan 100
    lines = cv.HoughLinesP(edges*roi, 0.5, np.pi/180, 20, None, 180, 120) # transformasi hough untuk mendeteksi garis lurus pada gambar, dan dikembalikan daftar garis dalam bentuk array
    # melakukan iterasi pada setiap line yang terdeteksi
    for line in lines:
        for x1, y1, x2, y2 in line:
            normal = np.array([[-(y2-y1)], [x2-x1]], dtype=np.float32)
            normal /= np.linalg.norm(normal) # normalisasi vektor dengan membaginya dengan norma Euclidean
            point = np.array([[x1],[y1]], dtype=np.float32) # membentuk vektor titik yang melewati garis yang terdeteksi
            outer = np.matmul(normal, normal.T) # menghitung hasil perkalian matriks antara normal dan transposenya hasilnya matriks luar (outer product)
            Lhs += outer # mengakumulasi kontribusi dari setiap garis dalam mengestimasi garis yang lebih akurat
            Rhs += np.matmul(outer, point) # mengakumulasi kontribusi vektor point dalam estimasi garis
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2) # menggambar garis yang terdeteksi pada gambar

    cv.imshow('canny', img)
    if cv.waitKey(0) & 0xFF == ord('q'):
         break


# Calculate the vanishing point 
vanishing_point = np.matmul(np.linalg.inv(Lhs), Rhs) # menghitung invers matriks LHS dan menghasilkan vektor vanishing point
print("vp", vanishing_point[0])
print("vp(x, y)", vanishing_point)

top = vanishing_point[1] + 60 
bottom = np.array([settings.ORIGINAL_SIZE[1]-35], dtype=np.float32) # menghitung titik bawah dan atas dari gambar yang akan dibuat transformasi perspektif
width = 530 # lebar gambar yang akan digunakan untuk transformasi perspektif
# menghitung titik pada garis yang memiliki koordinat y tertentu
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

# src_points = np.array([[373.24634, 479.86014], [903.24634, 479.86014], [1809.2808, 685], [-532.788, 685]], dtype=np.float32)
src_points = np.array([p1, p2, p3, p4], dtype=np.float32)
dst_points = np.array([[0, 0], [settings.UNWARPED_SIZE[0], 0],
                       [settings.UNWARPED_SIZE[0], settings.UNWARPED_SIZE[1]],
                       [0, settings.UNWARPED_SIZE[1]]], dtype=np.float32)

print("src", src_points)
print("dst", dst_points)

# Draw the trapezoid
cv.polylines(img, [src_points.astype(np.int32)], True, (0, 0, 255), thickness=5)

# Find the projection matrix
M = cv.getPerspectiveTransform(src_points, dst_points) # menghitung matriks transformasi perspektif
min_wid = 1000 # lebar minimum dalam piksel 

print('matriks', M)

for img_path in straight_images:
    img = pimg.imread(img_path)
    img = cv.undistort(img, cam_matrix, dist_coeffs) # melakukan proses undistorsi
    img = cv.warpPerspective(img, M, settings.UNWARPED_SIZE) # melakukan transformasi perspektif pada gambar
    img_hls = cv.cvtColor(img, cv.COLOR_RGB2HLS) # mengubah warna gambar dari RGB ke HLS 
    mask = img_hls[:, :, 1] > 128 # membuat mask yang berisi piksel dg nilai channel L yang lebih besar dari 128
    mask[:, :50] = 0 # mengubah nilai piksel pada kolom pertama hingga kolom ke-50 menjadi 0
    mask[:, -50:] = 0 # mengubah nilai piksel pada kolom ke-50 dari belakang menjadi 0
    mom = cv.moments(mask[:, :settings.UNWARPED_SIZE[0]//2].astype(np.uint8)) # menghitung momen dari area mask pada setengah kiri gambar, area kolom hingga setengah lebar gambar
    x1 = mom["m10"]/mom["m00"] # menghitung koordinat x dari pusat area mask pada setengah kiri gambar
    mom = cv.moments(mask[:, settings.UNWARPED_SIZE[0]//2:].astype(np.uint8)) # menghitung momen dari area mask pada setengah kanan gambar, area kolom hingga setengah lebar gambar
    x2 = settings.UNWARPED_SIZE[0]//2 + mom["m10"]/mom["m00"] # menghitung koordinat x dari pusat area mask pada setengah kanan gambar
    cv.line(img, (int(x1), 0), (int(x1), settings.UNWARPED_SIZE[1]), (255, 0, 0), 3) # menggambar garis vertikal biru pada gambar
    cv.line(img, (int(x2), 0), (int(x2), settings.UNWARPED_SIZE[1]), (0, 0, 255), 3) # menggambar garis vertikal merah pada gambar
    # memeriksa selisih antara x1 dan x2, jika selisihnya lebih kecil dari min_wid, maka min_wid akan diubah menjadi selisih x1 dan x2
    if (x2-x1<min_wid):
        min_wid = x2-x1
    
    cv.imshow("img", img)
    if cv.waitKey(0) & 0xFF == ord('q'):
        break

meter_per_foot = 1/3.28084 # konversi meter ke foot
pixel_per_meter_x = min_wid/(12* meter_per_foot) # menghitung nilai pixel per meter pada sumbu x dengan membagi min_wid dengan 12 kaki dikalikan meter/foot
Lh = np.linalg.inv(np.matmul(M, cam_matrix)) # menghitung invers dari hasil perkalian matriks M dan cam_matrix
pixel_per_meter_y = pixel_per_meter_x * np.linalg.norm(Lh[:,0]) / np.linalg.norm(Lh[:,1]) # menghitung jumlah piksel per meter pada sumbu y dengan memperhitungkan rasio antara norma kolom pertama dan kedua dari matriks Lh
print(pixel_per_meter_x, pixel_per_meter_y)



plt.imshow(img)
plt.show



# perspective_data = {'perspective_transform': M,
#                     'pixels_per_meter': (pixel_per_meter_x, pixel_per_meter_y),
#                     'orig_points': src_points}

# with open(settings.PERSPECTIVE_FILE_NAME, 'wb') as f:
#     pickle.dump(perspective_data, f)

# data = {'perspective_transform': np.asarray(M).tolist(), 'pixels_per_meter': np.asarray((pixel_per_meter_x, pixel_per_meter_y)).tolist()}
# print('M', M)
# print('pixel_per_meter', (pixel_per_meter_x, pixel_per_meter_y))

# with open("perspective_transform.yaml", 'w') as f:
#     yaml.dump(data, f)
