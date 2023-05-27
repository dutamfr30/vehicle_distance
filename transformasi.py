import cv2
import numpy as np
import settings
import pickle
import matplotlib.image as pimg
import matplotlib.pyplot as plt

vidcap = cv2.VideoCapture('challenge_video.mp4')
video_files = ['challenge_video.mp4']
straight_images = ["test_images/straight_lines1.jpg", "test_images/straight_lines2.jpg"]
success,image = vidcap.read()
roi_points = np.array([[0, settings.ORIGINAL_SIZE[1]-50], [settings.ORIGINAL_SIZE[0],settings.ORIGINAL_SIZE[1]-50], [settings.ORIGINAL_SIZE[0]//2,settings.ORIGINAL_SIZE[1]//2+50]], dtype=np.int32)
roi = np.zeros((settings.ORIGINAL_SIZE[1], settings.ORIGINAL_SIZE[0]), dtype=np.uint8)
cv2.fillPoly(roi, [roi_points], 1)

with open(settings.CALIBRATION_FILE_NAME, 'rb') as f:
    calib_data = pickle.load(f)
    cam_matrix = calib_data["cam_matrix"]
    dist_coeffs = calib_data["dist_coeffs"]

Lhs = np.zeros((2,2), dtype=np.float32)
Rhs = np.zeros((2,1), dtype=np.float32)

while success:
    success,image = vidcap.read()
    frame = cv2.resize(image, (1280, 720))
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    edges = cv2.Canny(frame[:, :, 1], 200, 100)
    lines = cv2.HoughLinesP(edges*roi, 0.5, np.pi/180, 20, None, 180, 120)
    # print('lines', lines)

    for line in lines:
        for x1, y1, x2, y2 in line:
            normal = np.array([[-(y2-y1)], [x2-x1]], dtype=np.float32)
            normal /= np.linalg.norm(normal)
            point = np.array([[x1], [y1]], dtype=np.float32)
            outer = np.matmul(normal, normal.T)
            Lhs += outer
            Rhs += np.matmul(outer, point)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            # print('normal', normal)
            # print('point', point)
            # print('outer', outer)

    vanishing_point = np.matmul(np.linalg.inv(Lhs), Rhs) # Mengalikan matriks invers Lhs dengan Rhs
    print('vanishing_point', vanishing_point)
    titik_hilang = (int(vanishing_point[0]), int(vanishing_point[1]))
    # print('Lhs', Lhs)
    # print('Rhs', Rhs)

    cv2.circle(frame, titik_hilang, 5, (0, 255, 0), -1)

    top = vanishing_point[1] + 60
    bottom = np.array([settings.ORIGINAL_SIZE[1]-35], dtype=np.float32)
    width = 530

    def on_line(p1, p2, ycoord): 
        return [p1[0] + (p2[0]-p1[0])/float(p2[1]-p1[1])*(ycoord-p1[1]), ycoord]
    
    p1 = [vanishing_point[0] - width/2, top]
    p2 = [vanishing_point[0] + width/2, top]
    p3 = on_line(p2, vanishing_point, bottom)
    p4 = on_line(p1, vanishing_point, bottom)

    print("p1", p1)
    print("p2", p2)
    print("p3", p3)
    print("p4", p4)

    src_points = np.array([p1, p2, p3, p4], dtype=np.float32)
    dst_points = np.array([[0, 0], [settings.UNWARPED_SIZE[0], 0],
                        [settings.UNWARPED_SIZE[0], settings.UNWARPED_SIZE[1]],
                        [0, settings.UNWARPED_SIZE[1]]], dtype=np.float32)

    print("src", src_points)
    print("dst", dst_points)
 
    # Draw the trapezoid
    cv2.polylines(frame, [src_points.astype(np.int32)], True, (255, 0, 0), thickness=5)

    # Find the projection matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points) 
    min_wid = 1000
    
    while success:
        success, frame = vidcap.read()
        # frame = vidcap.read()
        frame = cv2.undistort(frame, cam_matrix, dist_coeffs) # melakukan proses undistorsi
        frame = cv2.warpPerspective(frame, M, settings.UNWARPED_SIZE) # melakukan transformasi perspektif pada gambar
        frame_hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS) # mengubah warna gambar dari RGB ke HLS 
        mask = frame_hls[:, :, 1] > 128 # membuat mask yang berisi piksel dg nilai channel L yang lebih besar dari 128
        mask[:, :50] = 0 # mengubah nilai piksel pada kolom pertama hingga kolom ke-50 menjadi 0
        mask[:, -50:] = 0 # mengubah nilai piksel pada kolom ke-50 dari belakang menjadi 0
        mom = cv2.moments(mask[:, :settings.UNWARPED_SIZE[0]//2].astype(np.uint8)) # menghitung momen dari area mask pada setengah kiri gambar, area kolom hingga setengah lebar gambar
        x1 = mom["m10"]/mom["m00"] # menghitung koordinat x dari pusat area mask pada setengah kiri gambar
        mom = cv2.moments(mask[:, settings.UNWARPED_SIZE[0]//2:].astype(np.uint8)) # menghitung momen dari area mask pada setengah kanan gambar, area kolom hingga setengah lebar gambar
        x2 = settings.UNWARPED_SIZE[0]//2 + mom["m10"]/mom["m00"] # menghitung koordinat x dari pusat area mask pada setengah kanan gambar
        cv2.line(frame, (int(x1), 0), (int(x1), settings.UNWARPED_SIZE[1]), (255, 0, 0), 3) # menggambar garis vertikal biru pada gambar
        cv2.line(frame, (int(x2), 0), (int(x2), settings.UNWARPED_SIZE[1]), (0, 0, 255), 3) # menggambar garis vertikal merah pada gambar
        # memeriksa selisih antara x1 dan x2, jika selisihnya lebih kecil dari min_wid, maka min_wid akan diubah menjadi selisih x1 dan x2
        if (x2-x1<min_wid):
            min_wid = x2-x1
        
        # cv2.imshow("img", img)
        # if cv.waitKey(0) & 0xFF == ord('q'):
        #     break

    meter_per_foot = 1/3.28084 # konversi meter ke foot
    pixel_per_meter_x = min_wid/(12* meter_per_foot) # menghitung nilai pixel per meter pada sumbu x dengan membagi min_wid dengan 12 kaki dikalikan meter/foot
    Lh = np.linalg.inv(np.matmul(M, cam_matrix)) # menghitung invers dari hasil perkalian matriks M dan cam_matrix
    pixel_per_meter_y = pixel_per_meter_x * np.linalg.norm(Lh[:,0]) / np.linalg.norm(Lh[:,1]) # menghitung jumlah piksel per meter pada sumbu y dengan memperhitungkan rasio antara norma kolom pertama dan kedua dari matriks Lh
    print(pixel_per_meter_x, pixel_per_meter_y)



    plt.imshow(frame)
    plt.show
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

    # #selecting Coordinates
    # tl = (373, 479)
    # bl = (903, 479)
    # tr = (1709, 685)
    # br = (-532, 685)

    # cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    # cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    # cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    # cv2.circle(frame, br, 5, (0, 0, 255), -1)

    
    
    # # Apply Geometrical Transformation
    # pts1 = np.float32([tl, bl, tr, br])
    # # pts2 = np.float32([[0, 0], [0, 720], [1280, 0], [1280, 720]])

    # cv2.polylines(frame, [pts1.astype(np.int32)], True, (0, 0, 255), thickness=5)
    
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # result = cv2.warpPerspective(frame, matrix, (1280, 720))

    # cv2.imshow("Frame", frame)
    # # cv2.imshow("Perspective transformation", result)

    # if cv2.waitKey(1) == ord('q'):
    #     break

    #lines [  [[315 667 627 423]]
            # [[341 664 628 423]]
            # [[375 619 626 423]]
            # [[337 668 630 422]]
            # [[314 668 615 433]]
            # [[613 427 842 543]]
            # [[615 427 854 543]]
            # [[623 424 844 546]]
            # [[641 440 847 545]]
            # [[661 452 846 546]]  ]