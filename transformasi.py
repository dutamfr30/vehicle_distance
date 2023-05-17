import cv2
import numpy as np
vidcap = cv2.VideoCapture('project_video.mp4')
success,image = vidcap.read()

while success:
    success,image = vidcap.read()
    frame = cv2.resize(image, (1920, 1080))

    #selecting Coordinates
    tl = (373, 479)
    bl = (903, 479)
    tr = (1709, 685)
    br = (-532, 685)

    cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    cv2.circle(frame, br, 5, (0, 0, 255), -1)

    
    
    # Apply Geometrical Transformation
    pts1 = np.float32([tl, bl, tr, br])
    # pts2 = np.float32([[0, 0], [0, 720], [1280, 0], [1280, 720]])

    cv2.polylines(frame, [pts1.astype(np.int32)], True, (0, 0, 255), thickness=5)
    
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # result = cv2.warpPerspective(frame, matrix, (1280, 720))

    cv2.imshow("Frame", frame)
    # cv2.imshow("Perspective transformation", result)

    if cv2.waitKey(1) == ord('q'):
        break