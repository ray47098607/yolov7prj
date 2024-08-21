import cv2
import mediapipe as mp
import numpy as np
import csv

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測

cap = cv2.VideoCapture('D:\CODE\IAMGE_R\image_recognition\VideoElement\V1.mp4') #捕獲視訊(選擇鏡頭)

def get_landmark(landmarks, part_index):
    return [
        landmarks[part_index].x,
        landmarks[part_index].y,
        landmarks[part_index].z,
    ]

def calc_angles(a, b, c):#三點角度
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    # print(np.arctan2(c[1] - b[1], c[0] - b[0]), np.arctan2(a[1] - b[1], a[0] - b[0]))
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

file = open('store01_data.csv',mode='w', newline='')
writer = csv.writer(file)
# 啟用姿勢偵測
i=0
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img = cv2.resize(img,(520,300))               # 縮小尺寸，加快演算速度
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = pose.process(img2)                  # 取得姿勢偵測結果
        
        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        cv2.imshow('oxxostudio', img)

        #########################
        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
        else:
            print("No landmarks detected")
            landmarks = None  # 設定為 None，以避免後續程式碼因為訪問 None 而引發錯誤

        print(checkpose(landmarks))#姿勢判定
        
        txt = i 
        txt = landmarks
        print(txt)
        writer.writerow(txt)
        i=i+1
        #########################

        if cv2.waitKey(5) == ord('q'):
            break     # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()
file.close()


