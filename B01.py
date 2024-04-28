import cv2
import mediapipe as mp
import pymysql
import numpy as np



mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測


cap = cv2.VideoCapture(1)




def get_knee_angle(landmarks):#角度
    r_hip = get_landmark(landmarks,24) #"RIGHT_HIP"
    l_hip = get_landmark(landmarks,23) #"LEFT_HIP"

    r_knee = get_landmark(landmarks,26) #"RIGHT_KNEE"
    l_knee = get_landmark(landmarks,25) #"LEFT_KNEE"

    r_ankle = get_landmark(landmarks,28) #"RIGHT_ANKLE"
    l_ankle = get_landmark(landmarks,27) #"LEFT_ANKLE"

    r_angle = calc_angles(r_hip, r_knee, r_ankle)
    l_angle = calc_angles(l_hip, l_knee, l_ankle)

    return [r_angle, l_angle]
    
def calc_angles(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

def get_landmark(landmarks, part_index):
    return [
        landmarks[part_index].x,
        landmarks[part_index].y,
        landmarks[part_index].z,
    ]

def checkpose(landmarks):#姿勢判斷式
    #搬
    
    if landmarks is None:
        poseaa = None 
    else:
        if landmarks[14].y<landmarks[12].y and landmarks[16].y<landmarks[14].y:
            if landmarks[13].y<landmarks[11].y and landmarks[15].y<landmarks[13].y:
                if get_knee_angle(landmarks)[0] < 60 or get_knee_angle(landmarks)[1] < 60:
                    poseaa="舉雙手+蹲"
                    print(get_knee_angle(landmarks)[0])
                else:
                    poseaa="舉雙手"
            else:
                poseaa="舉右手"
        elif landmarks[13].y<landmarks[11].y and landmarks[15].y<landmarks[13].y:
            poseaa="舉左手"
        elif get_knee_angle(landmarks)[0] < 60 or get_knee_angle(landmarks)[1] < 60:
            poseaa="蹲"
        else:
            poseaa = None  # 其他情況下重置姿勢為 None
    return[poseaa]



i=0
# 啟用姿勢偵測
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
        id=1
        i += 1
        if i == 20:
            i = 0
            # 在執行之前檢查是否有姿勢偵測結果
            if results.pose_landmarks is not None:
                landmarks = results.pose_landmarks.landmark
                # 繼續執行接下來的程式碼   
            else:
                print("No landmarks detected")
                landmarks = None  # 設定為 None，以避免後續程式碼因為訪問 None 而引發錯誤

            print(checkpose(landmarks)) #姿勢判定
            

        if cv2.waitKey(5) == ord('q'):
            break     # 按下 q 鍵停止


#END
cap.release()
cv2.destroyAllWindows()
