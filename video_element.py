import cv2
import mediapipe as mp
import numpy as np

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

# def calc_angles2(a, b, c, d):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#     d = np.array(d)
#     radians = np.arctan2(c[1] - c[1], d[0] - d[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])

#     angle = np.abs(radians * 180.0 / np.pi)

#     if angle > 180:
#         angle = 360 - angle

#     return angle
def pp_distance(a, b):#兩點距離
    a = np.array(a)
    b = np.array(b)
    distance = np.sqrt((a[1] - b[1]) ** 2 + (a[0] - b[0]) ** 2)
    
    return distance

def get_knee_angle(landmarks):#蹲,站
    r_hip = get_landmark(landmarks,24) #"RIGHT_HIP"
    l_hip = get_landmark(landmarks,23) #"LEFT_HIP"

    r_knee = get_landmark(landmarks,26) #"RIGHT_KNEE"
    l_knee = get_landmark(landmarks,25) #"LEFT_KNEE"

    r_ankle = get_landmark(landmarks,28) #"RIGHT_ANKLE"
    l_ankle = get_landmark(landmarks,27) #"LEFT_ANKLE"

    r_angle = calc_angles(r_hip, r_knee, r_ankle)
    l_angle = calc_angles(l_hip, l_knee, l_ankle)

    return [r_angle, l_angle]

def get_stoop_angle(landmarks):#彎腰
    r_shoulder = get_landmark(landmarks,12) #"RIGHT_shoulder"
    l_shoulder = get_landmark(landmarks,11) #"LEFT_shoulder"

    r_hip = get_landmark(landmarks,24) #"RIGHT_hip"
    l_hip = get_landmark(landmarks,23) #"LEFT_hip"

    r_knee = get_landmark(landmarks,28)
    l_knee = get_landmark(landmarks,27)
    # r_heel = get_landmark(landmarks,30) #"RIGHT_heel"
    # l_heel = get_landmark(landmarks,29) #"LEFT_heel"

    # r_f_index = get_landmark(landmarks,32) #"RIGHT_foot_index"
    # l_f_index = get_landmark(landmarks,31) #"LEFT_foot_index"

    r_angle = calc_angles(r_shoulder, r_hip, r_knee)
    l_angle = calc_angles(l_shoulder, l_hip, l_knee)

    return [r_angle, l_angle]

def get_walk_angle(landmarks):#走
    r_heel = get_landmark(landmarks,30)
    l_heel = get_landmark(landmarks,29)

    distance = pp_distance(r_heel, l_heel)
    # print(distance)
    # r_hip = get_landmark(landmarks,24)
    # l_hip = get_landmark(landmarks,23)

    # nose = get_landmark(landmarks,0)
    # n_angle = calc_angles(r_foot, l_foot, nose)

    return [distance]

def checkpose(landmarks):#姿勢判斷式
    poseC1 = []#上半身
    poseC2 = []#下半身
    poseSOL = []#最終判定

    if landmarks is None:
        poseC1 = ['None']
        poseC2 = ['None']
    else:
        #手
        if landmarks[14].y<landmarks[12].y and landmarks[16].y<landmarks[14].y and landmarks[13].y<landmarks[11].y and landmarks[15].y<landmarks[13].y:
                poseC1.append("舉雙手")
        else:
            if landmarks[14].y<landmarks[12].y and landmarks[16].y<landmarks[14].y:
                poseC1.append("舉右手")
            if landmarks[13].y<landmarks[11].y and landmarks[15].y<landmarks[13].y:
                poseC1.append("舉左手")
        #腳
        if get_stoop_angle(landmarks)[0] < 160 and get_stoop_angle(landmarks)[1] < 160:
            poseC2.append("彎腰")
        if get_knee_angle(landmarks)[0] < 140 and get_knee_angle(landmarks)[1] < 140:
            poseC2.append("蹲")
        if get_knee_angle(landmarks)[0] > 140 and get_knee_angle(landmarks)[1] > 140:
            poseC2.append("站")
        if get_walk_angle(landmarks)[0] > 0.09:
            poseC2.append("走")

    if poseC1 == ['None'] and poseC2 == ['None']:
        poseSOL.append('None')
    else:
        for i in range(len(poseC2)):
            if poseC2[i] == "蹲":
                poseSOL = "蹲"
                poseC2 = []
                break
        for i in range(len(poseC2)):
            if poseC2[i] == "彎腰":
                poseSOL = "彎腰"
                poseC2 = []
                break
        for i in range(len(poseC2)):
            if poseC2[i] == "走":
                poseSOL = "走"
                poseC2 = []
                break
        for i in range(len(poseC2)):
            if poseC2[i] == "站":
                poseSOL = "站"
                poseC2 = []
                break
    poseC1 = []#上半身未設定
    return poseSOL

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

        #########################
        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
        else:
            print("No landmarks detected")
            landmarks = None  # 設定為 None，以避免後續程式碼因為訪問 None 而引發錯誤

        print(checkpose(landmarks))#姿勢判定
        txt = f""
        print(txt)
        #########################

        if cv2.waitKey(5) == ord('q'):
            break     # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()