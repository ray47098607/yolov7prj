import cv2
import mediapipe as mp
import pymysql

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(1)

db = pymysql.connect(host='localhost',
                     user='web',
                     password='web',
                     database='web')
cursor = db.cursor()

i = 0
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
        
        img = cv2.resize(img, (520, 300))
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img2)

        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imshow('oxxostudio', img)
        i += 1
        if i == 100:
            i = 0
            landmarks = results.pose_landmarks.landmark
            
            sql = f"INSERT INTO `yolo`(`time`, `a0nose`, `a1lefteye(inner)`, `a2lefteye`, `a3lefteye(outer)`, `a4righteye(inner)`, `a5righteye`, `a6righteye(outer)`, `a7leftear`, `a8rightear`, `a9mouth(left)`, `a10mouth(right)`, `a11leftshoulder`, `a12rightshoulder`, `a13leftelbow`, `a14rightelbow`, `a15leftwrist`, `a16rightwrist`, `a17leftpinky`, `a18rightpinky`, `a19leftindex`, `a20rightindex`, `a21leftthumb`, `a22rightthumb`, `a23lefthip`, `a24righthip`, `a25leftknee`, `a26rightknee`, `a27leftankle`, `a28rightankle`, `a29leftheel`, `a30rightheel`, `a31leftfootindex`, `a32rightfootindex`) VALUES (current_timestamp(),"
            for lm in landmarks:
                sql += f"'x: {lm.x}y: {lm.y}z: {lm.z}visibility: {lm.visibility}',"
            sql = sql[:-1]  # 去掉最後的逗號
            sql += ")"
            print(sql)

            try:
                cursor.execute(sql)
                db.commit()
            except Exception as e:
                print("An error occurred:", e)
                db.rollback()

        if cv2.waitKey(5) == ord('q'):
            break

db.close()
cap.release()
cv2.destroyAllWindows()
