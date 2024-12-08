import cv2
import numpy as np
from ultralytics import YOLO
import torch  
import os                                       #FFMPEG 影像轉檔壓縮 調用win os cmd
from datetime import datetime                   #py時間
import mediapipe as mp
from tensorflow.keras.models import load_model  #仔入tensorflow
from collections import deque                   #加入deque做控制資料大小用
from PIL import ImageFont, ImageDraw, Image     # 載入 PIL 相關函式庫 寫中文字用
import pymysql

P3=130 #三連拍


#lstm_model = load_model('./RE7_augment_LSTM_ModelTrain.h5') #載入 lstm模型
lstm_model = load_model('.\D15_1207_LSTM_ModelTrain.h5') #載入 lstm模型

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測

# 自定義關鍵點和連接線的繪製樣式
landmark_style = mp_drawing.DrawingSpec( color=(200, 200, 200),thickness=10, circle_radius=5)  # 加粗關鍵點
connection_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=5)  # 加粗連接線

# Get the current time for file naming
time_now = datetime.now()
current_time = time_now.strftime("%Z_%Y_%m_%d_%H_%M_%S")

# Load the YOLOv8 model
#ver=["A","A2","A3","A4"]#A>n A2>x A3>n A4>s
ver=["C","C2","C3"]#c n*100 c2 n*200 c3 s*200
v=1
yolo_model = YOLO("./runs/train"+ver[v]+"/weights/best.pt")

# Open the video file
video_path = "V.mp4"                         #來源檔案位置如為0為鏡頭
#video_path = 0
if isinstance(video_path, str):
    video_path2 =video_path
else:
    video_path2 ="{video_path}"         #檔名須為字串

cap = cv2.VideoCapture(video_path)

# Get the video frame width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object 影片輸出
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
if isinstance(video_path, str):
    base_name = os.path.splitext(video_path)[0] #去附檔名
else:
    base_name = "camera_output"
OUTPUTPATH = f"{current_time}_{base_name}_{ver[v]}_output.MP4"
out = cv2.VideoWriter(OUTPUTPATH, fourcc, 30, (width, height))

#定義線段起終點
#原點在左上
x1, y1 = int(width*0.35), int(height*0.6)  # A起點
x2, y2 = int(width*0.2), int(height*1)  # A終點

x12, y12 = int(width*0.73), int(height*0.6)  # B起點
x22, y22 = int(width*0.8), int(height*1)  # B終點

print('Processing video...')

###########
STOREAold=0
STOREA =0
P4=0

###
label_name = ['站', '彎腰', '蹲', '走', '跌倒']     #動作標籤名
window_size = 30                                    #單次處理資料量
data_winsize = []                       
data_winsize = deque(maxlen=window_size)
###

text='初始化'
predicted_class = 10

#SQL連結
db = pymysql.connect(host='localhost',
                     user='web',
                     password='web',
                     database='web')
cursor = db.cursor()


def point_relative_to_line(x1, y1, x2, y2, xc, yc):
    # 計算位置值
    value = (y2 - y1) * (xc - x1) - (x2 - x1) * (yc - y1)
    
    return value
    

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    debug = 0
    if success:
        # Run YOLOv8 detection on the frame
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        results = yolo_model.predict(frame, imgsz=640, conf=0.5, device=device, save=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # 假設 results 是 YOLOv8 偵測返回的結果
        for result in results :
            a=0
            #print(result.boxes)  # 使用 boxes 屬性，列出每個幀的偵測框

        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

            results = pose.process(frame)                  # 取得姿勢偵測結果
            # 根據姿勢偵測結果，標記身體節點和骨架
            

            if results.pose_landmarks is not None:
                landmarks = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(
                    annotated_frame,                            # 用於顯示的影像（已標註的框架）
                    results.pose_landmarks,                     # 檢測到的姿勢關鍵點
                    mp_pose.POSE_CONNECTIONS,                   # 姿勢關鍵點之間的連接線
                    landmark_drawing_spec=landmark_style,       # 自定義關鍵點樣式
                    connection_drawing_spec=connection_style    # 自定義連接線樣式
                    ) 
            else:
                print("No landmarks detected")
                landmarks = None
            
            ###TensorFlow
            
            data=[]
            for j in range(33):
                if landmarks:
                    data.append(landmarks[j].x)
                    data.append(landmarks[j].y)
                else:
                    data.append(0)
                    data.append(0)
            data_winsize.append(data)

            if len(data_winsize) == 30:
                act_data = np.array(data_winsize)
                act_data = act_data.reshape(1,30,66)


                for _ in range(8):
                    removed_element = data_winsize.popleft()

                
                # print(act_data.shape)
                # print(act_data)

                predictions = lstm_model.predict(act_data)
                predicted_class = np.argmax(predictions)
                # print("Predicted Classes:", label_name[predicted_class])
                text = label_name[predicted_class]
            ######TensorFlow
            #加字
                             
            # 繪製黑色背景並加上文字
            img = np.zeros((60, 180, 3), dtype='uint8')             # 繪製黑色畫布
            fontpath = './font/NotoSansTC-Regular.ttf'              # 載入字型
            font = ImageFont.truetype(fontpath, 50)                 # 設定字型與文字大小
            imgPil = Image.fromarray(img)                           # 將 img 轉換成 PIL 影像
            draw = ImageDraw.Draw(imgPil)                           # 準備開始畫畫
            draw.text((0, 0), text=text , fill=(255, 255, 255), font=font)  # 畫入文字                
            img = np.array(imgPil)                                  # 將 PIL 影像轉換成 numpy 陣列

            # 確認 img 和 annotated_frame 的範圍正確
            h, w, _ = img.shape
            if annotated_frame.shape[0] >= 150 + h and annotated_frame.shape[1] >= 100 + w:
                annotated_frame[75:75 + h, 50:50 + w] = img
            else:
                print("Image out of bounds")
            #加字
            #畫線
            cv2.line(annotated_frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.line(annotated_frame, (x12, y12), (x22, y22), color=(0, 0, 255), thickness=2)
            
###########出入庫
            if result.boxes.xywhn is not None and len(result.boxes.xywhn) > 0:
                for box in result.boxes.xywhn:
                    x_rel, y_rel, w_rel, h_rel = box.tolist()
                    x_pixel = x_rel * width
                    y_pixel = y_rel * height
            else:
                x_pixel, y_pixel = None, None

            if x_pixel is not None and y_pixel is not None:
                L1P = point_relative_to_line(x1, y1, x2, y2, x_pixel, y_pixel)
                L2P = point_relative_to_line(x12, y12, x22, y22, x_pixel, y_pixel)
           
                
                
                #原點左上指定由左至右為1BD/2AD/3AC 以STOREA
##############     
                if L1P >= 0:
                        #wh1I
                    print("AAAAAAA")
                    if L2P >= 0:
                        print("CCCCCCCCCC")
                        STOREA=0 

                    elif L2P <0:
                        print("DDDDDDDD")
                        STOREA=2

                    else:
                        STOREA=0

                elif L1P < 0:
                        print("BBBBBBBBBBB")
                        print("DDDDDDDD")    
                        STOREA=1
                else:
                    STOREA=0 
    ###############
                if  STOREAold is not STOREA:
                    
                    if STOREA == 0:
                        text1 = "貨進A區"
                    elif STOREA == 1:
                        text1 = "貨進B區"
                    elif STOREAold == 0:
                        text1 = "貨出A區"
                    elif STOREAold == 1:
                        text1 = "貨出B區"
                    else :
                        text1 = "X"
                        


                    SQLA1 = datetime.now().strftime("%Y-%m-%d")
                    SQLB1 = datetime.now().strftime("%H:%M:%S")
                    SQLC1 = text1 + "_D15_1207"

                    if text1 != "X":
                        
                        SQLUPA = f"INSERT INTO io(日期, 時間, 事件) VALUES ('{SQLA1}', '{SQLB1}', '{SQLC1}')"
                        try:
                            # 执行SQL语句
                            cursor.execute(SQLUPA)
                            # 提交到数据库执行
                            db.commit()
                        except:
                            # 发生错误时回滚
                            db.rollback()

                            #os.system('pause')
                    STOREAold =STOREA
            else:
                print("進出ERR")    

            #DEBUG
            if cv2.waitKey(1) & 0xFF == ord("j"):
                debug = 1
                                        
            if debug ==1:
                os.system('pause')
                debug = 0

            #跌倒SQLIN
            P3=P3+1
            resized_image =cv2.resize(annotated_frame,(640,480))
            #如果偵測到 輸出三張圖
            if predicted_class == 4 and P3 > 100:
                P3=0
                # 寫入圖檔
                cv2.imwrite('P1.jpg', resized_image)
                
                SQLA = datetime.now().strftime("%Y-%m-%d")
                SQLB = datetime.now().strftime("%H:%M:%S")
                SQLC = text + "_D15_1207"
                
                

            if P3 == 20:
                # 寫入圖檔
                cv2.imwrite('P2.jpg', resized_image)

            if P3 == 40:
                # 寫入圖檔
                cv2.imwrite('P3.jpg', resized_image)


            #SQL
            if P3 == 50:
                with open('P1.jpg', 'rb') as file1, open('P2.jpg', 'rb') as file2, open('P3.jpg', 'rb') as file3:
                    binary_data1 = file1.read()
                    binary_data2 = file2.read()
                    binary_data3 = file3.read()
                    SQLUP = f"INSERT INTO event(日期, 時間, 事件, P1, P2, P3) VALUES ('{SQLA}', '{SQLB}', '{SQLC}', %s, %s, %s)"
                    try:
                    # 执行SQL语句
                        cursor.execute(SQLUP, (binary_data1,binary_data2,binary_data3))
                    # 提交到数据库执行
                        db.commit()
                    except:
                    # 发生错误时回滚
                        db.rollback()

            if predicted_class == 4 and P3 >= 100:
                P3 = 0  # 重置計數器
                cv2.imwrite('P1.jpg', resized_image)
                    
        # Write the annotated frame
        out.write(annotated_frame)
        out.write(annotated_frame)
        
        # Display the frame with annotations
        #resized_image =cv2.resize(annotated_frame,(640,480))
        cv2.imshow("YOLOv8 Tracking", resized_image)        #顯示畫面

        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

print("Processing complete.")
# Release video objects
cap.release()
out.release() 
db.close()

cv2.destroyAllWindows()

# Check if the output file exists and convert it with ffmpeg 檔案轉碼
if os.path.exists(OUTPUTPATH):
    os.system(f"ffmpeg -i {OUTPUTPATH} -vcodec libx264 -f mp4 output_A_{OUTPUTPATH}")
    os.remove(OUTPUTPATH)
else:
    print(f"Error: File {OUTPUTPATH} not found.")
