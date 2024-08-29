import numpy as np
import cv2 as cv
import sys
import winsound
from PyQt5.QtWidgets import *

# 비전에이전트 - 정적 이미지 검출
class stillImageDetection(QMainWindow):
    def __init__(self) :
        super().__init__()
        self.setWindowTitle('물체 검출기')
        self.setGeometry(200, 200, 700, 100)
        
        fileButton = QPushButton('파일 열기', self)
        detectionButton = QPushButton('물체 검출', self)
        quitButton = QPushButton('나가기', self)
        
        fileButton.setGeometry(10, 10, 100, 30)
        detectionButton.setGeometry(110, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)
        
        fileButton.clicked.connect(self.pictureOpenFunction)
        detectionButton.clicked.connect(self.detectionFunction)
        quitButton.clicked.connect(self.quitFunction)
        
    
        
    # 파일 열기 버튼 기능
    def pictureOpenFunction(self):
        fname = QFileDialog.getOpenFileName(self, '사진 불러오기', './')
        self.img = cv.imread(fname[0])
        if self.img is None: sys.exit('파일을 찾을 수 없습니다.')
        
        # 불필요 cv.imshow('Still Image', self.img)
        
    # 물체 검출 버튼 기능
    def detectionFunction(self):
        # YOLO 모델 구성
        def construct_yolo_v3():
            f = open('coco_names.txt', 'r')
            class_names = [line.strip() for line in f.readlines()]
            
            model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')
            layer_names = model.getLayerNames()
            out_layers = [layer_names[i-1] for i in model.getUnconnectedOutLayers()]
            
            return model, out_layers, class_names
        
        # YOLO 검출
        def yolo_detection(img, yolo_model, out_layers):
            height, width = img.shape[0], img.shape[1]
            test_img = cv.dnn.blobFromImage(img, 1.0 / 256, (448, 448), (0, 0, 0), swapRB = True)
            yolo_model.setInput(test_img)
            output3 = yolo_model.forward(out_layers)
            
            # 박스, 신뢰도, 부류 번호
            box, conf, id = [], [], []
            for output in output3:
                for vec85 in output:
                    scores = vec85[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    # 신뢰도가 50% 이상인 경우면 취함
                    if confidence > 0.5: 
                        centerx, centery = int(vec85[0] * width), int(vec85[1] * height)
                        w, h = int(vec85[2] * width), int(vec85[3] * height)
                        x, y = int(centerx - w / 2), int(centery - h / 2)
                        box.append([x, y, x + w, y + h])
                        conf.append(float(confidence))
                        id.append(class_id)
                        
            ind = cv.dnn.NMSBoxes(box, conf, 0.5, 0.4)
            objects = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]
            return objects


        # yolo 모델 생성, 부류마다 색깔
        model, out_layers, class_names = construct_yolo_v3()
        colors = np.random.uniform(0, 255, size = (len(class_names), 3))
    
        # YOLO 모델로 물체 검출
        res = yolo_detection(self.img, model, out_layers)

        # 검출된 물체를 영상에 표시
        for i in range(len(res)):
            x1, y1, x2, y2, confidence, id = res[i]
            text = str(class_names[id]) + '%.3f' % confidence
            cv.rectangle(self.img, (x1, y1), (x2, y2), colors[id], 2)
            cv.putText(self.img, text, (x1, y1 + 30), cv.FONT_HERSHEY_PLAIN, 1.5, colors[id], 2)
        
        # 출력
        cv.imshow("Object detection by YOLO v.3", self.img)
        winsound.Beep(1000, 500)
    
    # 나가기 버튼 기능
    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()



app = QApplication(sys.argv)
win = stillImageDetection()
win.show()
app.exec_()

    