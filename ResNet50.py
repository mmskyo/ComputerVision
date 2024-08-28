# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:19:28 2024

@author: kyo07
"""

import cv2 as cv
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions


#ImageNet으로 학습한 ResNet50을 백본으로 사용
model = ResNet50(weights = 'imagenet')

img = cv.imread('rabbit.jpg')
x = np.reshape(cv.resize(img, (224, 224)), (1, 224, 224, 3))
x = preprocess_input(x)

preds = model.predict(x)
top5 = decode_predictions(preds, top = 5)[0]
print('예측 결과 : ', top5)

for i in range(5):
    cv.putText(img, top5[i][1]+ ' : ' + str(top5[i][2]), (10, 200 +  i* 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
cv.imshow('Recognition result', img)

cv.waitKey()
cv.destroyAllWindows()
    
    

