# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 17:05:18 2024

@author: kyo07
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
""" convolution층 필터 갯수와 크기, 풀링층(최대풀링), 1차원, 완전연결층, ? """
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
""" 최적화 학습 위한 optimizer """
from tensorflow.keras.optimizers import Adam


""" 데이터 코드해서 훈련과 테스트. 데이터로 분류 """
(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()
x_train = x_train.astype(np.float32) / 255.0 
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

""" 객체 만든뒤 층쌓기 """
cnn = Sequential()
""" 컨볼루션층 3 * 3 커널을 32개(필터) 사용 
input_shape 출력 특징 맵의 모양 in CIFAR *첫번째 층이라 크기를 정해줘야한다.
2차원 data 그대로 넣으면 된다."""
cnn.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
cnn.add(Conv2D(32, (3, 3), activation = 'relu'))
""" 풀링할 때의 윈도우 사이즈 """
cnn.add(MaxPooling2D(pool_size = (2, 2)))
""" 규제 기법인 드롭아웃 적용
: over fittiing(새로운 데이터 인식이 어려움, 특정 데이터에만 너무 딱맞는 특징)
을 방지하기 위한 규제기법 중 하나로, 특징 맵을 구성하는 요소 중
일부를 랜덤 선택하여 학습에서 배제하는 기법 
BUT 학습할 때만 사용한다. 예측할 때는 정확해야되기 때문이다."""
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64, (3, 3), activation = 'relu'))
cnn.add(Conv2D(64, (3, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
""" 25% 배제 """
cnn.add(Dropout(0.25))
""" 1차원 구조로 변환 """
cnn.add(Flatten())
""" 512 = 다음 노드의 갯수 """
cnn.add(Dense(units = 512, activation = 'relu'))
cnn.add(Dropout(0.5))
""" 최종 분류 결과는 10 unit, softmax로 분류 """
cnn.add(Dense(units = 10, activation = 'softmax'))

""" 실제 예측, loss = 손실함수, 최적화 함수 """
cnn.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = 0.001), metrics = ['accuracy'])
""" 실제 영상과 실제 학습으로 ???? 히스토그램 평활화???"""
hist = cnn.fit(x_train, y_train, batch_size = 128, epochs = 100, validation_data = (x_test, y_test), verbose = 2)

res = cnn.evaluate(x_test, y_test, verbose = 0)
print('정확률 = ', res[1] * 100)

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()
