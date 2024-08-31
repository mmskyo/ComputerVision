# Computer Vision and Deep Learning
## Detecting Objects(up to 3 objects) using YOLO version 3!
<details>
  <summary>click to view</summary>
  <div markdown = "1">

#### result

![yolo_v3_pyqt5](https://github.com/user-attachments/assets/43568565-7a9f-4535-8ca7-601b5d1b4aad)

cited
@Article{yolov3,
title={YOLOv3: An Incremental Improvement},
author={Redmon, Joseph and Farhadi, Ali},
journal = {arXiv},
year={2018}
}
</div>
</details>

## Safer Street for the Traffic Weak
교통 약자 보호구역 알림 시스템
sift를 사용한 교통 표지판 인식기

<details>
  <summary>click to view</summary>
  <div markdown = "1">
    ![TrafficWeakAlertResult](https://github.com/user-attachments/assets/e997f6e7-1b9e-4930-92ce-2a0311a6c9a0)

  </div>
</details>

## recognizing image using CIFAR10
<details>
<summary>click to view</summary>

<div markdown = "1">

accuracy and loss graph


![CIFAR_accuracy](https://github.com/user-attachments/assets/7fc30d71-35d7-4750-9f5a-1eebd2188e0a)
![CIFAR_loss](https://github.com/user-attachments/assets/72748bf7-c2f0-458d-b6e2-1749328bb5e7)

what console looks like
![CIFAR__console](https://github.com/user-attachments/assets/a8dff7d6-0e65-4d63-b0c8-1e1b94d2dc51)



## ResNet50

result being
![ResNet50_result](https://github.com/user-attachments/assets/8100bf36-787a-4259-99e9-9d747cc2074e)



## Sequential module - postcode recognition program
using pre-trained model 'cnn_v2'
![postcode_v 2_result](https://github.com/user-attachments/assets/1fe0f413-61de-4981-a10f-9943f6b31b5c)

</div>
</details>

# 개념
<details>
<summary><b>Convolution Layers</b></summary>

<div markdown = "1">

## 컨볼루션 신경망을 이용한 자연 영상 인식

### 컨볼루션 설명
---
인간은 2차원에서 특징 추출
- 수용장(receptive field)이라는 작은 영역에서 특징을 추출


컨볼루션 신경망(CNN:Convolutional Neural Network)은 인간 시각을 모방
- 딥러닝 성공에 가장 기여한 모델
- 다양한 응용
  - 컴퓨터 비전에서는 분류, 검출, 분할, 추적 등의 문제 해결
  - 비디오 게임 인공지능에서는 화면 장면을 분석
  - 알파고는 19 * 19 바둑판 형세 판단


컨볼루션 신경망
- CNN은 컨볼루션 연산을 하는 신경망 구조
- 고전적ㅇ인 방법에서는 사람이 필터 설계(e.g. 가우시안 스무딩, 소벨 엣지 등)
  - 사람이 설계->적용
  - 인식에 최적이진 않다
  - 매번 데이터셋마다 최적 필터가 달라진다.
 
  
 
#### CNN의 핵심 아이디어는 '최적의 필터(가중치)를 학습'으로 알아낸다.

인간의 입장에서는 같은 문양일지라도 컴퓨터 입장에서는 불일치하는 픽셀이 많은 경우,
CNN은 픽셀 단위로 보지않고, 영상의 작은 특징 부분을 추출해서 특징끼리 비교한다.

#### 컨볼루션 층에서의 영상 특징 추출
문양을 분류하는 convolution layer
특징 추출 : 가중치를 가진 필터를 이용

실제 CNN 모델에서는 위처럼 미리 가중치를 정해주지 않고, 학습하면서 가중치를 계속 업데이트하여 최적의 가중치를 스스로 찾아낸다.

필터와 영상을 컨볼루션하고, 필터와 일치하면 1값을 갖게된다.

필터(마스크)와 영상을 컨볼루션
  - 필터와 일치하지 않는 부분은 1보닫 작은 값을 가지게 된다.

=> 그 결과 3개의 특징맵을 구할 수 있게된다.

#### Pooling Layer - 추출된 특징의 압축
윈도우 사이즈를 정한다
-> 윈도우 내 최대 값을 선택한다 = max pooling
-> 윈도우 내 평균 값을 선택한다 = mean pooling
-> 윈도우 내 최소 값을 선택한다 = minimum pooling

7 by 7 사이즈 영상이 4 by 4 사이즈로 작게 되었다.
효과 : 계산 효율을 높이고, 특징의 정확한 위치에 덜 민감하게 된다. (overfitting 방지)
*압축을 했어도 여전히 특징 위치는 유지한다

필터링된 모든 특징맵에 풀링을 적용한다.

#### ReLU - 활성화 함수
- 필요성
  - 필터를 통과한 데이터는 덧셈, 곱셈으로만 이루어져 있어서 선형적인 특성을 갖는 상태이다. 이 경우 복잡한 데이터 분류는 힘들다.
  - ReLU와 같은 activation function은 비선형성을 부여해주어 복잡한 데이터도 분류할 수 있도록한다.
  - e.g. 뉴런이 다음 뉴런으로 신호를 보낼 때 입력 신호가 일정 기준 이상이면 보내고, 기준에 달하지 않으면 보내지 않도록 한다.
 
ReLU Layer
일정 값 이상의 정보들만 통과된다. (양수만 통과되고 음수는 0이된다) (=죽은 뉴런이 생긴다. 필요없는 정보는 날리지만 중요한, 값이 큰 부분은 유지한다.)

#### 레이어에서 나온 결과는 다음 레이어의 입력으로 들어간다.
#### 순서 컨볼루션->ReLU(먼저 필요없는 값을 죽이기 위해)->풀링
더 깊게 레이어를 쌓을 수 있다. 목적에 따라 반복하여 쌓아서 사용한다.

#### Fully Connected Layer - 분류를 위한 층
- 추출된 특징들을 평평하게 펼쳐서 다층 신경망에 넣어 이 영상이 무슨 영상인지 분류할 수 있다.
- 다층 신경망을 지나면 문양을 분류하게 된다.

***기억할 점!
실제 분류 문제에서는 필터의 가중치를 미리 정해놓고 영상의 특징을 추출하는 것이 아니라, 필터의 가중치를 랜덤하게 두고 오차가 발생하면 오차가 줄어드는 방향으로 학습하여 가중치를 업데이트 하면서 최적의 가중치를 스스로 찾는다.(Back Propagation) (손실함수가 0이되는 방향으로)***

#### 컨볼루션 층
- 입력 특징 맵이 m * n * k 텐서라면, h * h * k텐서 사용
- 하나의 필터는 bias 하나를 가진다(kh^2 + 1개의 가중치)
- 필터를 여러개(k'개) 적용하여 풍부한 특징 맵을 추출한다
- 출력 특징 맵은 m * n * k' 텐서
- 덧대기(padding)와 보폭(stride)가 존재

- 컨볼루션 층의 연산
  - 5 * 5 * 3 특징 맵에 3 * 3 * 3 필터 2개 적용
     - padding = 0(제로 패딩)적용, stride = 2 or 1
  - 5 * 5 * 3 입력 특징 맵이 3 * 3 * 2 출력 특징맵이 된다.

- 컨볼루션 층의 바람직한 특성
  1. 입력 특징 맵의 모든 화소가 같은 필터 사용 -> 가중치를 공유(필터의 값이 가중치)
  2. 필터는 해당 화소 주위에 국한하는 연산을 수행(나와 주변 화소) -> 부분 연결성을 만족
  3. 가중치 갯수가 획기적으로 줄어든다.
    - k'개의 h * h * k 필터(필터 갯수 * 필터 크기)를 쓰는 경우 가중치는 k'(kh^2+1)개 (= 가중치의 갯수가 dense보다 줄어들어서 훨씬 효율적이다.)

#### 풀링 층
  - 최대 풀링은 필터 안의 화소의 최댓값을 취한다.
  - 평균 풀링은 필터 안의 화소의 평균을 취한다.
  - (압축하면)지나친 상세함을 줄이는 효과와 특징 맵의 크기를 줄이는 효과
    cf. 풀링은 패딩을 안한다 in contrary to 컨볼루션
    CS231n - 스탠포드 무료 딥러닝 강의

- 빌딩 블록 쌓기
  - 컨볼루션 층과 풀링 층을 하나의 구성요소로 사용
  - 풀링 층에서는 텐서 깊이(채널)가 유지된다
  - 신경망 앞 부분은 특징 추출, 뒷 부분은 분류 담당이다.

    - 유연한 구조
      -문제에 따라 다양한 모양으로 조립 가능(순서가 달라도 가능)
    - 역전파 학습 알고리즘 사용(다층 퍼셉트론과 비슷)
      - 컨볼루션 층과 완전연결층의 U^i(가중치)가 학습 대상
      - 풀링층은 가중치 없음(역전파 학습대상X)
        
#### 컨볼루션 신경망의 특징1
- 특징을 담당하는 필터를 학습한다 -> 특징 학습 feature training
- 학습 알고리즘이 주어진 데이터셋을 인식하는데 최적인 필터를 알아낸다

#### 컨볼루션 신경망의 특징2
- 통째 학습 end-to-end learning
- 특징 학습과 분류기 학습을 한꺼번에 진행(입력만 넣으면 최종 출력이 나온다)

컨볼루션 신경망이 우수한 이유
- 데이터의 원래 구조를 유지
- 특징학습을 통해 최적의 특징을 추출
- 신경망의 깊이를 깊게하는 것이 가능하다.

  </div>
</details>
