# Why You Should Try the Real Data for the Scene Text Recognition

## OCR 관련 SOTA 논문

- 우편물 순로구분기

  - OCR + Classification + Sorting

  ![EULJI](https://lh3.googleusercontent.com/proxy/k-QSCuGUFi5vbmODj83t5OOkN3HYKHt6pB3cEX504u0Jtm0Lav2z7OPQEWqjdbphbA0Gd3JKowlHsePcJm0UGN4SHPnEKgoDNvowMIV4ykLgRAfWvjaH0auyEybxPFOzFFE)

- `13년 ETRI에서 도입한 머신러닝 기반 OCR을 사용하여 분류 중 

  - 좋은 데이터가 들어와서 성능이 나쁘지 않음

  

  [성능 통계]
                 투입우편물	               2138	                  0	                  0
                  획득 영상	               2131	                  0	                  0
           바코드 판독률(%)	               0.00	               0.00	               0.00
           바코드 판독 성공	                  0	                  0	                  0
           바코드 판독 실패	                  0	                  0	                  0
           바코드 인쇄율(%)	               0.00	               0.00	               0.00
           바코드 인쇄 성공	                  0	                  0	                  0
           바코드 인쇄 실패	                  0	                  0	                  0
         우편번호 판독률(%)	              98.92	               0.00	               0.00
         우편번호 판독 성공	               2108	                  0	                  0
         우편번호 판독 실패	                 23	                  0	                  0
             주소 판독률(%)	              89.68	               0.00	               0.00
             주소 판독 성공	               1911	                  0	                  0
             주소 판독 실패	                220	                  0	                  0
             주소 검색률(%)	             100.00	               0.00	               0.00
             주소 검색 성공	               1911	                  0	                  0
             주소 검색 실패	                  0	                  0	                  0
              구분된 우편물	               1911	                  0	                  0
              처리량(통/시)	              30183	                  0	                  0
                  구분률(%)	              89.68	               0.00	               0.00
                   사전거부	                  1	                  0	                  0
                       기각	                226	                  0	                  0
                         잼	                  0	                  0	                  0
             작업 시작 시간	2021-01-04 07:36:29	1970-01-01 09:00:00	1970-01-01 09:00:00
             작업 종료 시간	2021-01-04 07:41:50	1970-01-01 09:00:00	1970-01-01 09:00:00
                  유휴 시간	                 66	                  0	                  0

  - 외부 빛 차단 + 고성능 카메라 -> 이진변환 사진 사용 
  - ![img54712853](https://law.go.kr/LSW/flDownload.do?flSeq=54712853)

![img54712049](https://law.go.kr/LSW/flDownload.do?flSeq=54712049)





### 1. 들어가는 말

- OCR(Text recognition)의 경우 오랜시간 연구되어 왔고 딥러닝으로인해 높은 성능을 보이고 있는 분야중에 하나임
  - 비전자 기록물을  디지털화
  - 시각장애인을 위한 점자로 변환
  - 자동차 번호판 인식
  - 우편물 인식
- 하지만 괜찮은 데이터셋의 부재로 인해 모델의 정확도가 충분하게 끌어올려지지 않은 단점이 있음
  - 사람이 레이블한 리얼데이터(ICDAR)의 경우 데이터 수의 부족
  - 생성 데이터 셋(MJSynth, Synthatext)의 경우 Diversity 부족

- OpenImage V5의 경우 좀더 Diversity한 예제가 있어 해당 데이터를 활용하여 모델을 보강

- 연구 방향

0. 기존의 연구들

   데이터를 조정하기 보다는 주로 모델 아키텍처를 조정하여 성능을 높이는데 중점을 두고 연구

1. Baek et al.(2019)

   텍스트 인식 모델을 **변환**, **특징 추출, 시퀀스 모델링 및 예측**(transformation, feature extraction, sequence modelling and prediction)의 4단계로 구분하는 아이디어를 제안

   ![image-20211201200244252](C:\Users\LeeChul-Ghi\AppData\Roaming\Typora\typora-user-images\image-20211201200244252.png)

   많은 텍스트 인식 모델 아키텍쳐가 이 체계에 따라 학습 - **해당 논문도 같은 프로세스로 구성**

- 논문에 사용된 아키텍쳐는 기존 연구의 모델을 그대로 차용하여 사용

  **1) 변환(Transformation)**: TPS Jaderberg et al. (2015) 

  **2) 특징추출(Feature extraction)**: ResNeXt Xie et al. (2016) 

  **3) 시퀀스 모델링 및 예측(sequence modelling and prediction):**  YAMTS Krylov et al.(2021) 

- Contribution

  - 모델링에 사용하는 데이터셋을 새롭게 구성하는(Snthetic + Real) 아이디어 적용
  - 위의 데이터에 기존 모델 파라미터 튜닝으로 SOTA 성능 

### 2. 관련연구

#### 2.1 데이터

##### MJSynth (Jaderberg et al. (2014))

 MJSynth는 9만 개의 영어단어가 포함된 900만 개의 이미지로 구성되며 데이터 자체에서 훈련, 검증, 평가용으로 구분

![img](https://webzine.aihub.or.kr/insight/vol05/resources/img/sub/img_sub401.jpg)



https://www.robots.ox.ac.uk/~vgg/data/text/



#####  SynthText (Gupta et al. (2016))

네츄럴 이미지에 가상의 문자를 합성하여 약 80만 개의 생성된 훈련 이미지를 포함한 텍스트 데이터 셋

![Synthetic Scene-Text Samples](https://github.com/ankush-me/SynthText/raw/master/samples.png)

##### OpenImagesV5 (Krylov et al.(2021))

임의의 모양의 텍스트 이미지에 대한 데이터  셋

자연 이미지 28,134개
903,069개의 주석이 달린 장면 텍스트 단어
이미지당 평균 32단어

![TextOCR Dataset | Papers With Code](https://production-media.paperswithcode.com/datasets/Screenshot_2021-05-13_at_10.17.05.jpg)

#### 2.2 모델 관련연구

- Baek et al. (2021)

  텍스트 인식 모델을 훈련하기 위해 실제 데이터 세트를 사용하는 아이디어를 제안한 최초의 연구

   large annotated datasets 에 사용할 수 없는 단점 

- Krylov et al(2021) - OpenImagesV5  / Singh et al. (2021) - TextOCR 

   large annotated datasets에 대한 텍스트 인식 모델 교육을 가능하게 했습니다.

- Shiet al. (2019) - ASTER 

  Thin Plate Spline  변환 모델을 제안, 해당 모델은 인식 정확도를 높힐 수 있음

- Xie et al. (2016) - ResNeXt, He et al. (2016) - ResNet-like, Shi et al.(2019),  Cai et al.(2021),  Qiao et al.(2020),  Beck et al.(2019).

  여러가지 OCR 백본 연구가 제안 되고 있음

- Bahdanau et al. (2016)

  OCR에 어텐션 모델을 처음 적용한 논문, 최근에는 모든 텍스트 인식모델이 어텐션 사용

### 3. 모델 아키텍쳐



![image-20211201203145345](C:\Users\LeeChul-Ghi\AppData\Roaming\Typora\typora-user-images\image-20211201203145345.png)



#### 3.1 Thin Plate Spline

- 박판 스플라인 ( TPS )은 데이터 보간 및 평활화를 위한 스플라인 기반 기술
  - TPS(Thin Plate Spline) Jaderberg et al. (2015)는 보정 작업에서 이미지 인식을 위해 처음 제안
  - 나중에 이 모듈은 일부 텍스트 인식 모델에서 활용 Shi et al. (2019); Qiao et al. (2020).
    1) 계산 순서에 따라 먼저 로컬라이제이션 네트워크가 이미지를 가져와 공간 변환을 위한 매개변수를 생성
    2)  예측된 매개변수를 사용하여 변환된 출력을 생성하기 위해 입력 맵을 샘플링해야 하는 점 집합인 샘플링 그리드가 생성
    3) 마지막으로 샘플러는 그리드와 입력 이미지를 사용하여 출력 이미지를 생성

![Thin plate spline | Virtual Anthropology](http://www.virtual-anthropology.com/wp-content/uploads/2017/02/thinplatespline1.jpg)



#### 3.2 Backbone

ResNet의 학습 및 용량의 단순성으로 인해 많은 컴퓨터 비전 작업에 널리 사용

텍스트 인식 모델 또한 ResNet과 같은 네트워크 적용으로 좋은 결과를 보여주었음

ResNet 계열의 다양한 Beta 연구가 수행 [ASTER Shi et al. (2019), CSTR Cai et al. (2021), SEED Qiao et al. (2020), Baek et al. (2019). ]

해당 논문에서는 ImageNet Russakovsky et al. (2014) 에서 연구된 pre-Train ResNeXt-101 모델을 백본으로 사용

- ResNet / ResNeXt 차이

  https://benlee73.tistory.com/33

#### 3.4 Text Recognition Head

Krylov et al. (2021)  논문에서 사용한 문자 인식 해드 사용 

- 튜닝한 부분 
  - 먼저 ResNext-101(마지막 단계 제외)의 채널과 일치시키기 위해  컨볼루션 인코더에서 채널 수를 1024로 변경 
  - 헤드의 컨볼루션 인코더도 1024개의 출력 채널을 생성
  -  같은 이유로 기능 맵의 크기를 3 × 12로 변경
  - GRU 디코더를 활용하여 최종적으로 문자 인식

### 4. 실험

![image-20211201204957878](C:\Users\LeeChul-Ghi\AppData\Roaming\Typora\typora-user-images\image-20211201204957878.png)



#### 4.1 데이터 중요성을 강조한 위한 추가실험

가상 데이터만 가지고는 강력한 텍스트 인식 모델을 훈련시키기에 충분하지 않다는 것과 Real 데이터만 사용하는것도 최선은 아니라는 것을 실험으로 증명 



![image-20211201205221351](C:\Users\LeeChul-Ghi\AppData\Roaming\Typora\typora-user-images\image-20211201205221351.png)

Comparison of the recognition accuracy using different training datasets.