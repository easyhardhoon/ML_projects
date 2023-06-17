#make object detection model
1. can use any DNN model by backbone network (should tuning before)
2. result data should contains **Classfication**  and **localization** data
3. ----> Id, xmin, ymin, xmax, ymax, confidence score, class
   ----> [object num],[b_box localization data],[b_box object_is_exist score],[object class score]

#role
1. preprocessing
2. normalization (data augmentation, dropout, weight decay(ridge, lasso), batch normalization)
3. design DNN (backbone, classifier(NOT))
4. predict algo (optimizer, hyperparameters...)

#NOTE
1. only accuracy, not inference time
2. maybe ssd or R-CNN? 

#NOTE
1. ~/24 model make 
2. yolo, ssd, R-CNN / maybe change some layers "focued on invoke time" to "focused on accuracy"
3. don't rely on chatgpt...

#230516
-----------------------------------------------
<YOLO> <backbone + classfier >
yolov1: DarkNet-24(tuning googlenet. conv*24 + FC*2)
yolov2: Darknet-19(conv*19 + pooling*5)
yolov3: Darknet-53(similar to ResNet. conv*53)
yolov4: CSPDarknet53(cross stage . split output character map & merge)

+ 1*1 conv is not role of Classifier. just reduce parameter , FLOPS, channel size
  used in every yolo series. 
+ maybe use yolov 1 (tuning goolgenet focued on yolov1's darknet 24)
-----------------------------------------------
<R-CNN> <backbone + classifier>

----------------------------------------------
<SSD> <backbone>

----------------------------------------------
#FLOW
1. make_folders_and_data_downloads.py
2. Dataset_DataLoader.py
3. SSD_model_forward.py + loss_function.py
4. SSD_training.py 
5. SSD_inference.py 

#NOTE
1. this model is "SSD" , using VGGnet as backbone
2. the only thing to modify code is "SSD_model_forward.py" 
3. in SSD_model_forward.py, just modify [make_vgg func, make_extras func]

#IDEA
1. replace specific layer which helps to improve inferece time with other layer 
   which helps to improve accuracy
2. maybe focused on CONV layer + Pool layer


#SSD
SSD Network(model) : vgg(backbone) + extras(+@) + L2Norm + make_loc_conf
Should focus on "Class SSD" , line 500~
Feature map :  vgg[:4.3] + L2Norm / vgg[:] / vgg + extras[1/4] / vgg + extras [2/4] /                                        vgg + extras[3/4] / vgg + extras [4/4]
two branch : localization & Classfication.. loc.append(Featuremap) / conf.append(Featuremap)

#ISSUE
++ NN s
backbone  : 그냥 다른 모델로 갈아끼우는 방법 혹은 약간의 vgg 튜닝 (하)
extras : yolo에 비해 우수한 정확도에 기여. feature map 다수 생성 . 구조 수정(중)
L2Norm : 가장 부실한 첫번째 feature map을 정규화 하는 계층. 해볼 수 있을듯 (중)
make_loc_conf :  이건 6가지 feature map을 lc& cl branch임 건드리는게 아님

++ Plan s
change backbone VGGmodel (안의 내부 구조 약간씩 수정)
change other backbone model ( 그냥 다른 모델로 백본 모델 변경)
change L2Norm ( 1이든 2이든 사실상 L2Norm 은 반드시 변경해야함)
change extras (여기는 +@. 1,2,3의 방식과 유사성이 있어야함)
즉, backbone, L2Norm, extras 3개의 NN을 consistency에 맞게 튜닝하기.


#----------------------------------------------------------------------------------

**Summary**

1. pdf 처럼, 총 6가지 방법으로  custom ssd 제작을 하였다. 
2. 다양한 방법들을 조합하여 3가지 approach(브랜치)로 나누어 진행하였다.
3. 총 6가지 방법 중 뒷 5가지 방법을 주로 이용하였지만 결과적으론 데이터 전처리를 확실히 하는것이
   급격한 성능 상승을 이룰 수 있었다. data augmentation 을 많이 해보지 못한 것이 아쉽다
4. 기존 vanilla ssd 도 훌륭하였기 때문에 mAP를 올릴려면 추론시간이 더욱 오래 걸릴 수 밖에 없게 하는     것이 맞다고 생각하였다. 
5. 하지만 결과적으로 마지막 피쳐맵을 제거하는 것이 효과적이었다. 크기가 1인 피쳐맵이다 보니 정확도를    떨어뜨리게 하는 요인일 수 있었다고 생각한다. 
6. 즉 피쳐맵을 증가시커서 얻는 표본의 증가의 이점보다 크기가 1이다 보니 정교하지 않은 데이터의 불균형   이 더 컸다고 생각한다.
7. 이 문제는 이미지 사이즈를 512로 늘리고 데이터 증강을 이용하여 학습 데이터를 늘리는 것으로 더욱       좋아졌을 것이다.
8. 시간관계상 위의 접근법을 못해봐서 아쉽다. 하지만 우린 mAP 수학공식을 분석하여 색다른 접근법을 활용   하였다.
9. 결과적으로 마지막 피쳐맵 제거 + conf_thresh 변경까지만 한 것이 mAP 72로 가장 높았다.
10.  마지막 피쳐맵 제거 + 입력 이미지 사이즈 증가 512 + 데이터 증강 (데이터 전처리) + optimal conf_thresh의 조합이면 가장 mAP가 높을 것으로 기대하며 이번 프로젝트를 마친다.
