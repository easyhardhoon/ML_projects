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


