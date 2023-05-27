#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models.resnet import resnet18

device = "cuda" if torch.cuda.is_available() else "cpu"
#-------------------------------------------------------------------------------------
#1-1. make model + modify Classifier (uni-layer)
model = resnet18(pretrained=True) # ❶ vgg16 모델 객체 생성
num_output = 10; num_ftrs = model.fc.in_features   #512 . can access to "in_features"
#model.fc = nn.Linear(num_ftrs,num_output) #single
#summary(model,input_size=(3,224,224))
#-------------------------------------------------------------------------------------
#1-2. make model + modify Classifier (multi-layer) (use sequential class)
model = resnet18(pretrained=True) # ❶ vgg16 모델 객체 생성
fc = nn.Sequential( # ❷ 분류층의 정의
       nn.Linear(num_ftrs, 256),
       nn.ReLU(),
       nn.Linear(256, num_output),
   )
#---------------------------------------------------------
model.classifier = fc # ➍ VGG의 classifier를 덮어씀
model.to(device)
print(model)
summary(model,input_size=(3,224,224))
#---------------------------------------------------------

#2. train data preprocessing

# # 데이터 전처리와 증강
import tqdm
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam

transforms = Compose([
   Resize(224),
   RandomCrop((224, 224), padding=4),
   RandomHorizontalFlip(p=0.5),
   ToTensor(),
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])


training_data = CIFAR10(root="./", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

cifar10_classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



#---------------------------------------------------------------------------
#Test uni-image preprocessing

import numpy as np
import json
from torchvision.transforms import Normalize, CenterCrop,ToPILImage
from PIL import Image
import matplotlib.pyplot as plt

resize = 224
testImage_transforms = Compose([
   Resize(224),
   CenterCrop(224),
   ToTensor(),
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

image_file_path = './testImg/goldenretriever.jpg'
testImg = Image.open(image_file_path)
#resize = Resize(224)
#testImg = resize(testImg)
plt.imshow(testImg)
plt.show()

testImgTensor_transformed = testImage_transforms(testImg)  # torch.Size([3, 224, 224])
print(testImgTensor_transformed.shape)

#toPIL = ToPILImage()
#testImg_transformed = toPIL(testImgTensor_transformed)
testImgNumpy_transformed = testImgTensor_transformed.numpy().transpose((1, 2, 0))
#print(testImgNumpy_transformed.shape)
#print(testImgNumpy_transformed)
#testImg_transformed = np.clip(testImgNumpy_transformed, 0, 1)
#print(testImg_transformed)
plt.imshow(testImgNumpy_transformed)
plt.show()
#---------------------------------------------------------------------------

#train model

lr = 1e-3
optim = Adam(model.parameters(), lr=lr)

for epoch in range(10):
   iterator = tqdm.tqdm(train_loader) # ➊ 학습 로그 출력
   for data, label in iterator:
       optim.zero_grad()

       preds = model(data.to(device)) # 모델의 예측값 출력

       loss = nn.CrossEntropyLoss()(preds, label.to(device))
       loss.backward()
       optim.step()
     
       # ❷ tqdm이 출력할 문자열
       iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")
   print(f"epoch{epoch+1} loss:{loss.item()}")

torch.save(model.state_dict(), "CIFAR_pretrained_ResNet.pth") # 모델 저장
model.load_state_dict(torch.load("CIFAR_pretrained_ResNet.pth", map_location=device))
num_corr = 0

with torch.no_grad():
   for data, label in test_loader:
       output = model(data.to(device))
       _, preds = output.data.max(1)
       corr = preds.eq(label.to(device).data).sum().item()
       num_corr += corr

   print(f"Accuracy:{num_corr/len(test_data)}")


#----------------------------------------------------------------------------
# test model by uni-image

testImg = testImgTensor_transformed.unsqueeze(0)
print(testImg.shape)
output = model(testImg.to(device))
_,preds = output.data.max(1)
print(preds.shape)
obj_class = cifar10_classes[preds]
print(obj_class)

