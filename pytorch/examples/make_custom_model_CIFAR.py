#!/usr/bin/env python
# coding: utf-8


#label for object class
class_name = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize, ToTensor
import torch
import torch.nn as nn


class CNN(nn.Module):
   def __init__(self, num_classes): 
       super(CNN, self).__init__()

       #HOON
       #-----------------------------------------------------------------------
       self.conv1 = nn.Conv2d(in_channels = 3,out_channels=32, kernel_size=3, padding=1, stride = 1)
       self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
       self.conv2 = nn.Conv2d(in_channels = 32,out_channels=64, kernel_size=3, padding=1, stride = 1)
       self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 
       self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride = 1)
       self.fc1 = nn.Linear(in_features=4096, out_features=512)
       self.fc2 = nn.Linear(in_features=512, out_features=128)
       self.fc3 = nn.Linear(in_features=128, out_features=num_classes)
       self.relu = nn.ReLU()
       self.softmax = nn.Softmax()
       #----------------------------------------------------------------------- 

   def forward(self, x):
       #x = self.block1(x)
       #x = self.block2(x)
       #x = self.block3(x)  
       x = self.conv1(x)
       x = self.relu(x)
       x = self.pool1(x)
       x = self.conv2(x)
       x = self.relu(x)
       x = self.pool2(x)
       x = self.conv3(x)
       x = self.relu(x)
       x = torch.flatten(x, start_dim =1)  # --> for batch num
       x = self.fc1(x)
       x = self.relu(x)
       x = self.fc2(x)
       x = self.relu(x)
       x = self.fc3(x)
       #x = self.softmax(x)   ----> same role of CrossEntropyLoss 

       return x



from torch.utils.data.dataloader import DataLoader

from torch.optim.adam import Adam

transforms = Compose([
   RandomCrop((32, 32), padding=4),  # ❶ 랜덤 크롭핑
   RandomHorizontalFlip(p=0.5),  # ❷ y축으로 뒤집기
   ToTensor(),  # ❸ 텐서로 변환
   # ❹ 이미지 정규화
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])


# # 데이터 로드 및 모델 정의

# In[32]:

# ❶ 학습 데이터와 평가 데이터 불러오기
training_data = CIFAR10(root="./", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./", train=False, download=True, transform=transforms)

# ❷ 데이터로더 정의
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# ❸ 학습을 진행할 프로세서 설정
device = "cuda" if torch.cuda.is_available() else "cpu"


# ➍ CNN 모델 정의
model = CNN(num_classes=10)

# ➎ 모델을 device로 보냄
model.to(device)
print(model)
#model.block1.conv1 = nn.Conv2d(3,16,kernel_size=5, stride=1, padding=2) # !!!
#print(model)


# # 모델 학습하기
import tqdm

# ❶ 학습률 정의
lr = 1e-3

# ❷ 최적화 기법 정의
optim = Adam(model.parameters(), lr=lr)

# 학습 루프 정의
for epoch in range(100):   #100
   iterator = tqdm.tqdm(train_loader)
   for data, label in train_loader:  # ➌ 데이터 호출
       optim.zero_grad()  # ➍ 기울기 초기화

       preds = model(data.to(device))  # ➎ 모델의 예측

       # ➏ 오차역전파와 최적화
       loss = nn.CrossEntropyLoss()(preds, label.to(device)) 
       loss.backward() 
       optim.step() 
       iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")

   print(f"epoch{epoch+1} loss:{loss.item()}")


# 모델 저장
torch.save(model.state_dict(), "CIFAR.pth")
model.load_state_dict(torch.load("CIFAR.pth", map_location=device))
num_corr = 0

with torch.no_grad():
   for data, label in test_loader:

       output = model(data.to(device))
       preds = output.data.max(1)[1]
       corr = preds.eq(label.to(device).data).sum().item()
       num_corr += corr

   print(f"Accuracy:{num_corr/len(test_data)}")
   
for i in range(9):
   plt.subplot(3, 3, i+1)
   plt.title(f"class:" + class_name[training_data.targets[i]])
   plt.imshow(training_data.data[i])
plt.show()


