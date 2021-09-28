import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm, trange

class mnist_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,32, kernel_size=3, padding=1)
        self.conv2=nn.Conv2d(32,64, kernel_size=3, padding=1)
        self.fc1=nn.Linear(7*7*64, 256)
        self.fc2=nn.Linear(256,10)
        
    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=F.max_pool2d(x, kernel_size=2)
        
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x, kernel_size=2)
        
        x=x.view(-1, 7*7*64)
        x=self.fc1(x)
        x=F.relu(x)
        
        x=self.fc2(x)
        return x
    
mnist_train= datasets.MNIST(root='./datasets',train=True, transform=transforms.ToTensor(),download=True)
mnist_test= datasets.MNIST(root='./datasets',train=False, transform=transforms.ToTensor(),download=True)

trainloader=T.utils.data.DataLoader(mnist_train , batch_size=100, shuffle=True)
testloader=T.utils.data.DataLoader(mnist_test , batch_size=100, shuffle=True)

model=mnist_cnn()

criterion= nn.CrossEntropyLoss()
optimizer=T.optim.Adam(model.parameters(), lr=0.001)

for i in trange(3):
    for images, labels in tqdm(trainloader):
        optimizer.zero_grad()
        
        x=images
        y=model.forward(x)
        
        loss=criterion(y, labels)
        
        loss.backward()
        optimizer.step()
        
correct=0
total=len(mnist_test)

with T.no_grad():
    for images, labels in tqdm(testloader):
        x=images
        y=model.forward(x)
        prediction=T.argmax(y, dim=1)
        correct+= T.sum((prediction==labels).float())
        
print("result --> ", correct/total)
print("no. of correct=", correct)

