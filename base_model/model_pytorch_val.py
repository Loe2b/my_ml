#time tracking
import time 

tmps = time.time()
def tmp():
	global tmps
	x = time.time() - tmps
	tmps = time.time()
	return x

#Import
import torch
from torch.utils import data
import torchvision as tv
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import os
from random import choice
from PIL import Image

print(" import time : %f" %(tmp()))

#Device set
if torch.cuda.is_available():
	device=torch.device("cuda:3")
else:
	device=torch.device("cpu")
print("using device %s"%(device))


#Image cropped
left=280
right=750
top=350
bottom=590

img_height = 240
img_width = 470
num_classes = 2

classes = ('valid', 'invalid')


images = [i for i in os.listdir("Validation_Pictures")]
img_name = choice(images)
print("image name is : %s"%(img_name))

# image = Image.open("Validation_Pictures/%s"%(img_name))
image = Image.open("Accepted_2.png")
image = image.crop((left, top, right, bottom))

plt.imshow(image)
#plt.show()

transform = tv.transforms.Compose([tv.transforms.ToTensor(),
	tv.transforms.Resize((100, 180)),
	tv.transforms.Grayscale(num_output_channels=1),
	tv.transforms.Normalize((0.5), (0.5))])

image = transform(image)
image = image.unsqueeze(0)

print(" image setup time : %f" %(tmp()))

#model
class Sequential(nn.Module):
	def __init__(self):
		super(Sequential,self).__init__()
		self.conv1=nn.Conv2d(1, 16, 3, padding="same")
		self.conv2=nn.Conv2d(16, 32, 3, padding="same")
		self.conv3=nn.Conv2d(32, 64, 3, padding="same")
		#self.conv4=nn.Conv2d(64, 128, 3)
		self.pool=nn.MaxPool2d(2, 2)
		#self.lin_size = 64*(img_height//8)*(img_width//8) #supposé juste...
		self.lin_size = 16896 
		self.fc1=nn.Linear(self.lin_size, 256)
		self.fc2=nn.Linear(256, 128)
		self.fc3=nn.Linear(128, 32)
		self.fc4=nn.Linear(32, num_classes)
	def forward(self, x):
		x=self.pool(F.relu(self.conv1(x)))
		x=self.pool(F.relu(self.conv2(x)))
		x=self.pool(F.relu(self.conv3(x)))
		#x=self.pool(F.relu(self.conv4(x)))
		x=x.view(x.size(0), -1)
		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=F.relu(self.fc3(x))
		return self.fc4(x)

model=torch.load("mymodel_robot_padding.pth", map_location=torch.device(device))
model.to(device)

print(" model setup time : %f" %(tmp()))


#Image cropped
left=280
right=750
top=350
bottom=590

img_height = 240
img_width = 470
num_classes = 2

classes = ('valid', 'invalid')


images = [i for i in os.listdir("Validation_Pictures")]
for img_name in images:
    print("image name is : %s"%(img_name))

    image = Image.open("Validation_Pictures/%s"%(img_name))
    image = image.crop((left, top, right, bottom))

    plt.imshow(image)
    #plt.show()

    image = transform(image)
    image = image.unsqueeze(0)

    print(" image setup time : %f" %(tmp()))


    image = image.to(device)
    outputs = model(image)

    sm = nn.Softmax(dim=1) 
    sm_outputs = sm(outputs) 
    proba, predic = torch.max(sm_outputs, 1)
    
    print("Name : %s"%(img_name))
    print("Prediction : %s with %i"%(classes[predic], proba*100), end="%.\n")