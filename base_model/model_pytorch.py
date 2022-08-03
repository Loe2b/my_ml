
#time tracking
import time 

fichier = open("Robot_time_padding.txt", "a")
fichier.write("\n")

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
from tqdm import tqdm as tqdm

x=tmp()
fichier.write("\n import time : %f" %(x))
print(" import time : %f" %(x))

#Device set
if torch.cuda.is_available():
	device=torch.device("cuda:3")
else:
	device=torch.device("cpu")
print("using device %s"%(device))


#Data_set
transform = tv.transforms.Compose([tv.transforms.Grayscale(num_output_channels=1),
	tv.transforms.Resize((100, 180)),
	tv.transforms.ToTensor(),
	tv.transforms.Normalize((0.5), (0.5))])
dataset = tv.datasets.ImageFolder('/grid_mnt/data__data.polcms/cms/sghosh/camdata/Augmented_dataset_bin/', transform=transform)

#split data for training/testing
train_size=int(0.9*len(dataset)) #90% for training
test_size=len(dataset)-train_size #10% for testing
train_set,test_set=data.random_split(dataset,[train_size,test_size])

print("len train_set : %i len test_set : %i"%(train_size, test_size))

batch_size = 500

train_loader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_set,batch_size=200,shuffle=True)

x=tmp()
fichier.write("\n loader setup time : %f" %(x))
print(" loader setup time : %f" %(tmp()))


classes = ('valid', 'invalid')


img_height = 240
img_width = 470
num_classes = 2

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

model=Sequential()
model.to(device)
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=0.001)

x=tmp()
fichier.write("\n model setup time : %f" %(x))
print(" model setup time : %f" %(tmp()))


nb_epoch=100

ptr_loss=[]
pte_loss=[]
pepoch=[]

train_time = 0
test_time = 0
loading_time = 0

for epoch in range(nb_epoch):
	train_cum_loss=0.0
	train_acc=0.0
	model.train() #activate autograd
	for x_train,y_train in tqdm(train_loader, desc="training..."):
		l_time = time.time()
		x_train, y_train = x_train.to(device), y_train.to(device)
		loading_time += time.time() - l_time
		optimizer.zero_grad()
		ypred=model(x_train)
		batch_loss=loss(ypred.squeeze(),y_train)
		batch_loss.backward() #calculate derivatives
		optimizer.step() #apply the optimizer step
		train_cum_loss+=batch_loss.item()
		sm = nn.Softmax(dim=1)
		sm_output = sm(ypred)
		sm_output = torch.argmax(sm_output, dim=1)

		train_acc += float(np.equal(sm_output.detach().cpu(),y_train.detach().cpu()).sum())/float(len(y_train.detach().cpu()))
		# print("num : ", np.equal(sm_output.detach().cpu(),y_train.detach().cpu()).sum())
		# print("len : ", len(y_train.detach().cpu()))
		# print(train_acc)

		# if batch_loss == 0:
		# 	print("Batch loss : ", batch_loss)
		# 	print("y pred : ", ypred)
		# 	print("y train : ", y_train)
		# 	sm = nn.Softmax(dim=1) 
		# 	sm_outputs = sm(ypred) 
		# 	print(sm_outputs)

	train_loss=train_cum_loss/len(train_loader)
	train_acc/=float(len(train_loader))
	t_time = tmp()
	train_time += t_time
	fichier.write("\n %i: Train loss: %f, train acc : %f, time : %f" %(epoch, train_loss, train_acc, t_time))
	print("%i: Train loss: %f, train acc : %f, time : %f" %(epoch, train_loss, train_acc, t_time))


	test_cum_loss=0.0
	test_acc=0.0
	model.eval() #deactivate autograd
	for x_test,y_test in tqdm(test_loader, desc="testing..."):
		l_time = time.time()
		x_test, y_test = x_test.to(device), y_test.to(device)
		loading_time += time.time() - l_time
		y_pred=model(x_test)
		batch_loss=loss(y_pred.squeeze(),y_test)
		test_cum_loss+=batch_loss.item()
		sm = nn.Softmax(dim=1)
		sm_output = sm(y_pred)
		sm_output = torch.argmax(sm_output, dim=1)
		test_acc += float(np.equal(sm_output.detach().cpu(),y_test.detach().cpu()).sum())/float(len(y_test.detach().cpu()))
	 
	test_loss=test_cum_loss/len(test_loader)
	test_acc/=float(len(test_loader))
	test_time += tmp()

	print("%i: Test loss: %f, test acc : %f, time : %f" %(epoch, test_loss, test_acc, test_time))

	ptr_loss.append(train_loss)
	pte_loss.append(test_loss)
	pepoch.append(epoch)

	tmp()

	if epoch % 10 == 0:
		torch.save(model,"mymodel_Robot_padding.%i.pth"%(epoch))



fichier.write("\n mean train time : %f" %(train_time / nb_epoch))
fichier.write("\n mean test time : %f" %(test_time / nb_epoch))
fichier.write("\n mean loading time : %f" %(loading_time / nb_epoch / 2))

print(" mean train time : %f" %(train_time / nb_epoch))
print(" mean test time : %f" %(test_time / nb_epoch))
print(" mean loading time : %f" %(loading_time / nb_epoch / 2))

#saving model
torch.save(model,"mymodel_Robot_padding.pth")
x=tmp()
fichier.write("\n model saving time : %f" %(x))
print(" model saving time : %f" %(x))

fichier.close()



#save values on a .txt
file = open("diagramme_val_Robot_padding.txt", "a")
file.write("\n")

for val in ptr_loss:
	file.write("%f " %(val))
file.write("|")
for val in pte_loss:
	file.write("%f " %(val))
file.write("|")
file.write("%i " %(nb_epoch))
file.write("|")

file.close()