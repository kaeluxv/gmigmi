import torch, os, classify, utils, dataloader, torchvision, pickle
import numpy as np
import torch.nn as nn
import torchvision.utils as tvls
from utils import *

target_path = "./mcnn_dict_state.tar"
#checkpoint = torch.load(target_path)
#model.load_state_dict(checkpoint['state_dict'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file = "./MNIST.json"
args = load_json(json_file=file)
file_path = args['dataset']['0to9_file_path']
dataloader = init_dataloader(args, file_path, batch_size=1, mode ="gan")

class MyModel(nn.Module):
	def __init__(self, num_classes = 10):
		super(MyModel, self).__init__()
		self.feat_dim = 256
		self.num_classes = num_classes
		self.feature = nn.Sequential(
              nn.Conv2d(1, 64, 7, stride=1, padding=1),
              nn.BatchNorm2d(64),
              nn.LeakyReLU(0.2),
              nn.MaxPool2d(2, 2),
              nn.Conv2d(64, 128, 5, stride=1),
              nn.BatchNorm2d(128),
              nn.LeakyReLU(0.2),
              nn.MaxPool2d(2, 2),
              nn.Conv2d(128, 256, 5, stride=1),
              nn.BatchNorm2d(256),
              nn.LeakyReLU(0.2))
		self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
	
	def forward(self, x):
		x = self.feature[:4](x)
	#	x = x.view(x.size(0), -1)
	#	out = self.fc_layer(x)
	#	return out
		return x
		
my_model = MyModel(10).to(device)
#my_model = nn.DataParallel(my_model).cuda()
#checkpoint = torch.load(target_path)['state_dict']
#utils.load_my_state_dict(my_model, checkpoint)
my_model.load_state_dict(torch.load('./mcnn_dict_state.tar')['state_dict'])
my_model.eval()


for i, imgs in enumerate(dataloader):
	imgs = imgs.to(device)
	torch.set_printoptions(threshold=np.inf)
#	print("imgs : ", imgs)
#	print(imgs.dtype)
	output = my_model.forward(imgs)
	output1 = my_model.feature[4:](output)
	output1 = output1.view(output1.size(0),-1)
	output1 = my_model.fc_layer(output1)
	output1 = output1.data.max(1)[1]
#input = torch.randn(1,1, 32, 32)1
#	output = my_model.features[:8](input)
	print("output : ", output1)
#	print("output size : ", output.size())
	torch.save(output,'output.pt',pickle_module = pickle)


