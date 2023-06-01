import torch, os, time, random, generator, discri, classify, utils
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls
from utils import *
import torch.optim as optim

device = "cuda"
num_classes = 10
log_path = "../attack_logs"
os.makedirs(log_path, exist_ok=True)

def inversion(G, D, T, E, iden, lr=0.2, momentum=0.9, lamda=1, iter_times=2000, clip_range=1):
#	iden = iden.cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	bs = iden.shape[0]
#	print("iden_size : " ,iden.size())
#	print("bs: ",bs)

	G.eval()
	D.eval()
	T.eval()
	E.eval()

	max_score = torch.zeros(bs)
	max_iden = torch.zeros(bs)
	z_hat = torch.zeros(bs, 100)
	
	for random_seed in range(10):
		tf = time.time()
		
		torch.manual_seed(random_seed) 
		torch.cuda.manual_seed(random_seed) 
		np.random.seed(random_seed) 
		random.seed(random_seed)

		z = torch.randn(bs, 100).cuda().float()
		z.requires_grad = True
		v = torch.zeros(bs, 100).cuda().float()
		
		for i in range(iter_times):
			fake = G(z)
			label = D(fake)
			out = T(fake)	
			if z.grad is not None:
				z.grad.data.zero_()

			Prior_Loss = -label.mean()
			Iden_Loss = criterion(out, iden)
			Total_Loss = Prior_Loss + lamda * Iden_Loss
			
			Total_Loss.backward()			

			v_prev = v.clone()
			gradient = z.grad.data
			v = momentum * v - lr * gradient
			z = z + ( - momentum * v_prev + (1 + momentum) * v)
			z = torch.clamp(z.detach(), -clip_range, clip_range).float()
			z.requires_grad = True
			
			Prior_Loss_val = Prior_Loss.item()
			Iden_Loss_val = Iden_Loss.item()

			if (i+1) % 500 == 0:
				fake_img = G(z.detach())
				eval_prob = E(fake_img)[-1]
				eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
				acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs   
				print("z_num:{}\tIteration:{}\tE:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}".format(random_seed, i+1, eval_iden, Prior_Loss_val, Iden_Loss_val))
				root_path = "./Attack"
				save_img_dir = os.path.join(root_path, "GMI_imgs_mid")
				os.makedirs(save_img_dir, exist_ok=True)
				save_tensor_images(fake_img.detach(), os.path.join(save_img_dir, "attack_image{}_{}.png".format(random_seed,i+1)), nrow = 10)

#			if (i+1) % 100 == 0:				
#				print("total_loss :", Total_Loss)
#				print("z : ",  z.detach())
		
#		print("total_loss : ", min_loss)
		# save images
#		root_path = "./Attack"
#		save_img_dir = os.path.join(root_path, "GMI_imgs")
#		os.makedirs(save_img_dir, exist_ok=True)
#		save_tensor_images(fake.detach(), os.path.join(save_img_dir, "attack_image1_{}.png".format(iter_times)), nrow = 10)

#		score = T(fake)[-1]
#		eval_prob = E(fake)[-1]
#		eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
		
#		cnt = 0
#		for i in range(bs):
#			gt = iden[i].item()
#			if score[i, gt].item() > max_score[i].item():
#				max_score[i] = score[i, gt]
#				max_iden[i] = eval_iden[i]
#				z_hat[i, :] = z[i, :]
#			if eval_iden[i].item() == gt:
#				cnt += 1
#			
#		interval = time.time() - tf
#		print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))
#
#	correct = 0
#	for i in range(bs):
#		gt = iden[i].item()
#		if max_iden[i].item() == gt:
#			correct += 1

		
#	acc = correct * 1.0 / bs
#	print("Acc:{:.2f}".format(acc))

if __name__ == "__main__":
	target_path = "./mcnn_dict_state.tar"
	
	T = classify.MCNN(num_classes)
	T = nn.DataParallel(T).cuda()
	ckp_T = torch.load(target_path)['state_dict']
	new_state_dict={}
	for key, value in ckp_T.items():
		new_key = 'module.' + key
		new_state_dict[new_key] = value
	utils.load_my_state_dict(T, new_state_dict)
#	T.load_state_dict(torch.load(target_path)['state_dict'])
#	print("1")

	e_path = "./scnn_dict_state.tar"
	E = classify.SCNN(num_classes)
	E = nn.DataParallel(E).cuda()
	ckp_E = torch.load(e_path)['state_dict']
	new_state_dict={}
	for key,value in ckp_E.items():
		new_key = 'module.' + key
		new_state_dict[new_key] = value
	utils.load_my_state_dict(E, new_state_dict)
#	print("2")
	
	g_path = "./dasol/MNIST_G.tar"
	G = generator.GeneratorMNIST()
	G = nn.DataParallel(G).cuda()
	ckp_G = torch.load(g_path)['state_dict']
	utils.load_my_state_dict(G, ckp_G)
#	print("3")

	d_path = "./dasol/MNIST_D.tar"
	D = discri.DGWGAN32()
	D = nn.DataParallel(D).cuda()
	ckp_D = torch.load(d_path)['state_dict']
	utils.load_my_state_dict(D, ckp_D)
#	print("4")

	iden = torch.load('output.pt')
	iden = iden.to(device)
#	print(iden.size())
#	iden = torch.zeros(5)
#	for i in range(5):
#	    iden[i] = i+5
	
	inversion(G, D, T, E, iden)
