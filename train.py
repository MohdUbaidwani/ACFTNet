from __future__ import print_function
import argparse
import os
from math import log10
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from utils import save_img,VGGPerceptualLoss,torchPSNR,ssim
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import cv2
import kornia
from torchvision.transforms.functional import pad
import torchvision.transforms as transforms
import torch.nn.functional as F
from network1 import define_G, define_D, GANLoss, get_scheduler, update_learning_rate,rgb_to_y,ContrastLoss
from MODEL.fdfp_1__t9_fus_g2 import mymodel
from data import get_training_set, get_test_set
#from visualisation import disp_error_img, save_images


# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=False, default = './uw_data/', help='facades')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--finetune', default=False, help='to finetune')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=300, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=300, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=500, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true',default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--edge_loss', default=False, help='apply edge loss for training')
parser.add_argument('--delta',help='for huber loss',default=0.5)#
parser.add_argument('--edge_loss_type', default='sobel', help='apply canny or sobel loss loss for training')
opt = parser.parse_args()

print(opt)
if opt.cuda and not torch.cuda.is_available():
	raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
	torch.cuda.manual_seed(opt.seed)

device = torch.device("cuda:0" if opt.cuda else "cpu")


print('===> Building models')

if opt.finetune is True :
	G_path = "checkpoint/uw_data/netG_model_epoch_{}.pth".format(opt.epoch_count)
	my_net = torch.load(G_path).cuda(device = 0)
	
else:            
	
	my_net = mymodel().to(device)
	
	my_net = my_net.cuda(device = 0)
	print("yes loaded on cuda")

print('===> Loading datasets')
root_path = "uw_data/"
train_set = get_training_set(opt.dataset)
test_set = get_test_set(opt.dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

class Gradient_Loss(nn.Module):
    def __init__(self):
        super(Gradient_Loss, self).__init__()

        kernel_g = [[[0,1,0],[1,-4,1],[0,1,0]],
                    [[0,1,0],[1,-4,1],[0,1,0]],
                    [[0,1,0],[1,-4,1],[0,1,0]]]
        kernel_g = torch.FloatTensor(kernel_g).unsqueeze(0).permute(1, 0, 2, 3)
        self.weight_g = nn.Parameter(data=kernel_g, requires_grad=False)

    def forward(self, x,xx):
        grad = 0
        y = x
        yy = xx
        gradient_x = F.conv2d(y, self.weight_g,groups=3)
        gradient_xx = F.conv2d(yy,self.weight_g,groups=3)
        l = nn.L1Loss()
        a = l(gradient_x,gradient_xx)
        grad = grad + a
        return grad

Gradient_Loss=Gradient_Loss().to(device)
criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)
L_per = VGGPerceptualLoss().to(device)  
# setup optimizer
optimizer_g = torch.optim.Adam(my_net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
##
loss_func = nn.L1Loss()
loss_mse_func = nn.MSELoss()
##
print('parameters of model are',sum(dict((p.data_ptr(), p.numel()) for p in my_net.parameters()).values()))
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
	# train
	for iteration, batch in enumerate(training_data_loader, 1):
		
		rgb, tar ,train_image_name= batch[0].to(device), batch[1].to(device),batch[2]#1,3,256,256
		
		fake_b = my_net(rgb)#1,3,256,256
		
		fake_b1=fake_b[0].detach().squeeze(0).cpu()#3,256,256

		######################
		# (2) Update network
		######################

		optimizer_g.zero_grad()
		loss_g_l1=0.25*criterionL1(tar,fake_b)+0.9*L_per(fake_b,tar)+0.7*(1-ssim(fake_b,tar,255))+0.3*Gradient_Loss(fake_b,tar)
		
		####edg_loss###33
		if opt.edge_loss is True:

			if opt.edge_loss_type == 'canny':

				################################################## Adding Edge Loss ##################################
				edge_out1 = kornia.filters.canny(fake_b, low_threshold=0.1, high_threshold=0.2, kernel_size=(5, 5), sigma=(1, 1), hysteresis=True, eps=1e-06)
				edge_gt = kornia.filters.canny(tar, low_threshold=0.1, high_threshold=0.2, kernel_size=(5, 5), sigma=(1, 1), hysteresis=True, eps=1e-06)
				edge_loss_p1_magnitude = criterionL1(edge_out1[1], edge_gt[1]) 
				

			else:


				############################################### Sobel Loss ###########################################
				edge_out1 = kornia.filters.sobel(fake_b, normalized=True, eps=1e-06)
				edge_gt = kornia.filters.sobel(tar, normalized=True, eps=1e-06)
				
				edge_loss_p1_magnitude = criterionL1(edge_out1[0], edge_gt[0]) 



			edge_loss = edge_loss_p1_magnitude 
			
			loss_g = loss_g_l1  + 1*edge_loss 

		else:
			loss_g = loss_g_l1

		

		
		
		loss_g.backward()

		optimizer_g.step()

		if iteration % 1000==0:
			out_image=torch.cat((rgb, fake_b,tar), 3)
			
			out_image = out_image[0].detach().squeeze(0).cpu()
			print("===> Epoch[{}]({}/{}): Loss_G: {:.4f} ".format(
				epoch, iteration, len(training_data_loader), loss_g.item() ))
			if not os.path.exists("train_image"):
				os.mkdir("train_image")
			save_img(out_image, 'train_image/'+train_image_name[0])#rgb input


	update_learning_rate(net_g_scheduler, optimizer_g)
     

     #test
	
	if not os.path.exists("test_images"):
		os.mkdir("test_images")

	for test_iter, batch in enumerate(testing_data_loader,1):

		rgb_input, target,image_name= batch[0].to(device), batch[1].to(device), batch[2]
		prediction = my_net(rgb_input)
		out=torch.cat((rgb_input,prediction,target),3)
		output_cat =out[0].detach().squeeze(0).cpu() 
		save_img(output_cat, 'test_images/'+image_name[0])
	#checkpoint
	if epoch % 1== 0:
		if not os.path.exists("checkpoint"):
			os.mkdir("checkpoint")#tu bana do
		if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
			os.mkdir(os.path.join("checkpoint", opt.dataset))
		net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
		torch.save(my_net, net_g_model_out_path)
		print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))
