from os import listdir
from os.path import join
import random
from PIL import Image ,ImageOps
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import is_image_file
from skimage import io, color
import torch
import cv2
from pytorch_msssim import ssim


def torchPSNR(tar_img, prd_img):
	imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
	rmse = (imdff**2).mean().sqrt()
	ps = 20*torch.log10(1/rmse)
	return ps

def torchSSIM(tar_img, prd_img):
	return ssim(tar_img, prd_img, data_range=1.0, size_average=True)

def transf(im):
  im = im.resize((256, 256), Image.BICUBIC)
  im=transforms.ToTensor()(im)
  im = transforms.Normalize((0.5),(0.5))(im)
  return im 	
class DatasetFromFolder(data.Dataset): 
	def __init__(self, image_dir,batch_size=1, shuffle=True, num_workers=0):
		super(DatasetFromFolder, self).__init__()
		self.a_path = join(image_dir, "a")                                                        
		self.b_path = join(image_dir, "b")
		self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]
        


	def __getitem__(self, index):

		input_rgb = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
		tar1 = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
		transform_list = transforms.Compose([transforms.ToTensor(),
						  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                     	  ])

		

		input_rgb = input_rgb.resize((256,256), Image.BICUBIC)
		tar1 = tar1.resize((256,256), Image.BICUBIC)
		input_rgb =transform_list(input_rgb)
		tar1 = transform_list(tar1)
		
		return input_rgb,tar1,self.image_filenames[index]

	def __len__(self):
		return len(self.image_filenames)



        